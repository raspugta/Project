"""BAYMAX engine: the orchestrator.

Pipeline for a user turn (each stage's output is recorded in a TurnTrace):

  Utterance ─► language id ─► NLU (intent + extraction)
            ─► SafetyInputs ─► safety state machine ─► actions
                 (say / timers / alarms / notify contacts / log events)
            ─► health-event DB ─► knowledge retrieval ─► deterministic composition
            ─► [optional] LLM enrichment ─► output guard ─► response

Perception samples and timers enter at the SafetyInput stage. The LLM runs
last, only outside emergency states, and can only append guarded sentences.
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Optional

from pydantic import BaseModel

from .contracts import (AudioLevel, BaymaxResponse, BodySample, HealthEvent, KnowledgeHit, Measurement, NLUResult,
                        PerceptionEvent, SafetyAction, SafetyInput, SafetyTransition, Statement, TurnTrace, Utterance)
from .conversation import compose, llm as llm_mod
from .knowledge.kb import KnowledgeBase, default_kb
from .language import phrases as ph
from .language.detect import detect_language, resolve_language
from .language.text import normalize
from .nlu.model import NLU
from .ontology import (RED_FLAG_CONCEPTS, MeasurementType, SUPPORTED_LANGS, Concept, ConsentScope, EventKind, Intent,
                       Modality, Provenance, SafetyState, Severity)
from .perception.pipeline import PerceptionHub
from .safety.escalation import Dispatcher, EscalationConfig, EscalationResult
from .safety.machine import SafetyMachine, SafetyPolicy
from .store.db import Store


class Profile(BaseModel):
    user_name: str = ""
    preferred_lang: str = "auto"
    require_wake_word: bool = True       # applies to continuous-listening speech only
    inactivity_minutes: float = 5.0
    llm_enabled: bool = True


# Intents that are processed from continuous listening even without the wake word.
SAFETY_INTENTS = {Intent.EMERGENCY_HELP, Intent.PAIN_EXCLAMATION, Intent.FALL_REPORT}
AWAITING_INTENTS = {Intent.USER_OK, Intent.CANCEL, Intent.CONFIRM, Intent.PAIN_REPORT}
LLM_INTENTS = {Intent.GREETING, Intent.SYMPTOM_REPORT, Intent.PAIN_REPORT, Intent.HEALTH_QUESTION,
               Intent.GOODBYE, Intent.MEDICATION_LOG, Intent.UNKNOWN}


def fmt_time(ts: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))


def fmt_measurement(m: Measurement) -> str:
    if m.value2 is not None:
        return f"{m.value:g}/{m.value2:g} {m.unit}"
    return f"{m.value:g} {m.unit}".replace(" /10", "/10")


class Engine:
    def __init__(self, store: Store, llm_backend: Optional[llm_mod.LLMBackend] = None,
                 dispatcher: Optional[Dispatcher] = None, kb: Optional[KnowledgeBase] = None,
                 clock: Callable[[], float] = time.time) -> None:
        self.store = store
        self.nlu = NLU()
        self.kb = kb or default_kb()
        self.llm = llm_backend
        self.dispatcher = dispatcher or Dispatcher()
        self.clock = clock
        self.lock = threading.RLock()
        self.profile = Profile(**store.get_setting("profile", {}))
        self.policy = SafetyPolicy(**store.get_setting("safety_policy", {}))
        self.escalation = EscalationConfig(**store.get_setting("escalation", {}))
        self.machine = SafetyMachine(self.policy)
        self.perception = PerceptionHub(inactivity_s=self.profile.inactivity_minutes * 60)
        self.session_lang: Optional[str] = None
        self.last_user_utt: Optional[str] = None
        self.observations: "OrderedDict[str, PerceptionEvent]" = OrderedDict()
        self.privacy = {"camera_paused": False, "mic_paused": False}
        self.listeners: list[Callable[[dict], None]] = []
        self.traces: "OrderedDict[str, TurnTrace]" = OrderedDict()
        self._told_no_store = False
        store.audit("system", "engine_started", llm=(self.llm.name if self.llm else None),
                    dry_run=self.escalation.dry_run)

    # ------------------------------------------------------------ settings
    def update_profile(self, **kw: Any) -> Profile:
        with self.lock:
            self.profile = self.profile.model_copy(update=kw)
            self.store.set_setting("profile", self.profile.model_dump())
            self.perception.activity.inactivity_s = self.profile.inactivity_minutes * 60
            return self.profile

    def update_policy(self, **kw: Any) -> SafetyPolicy:
        with self.lock:
            self.policy = SafetyPolicy(**{**self.policy.model_dump(), **kw})
            self.machine.policy = self.policy
            self.store.set_setting("safety_policy", self.policy.model_dump())
            return self.policy

    def update_escalation(self, cfg: EscalationConfig) -> EscalationConfig:
        with self.lock:
            self.escalation = cfg
            self.store.set_setting("escalation", cfg.model_dump())
            return cfg

    # --------------------------------------------------------------- utils
    def _emit(self, msg: dict) -> None:
        for fn in list(self.listeners):
            try:
                fn(msg)
            except Exception:
                pass

    def _lang_for_system_turn(self) -> str:
        if self.profile.preferred_lang in SUPPORTED_LANGS:
            return self.profile.preferred_lang
        return self.session_lang or "en"

    def _remember_trace(self, tr: TurnTrace) -> None:
        self.traces[tr.id] = tr
        while len(self.traces) > 200:
            self.traces.popitem(last=False)

    def _ui_hints(self, extra: Optional[dict] = None) -> dict:
        c = self.machine.ctx
        hints = {"state": c.state.value, "timer": c.timer_name, "deadline": c.timer_deadline,
                 "critical": c.critical, "reason": c.reason, "emergency_number": self.escalation.emergency_number,
                 "dry_run": self.escalation.dry_run}
        if extra:
            hints.update(extra)
        return hints

    # ============================================================ utterances
    def process_utterance(self, utt: Utterance) -> Optional[TurnTrace]:
        with self.lock:
            return self._process_utterance(utt)

    def _process_utterance(self, utt: Utterance) -> Optional[TurnTrace]:
        now = self.clock()
        if utt.modality == Modality.SPEECH and not self.store.has_consent(ConsentScope.MICROPHONE):
            self.store.audit("system", "speech_rejected_no_consent", utterance_id=utt.id)
            return None
        trace = TurnTrace(input={"type": "utterance", "id": utt.id, "modality": utt.modality.value,
                                 "stt_model_id": utt.stt_model_id, "stt_confidence": utt.stt_confidence,
                                 "wake_word": utt.wake_word})
        lang_res = detect_language(utt.text, prior=self.session_lang, stt_language=utt.stt_language)
        lang = resolve_language(lang_res, self.session_lang, self.profile.preferred_lang)
        ctx = self.machine.ctx
        nlu = self.nlu.analyze(utt, lang_res, lang, awaiting=ctx.awaiting)
        trace.nlu = nlu

        # Continuous listening without the wake word: only safety speech gets through.
        if utt.modality == Modality.SPEECH and self.profile.require_wake_word and not utt.wake_word:
            passes = (nlu.intent in SAFETY_INTENTS or
                      (ctx.awaiting is not None and nlu.intent in AWAITING_INTENTS) or
                      (ctx.state != SafetyState.MONITORING and nlu.intent in AWAITING_INTENTS))
            if not passes:
                self.store.audit("system", "speech_ignored_no_wake_word", utterance_id=utt.id, intent=nlu.intent.value)
                return None

        if lang_res.confidence >= 0.5 or self.session_lang is None:
            self.session_lang = lang
        self.last_user_utt = utt.id

        statements: list[Statement] = []
        hints: dict[str, Any] = {}

        # ---- 1. health events from what the user said (USER provenance)
        events = self._events_from_nlu(nlu, utt, lang)
        trace.events += events
        stored = {e.id: self.store.add_event(e) for e in events}

        # ---- 2. safety
        for inp in self._safety_inputs(nlu, utt):
            trace.safety_inputs.append(inp)
            tr = self.machine.feed(inp, now)
            if tr:
                trace.transitions.append(tr)
                statements += self._execute(tr, lang, hints, trace, pain_event=self._first(events, EventKind.PAIN, stored))

        # ---- 3. content
        statements += self._content(nlu, utt, lang, events, stored, trace, safety_said=bool(statements), hints=hints)

        # ---- 4. optional LLM enrichment (never in emergency states)
        state_after = self.machine.ctx.state
        if (self.llm and self.profile.llm_enabled and nlu.intent in LLM_INTENTS and not nlu.injection_suspected
                and state_after in (SafetyState.MONITORING, SafetyState.ASSISTING)
                and (not self.llm.is_cloud or self.store.has_consent(ConsentScope.CLOUD_LLM))):
            statements += self._llm(utt, lang, statements, events, stored, trace)

        if not statements:
            statements.append(compose.phrase("unknown", lang))

        resp = BaymaxResponse(lang=lang, statements=statements, safety=self.machine.ctx, ui_hints=self._ui_hints(hints))
        trace.response = resp
        self.store.add_utterance(utt.id, utt.text, lang, nlu.intent.value, role="user")
        self.store.add_utterance(resp.id, resp.text, lang, nlu.intent.value, role="baymax")
        self.store.audit("engine", "turn", trace_id=trace.id, utterance_id=utt.id, lang=lang,
                         intent=nlu.intent.value, intent_confidence=nlu.intent_confidence,
                         rules=[t.rule for t in trace.transitions],
                         provenance=[s.provenance.value for s in statements],
                         flags=[n for n in nlu.notes if n in ("injection_suspected", "private_data_request", "fake_marker")])
        self._remember_trace(trace)
        self._emit({"type": "response", "trace_id": trace.id, "response": resp.model_dump(mode="json")})
        return trace

    # ---------------------------------------------------------------------
    @staticmethod
    def _first(events: list[HealthEvent], kind: EventKind, stored: dict[str, bool]) -> Optional[str]:
        for e in events:
            if e.kind == kind:
                return e.id if stored.get(e.id) else None
        return None

    def _safety_inputs(self, nlu: NLUResult, utt: Utterance) -> list[SafetyInput]:
        out: list[SafetyInput] = []
        red = [c for c in nlu.concepts if c.concept in RED_FLAG_CONCEPTS and not (c.negated or c.hypothetical or c.past)]
        subject = "other" if "subject_other" in nlu.notes else "self"
        base = dict(source="nlu", ref_id=utt.id, confidence=nlu.intent_confidence)
        if nlu.intent == Intent.EMERGENCY_HELP:
            if nlu.intent_confidence >= 0.9:
                out.append(SafetyInput(kind="critical_utterance", severity=Severity.CRITICAL,
                                       detail={"concepts": [c.concept.value for c in red], "subject": subject,
                                               "reason": "critical"}, **base))
            else:
                out.append(SafetyInput(kind="possible_emergency", severity=Severity.HIGH,
                                       detail={"fake": "fake_marker" in nlu.notes}, **base))
        elif "red_flag_uncertain" in nlu.notes:
            out.append(SafetyInput(kind="possible_emergency", severity=Severity.HIGH,
                                   detail={"red_flag_uncertain": True}, **base))
        elif Intent.EMERGENCY_HELP in nlu.secondary_intents:
            out.append(SafetyInput(kind="possible_emergency", severity=Severity.HIGH, detail={}, **base))
        if nlu.intent == Intent.FALL_REPORT:
            head = any(c.concept == Concept.HEAD_INJURY and not c.negated for c in nlu.concepts)
            out.append(SafetyInput(kind="fall_reported", severity=Severity.HIGH,
                                   detail={"cannot_get_up": "cannot_get_up" in nlu.notes, "head_injury": head}, **base))
        if nlu.intent == Intent.PAIN_EXCLAMATION:
            out.append(SafetyInput(kind="pain_exclamation", severity=Severity.MODERATE, **base))
        if nlu.intent == Intent.PAIN_REPORT or Intent.PAIN_REPORT in nlu.secondary_intents:
            out.append(SafetyInput(kind="pain_report", severity=Severity.MODERATE,
                                   detail={"pain_score": nlu.pain_score}, **base))
        if nlu.intent == Intent.USER_OK:
            out.append(SafetyInput(kind="user_ok", **base))
        elif nlu.intent == Intent.CANCEL:
            out.append(SafetyInput(kind="user_cancel", **base))
        elif nlu.intent == Intent.CONFIRM:
            out.append(SafetyInput(kind="user_confirm", **base))
        return out

    def _events_from_nlu(self, nlu: NLUResult, utt: Utterance, lang: str) -> list[HealthEvent]:
        evs: list[HealthEvent] = []
        common = dict(provenance=Provenance.USER, source_ref=utt.id, lang=lang, model_id=nlu.model_id,
                      confidence=nlu.intent_confidence)
        affirmed = [c for c in nlu.concepts if not (c.negated or c.hypothetical)]
        pain = [c for c in affirmed if c.concept in (Concept.PAIN, Concept.HEADACHE, Concept.SORE_THROAT)]
        if pain or (nlu.pain_score is not None and nlu.intent == Intent.PAIN_REPORT):
            score = nlu.pain_score
            sev = Severity.HIGH if (score or 0) >= 8 else Severity.MODERATE if (score is None or score >= 5) else Severity.LOW
            regions = list(dict.fromkeys([c.body_region.value for c in pain if c.body_region] + [b.value for b in nlu.body_regions]))
            evs.append(HealthEvent(ts=self.clock(), kind=EventKind.PAIN, severity=sev, concepts=[c.concept.value for c in pain] or ["pain"],
                                   body_regions=regions, pain_score=score,
                                   summary=f"pain{f' {score}/10' if score is not None else ''}{' ' + ','.join(regions) if regions else ''}",
                                   **common))
        other = [c for c in affirmed if c.concept not in (Concept.PAIN, Concept.HEADACHE, Concept.SORE_THROAT, Concept.FALL)
                 and c.concept not in RED_FLAG_CONCEPTS and not c.past]
        if other:
            evs.append(HealthEvent(ts=self.clock(), kind=EventKind.SYMPTOM, severity=Severity.LOW, concepts=[c.concept.value for c in other],
                                   body_regions=[c.body_region.value for c in other if c.body_region],
                                   summary="symptoms: " + ",".join(c.concept.value for c in other), **common))
        for m in nlu.measurements:
            evs.append(HealthEvent(ts=self.clock(), kind=EventKind.MEASUREMENT, severity=Severity.INFO, measurement=m,
                                   summary=f"{m.type.value} {fmt_measurement(m)} (user reported: '{m.raw}')", **common))
        if nlu.intent == Intent.MEDICATION_LOG and nlu.medication:
            evs.append(HealthEvent(ts=self.clock(), kind=EventKind.MEDICATION_TAKEN, severity=Severity.INFO, medication=nlu.medication,
                                   summary=f"took {nlu.medication}", **common))
        return evs

    # -------------------------------------------------------------- actions
    def _execute(self, tr: SafetyTransition, lang: str, hints: dict, trace: Optional[TurnTrace],
                 pain_event: Optional[str] = None) -> list[Statement]:
        inp = tr.input
        self.store.audit("safety", "transition", rule=tr.rule, before=tr.before.value, after=tr.after.value,
                         input=inp.kind, source=inp.source, ref=inp.ref_id, confidence=inp.confidence)
        out: list[Statement] = []
        number = self.escalation.emergency_number
        for a in tr.actions:
            if a.kind == "say":
                out.append(self._say(a, lang, inp, number, pain_event))
            elif a.kind == "notify_contacts":
                out += self._notify(a, lang, trace)
            elif a.kind == "log_event":
                self._log_event(a, inp, lang, trace)
            elif a.kind == "sound_alarm":
                hints["alarm"] = a.params.get("level", "loud")
            elif a.kind == "stop_alarm":
                hints["alarm"] = "off"
        return out

    def _say(self, a: SafetyAction, lang: str, inp: SafetyInput, number: str, pain_event: Optional[str]) -> Statement:
        key = a.phrase_key or "unknown"
        params = {"number": number, **a.params}
        if key in ph.INFERENCE_PHRASES:
            if inp.source == "perception" and inp.ref_id:
                obs = self.observations.get(inp.ref_id)
                return compose.phrase(key, lang, ref=f"obs:{inp.ref_id}", model_id=obs.model_id if obs else None,
                                      confidence=inp.confidence, **params)
            return compose.phrase(key, lang, ref=f"utt:{inp.ref_id}", model_id=self.nlu.model_id,
                                  confidence=inp.confidence, **params)
        if key == "pain_logged":
            return compose.phrase(key, lang, ref=f"event:{pain_event}" if pain_event else f"utt:{inp.ref_id}", **params)
        return compose.phrase(key, lang, **params)

    def _notify(self, a: SafetyAction, lang: str, trace: Optional[TurnTrace]) -> list[Statement]:
        level = a.params.get("level", "emergency")
        reason = a.params.get("reason") or ""
        res: EscalationResult = self.dispatcher.dispatch(
            self.escalation, level, reason,
            contacts_consent=self.store.has_consent(ConsentScope.EMERGENCY_CONTACTS),
            share_details=self.store.has_consent(ConsentScope.SHARE_HEALTH_IN_ALERTS))
        self.store.add_escalation(res.id, reason, level, res.status,
                                  {"results": [r.model_dump() for r in res.results]})
        self.store.audit("safety", "escalation_dispatched", escalation_id=res.id, level=level, reason=reason,
                         status=res.status, channels=[{"contact": r.contact_id, "ok": r.ok} for r in res.results])
        ev = HealthEvent(ts=self.clock(), kind=EventKind.ESCALATION, severity=Severity.CRITICAL if level == "emergency" else Severity.HIGH,
                         summary=f"alert {level}: {res.status}", provenance=Provenance.SYSTEM, source_ref=res.id, lang=lang)
        self.store.add_event(ev, emergency=True)
        if trace is not None:
            trace.events.append(ev)
        self._emit({"type": "escalation", "result": res.model_dump(mode="json")})
        if level == "all_clear":
            return []
        key = {"sent": "escalated", "partial": "escalated", "dry_run": "escalated_dry_run",
               "no_contacts": "no_contacts", "no_consent": "no_contacts", "failed": "escalation_failed"}[res.status]
        return [compose.phrase(key, lang)]

    def _log_event(self, a: SafetyAction, inp: SafetyInput, lang: str, trace: Optional[TurnTrace]) -> None:
        p = dict(a.params)
        kind = EventKind(p.pop("kind"))
        sev = Severity(p.pop("severity"))
        # perception -> model inference; speech/UI -> the user; timers -> the safety machine itself
        prov = {"perception": Provenance.MODEL, "nlu": Provenance.USER, "ui": Provenance.USER}.get(inp.source, Provenance.SYSTEM)
        obs = self.observations.get(inp.ref_id or "") if inp.source == "perception" else None
        model_id = obs.model_id if obs else (self.nlu.model_id if inp.source == "nlu" else None)
        ev = HealthEvent(ts=self.clock(), kind=kind, severity=sev, concepts=[str(c) for c in p.get("concepts", [])],
                         summary=f"{kind.value}: " + ", ".join(f"{k}={v}" for k, v in p.items() if k != "concepts"),
                         provenance=prov, source_ref=inp.ref_id, model_id=model_id,
                         confidence=inp.confidence if prov != Provenance.SYSTEM else None, lang=lang)
        self.store.add_event(ev, emergency=kind in (EventKind.EMERGENCY, EventKind.ESCALATION, EventKind.FALL_DETECTED))
        if trace is not None:
            trace.events.append(ev)

    # -------------------------------------------------------------- content
    def _content(self, nlu: NLUResult, utt: Utterance, lang: str, events: list[HealthEvent], stored: dict[str, bool],
                 trace: TurnTrace, safety_said: bool, hints: dict) -> list[Statement]:
        out: list[Statement] = []
        I = Intent
        intent = nlu.intent
        stored_any = any(stored.values())
        if nlu.injection_suspected:
            out.append(compose.phrase("cannot_follow", lang))
        if "private_data_request" in nlu.notes:
            out.append(compose.phrase("cannot_share_private", lang))
        if "fake_marker" in nlu.notes:
            out.append(compose.phrase("fake_emergency", lang))

        def add_kb(concepts: list[str], query: str = "", k: int = 1) -> None:
            hits: list[KnowledgeHit] = []
            for c in concepts:
                for e in self.kb.by_concept(c):
                    if e.id not in {h.entry_id for h in hits}:
                        hits.append(self.kb.hit(e.id, lang, 10.0))
                    break
            if not hits and query:
                ranked = self.kb.retrieve(query, lang, concepts, k=2)
                if ranked and ranked[0].score >= 3.0 and (len(ranked) == 1 or ranked[0].score >= 1.5 * ranked[1].score):
                    hits = ranked[:1]
            for h in hits[:k]:
                trace.knowledge.append(h)
                out.append(compose.knowledge(h))

        affirmed = [c for c in nlu.concepts if not c.negated and not c.past]
        red_now = [c.concept.value for c in affirmed if c.concept in RED_FLAG_CONCEPTS and not c.hypothetical]

        if intent == I.EMERGENCY_HELP:
            if red_now:
                add_kb(red_now, k=1)
            return out
        if intent == I.PRIVACY_COMMAND:
            act = nlu.privacy_action
            hints["privacy"] = act
            if act == "pause_camera":
                self.privacy["camera_paused"] = True
                out.append(compose.phrase("privacy_camera_paused", lang))
            elif act == "pause_mic":
                self.privacy["mic_paused"] = True
                out.append(compose.phrase("privacy_mic_paused", lang))
            elif act == "privacy_mode":
                self.privacy.update(camera_paused=True, mic_paused=True)
                out.append(compose.phrase("privacy_mode_on", lang))
            elif act == "resume":
                self.privacy.update(camera_paused=False, mic_paused=False)
                out.append(compose.phrase("privacy_resumed", lang))
            elif act == "forget_last":
                prev = self._previous_user_utterance(utt.id)
                n = self.store.delete_events_from(prev) if prev else 0
                out.append(compose.phrase("forgot_last" if n else "nothing_to_forget", lang))
            elif act == "delete_all":
                out.append(compose.phrase("delete_confirm_ui", lang))
            self.store.audit("user", "privacy_command", privacy_action=act)
            return out

        # symptom / pain logging acknowledgement
        sym = [e for e in events if e.kind == EventKind.SYMPTOM]
        if sym:
            if stored.get(sym[0].id):
                names = ", ".join(ph.name("c:" + c, lang) for c in sym[0].concepts)
                out.append(compose.phrase("symptom_logged", lang, ref=f"event:{sym[0].id}", what=names))
        needs_store_notice = [e for e in events if not stored.get(e.id)]
        if needs_store_notice and not self._told_no_store:
            out.append(compose.phrase("store_no_consent", lang))
            self._told_no_store = True

        if intent in (I.PAIN_REPORT, I.SYMPTOM_REPORT, I.FALL_REPORT):
            concepts = [c.concept.value for c in affirmed if not c.hypothetical and c.concept not in (Concept.PAIN,)]
            add_kb(concepts, k=2)
            if intent == I.SYMPTOM_REPORT and not safety_said:
                out.append(compose.phrase("seek_care_if_worse", lang))
        elif intent == I.PAIN_EXCLAMATION:
            pass
        elif intent == I.VITAL_REPORT:
            for e in events:
                if e.kind == EventKind.MEASUREMENT and e.measurement:
                    ref = f"event:{e.id}" if stored.get(e.id) else f"utt:{utt.id}"
                    out.append(compose.phrase("measurement_logged", lang, ref=ref,
                                              what=ph.name(e.measurement.type.value, lang), value=fmt_measurement(e.measurement)))
            for n in nlu.notes:
                if n.startswith("implausible:"):
                    out.append(compose.phrase("measurement_implausible", lang, what=ph.name(n.split(":")[1], lang)))
        elif intent == I.MEDICATION_LOG:
            for e in events:
                if e.kind == EventKind.MEDICATION_TAKEN:
                    ref = f"event:{e.id}" if stored.get(e.id) else f"utt:{utt.id}"
                    out.append(compose.phrase("med_logged", lang, ref=ref, med=e.medication))
        elif intent == I.MEMORY_STATEMENT and nlu.memory_fact:
            mid = self.store.add_memory(nlu.memory_fact["kind"], nlu.memory_fact["value"], utt.id, lang)
            if mid:
                out.append(compose.phrase("memory_saved", lang, ref=f"memory:{mid}", value=nlu.memory_fact["value"]))
            else:
                out.append(compose.phrase("memory_no_consent", lang))
        elif intent == I.RECALL_QUERY:
            out += self._recall(nlu, utt, lang)
        elif intent == I.HEALTH_QUESTION:
            if "diagnosis_request" in nlu.notes:
                out.append(compose.phrase("not_doctor", lang))
            if "dosage_request" in nlu.notes:
                out.append(compose.phrase("no_dosage", lang))
            if not ({"diagnosis_request", "dosage_request"} & set(nlu.notes)) or affirmed:
                before = len(out)
                add_kb([c.concept.value for c in affirmed], query=utt.text, k=1)
                if len(out) == before and "diagnosis_request" not in nlu.notes and "dosage_request" not in nlu.notes:
                    out.append(compose.phrase("no_kb", lang))
        elif intent == I.GREETING:
            out.append(compose.phrase("greet", lang))
        elif intent == I.GOODBYE:
            out.append(compose.phrase("goodbye", lang))
        elif intent in (I.USER_OK, I.CANCEL, I.CONFIRM) and not safety_said:
            out.append(compose.phrase("glad_ok" if intent == I.USER_OK else "listening", lang))
        elif intent == I.UNKNOWN and not safety_said and not out:
            out.append(compose.phrase("unknown", lang))
        return out

    def _previous_user_utterance(self, current: str) -> Optional[str]:
        for tr in reversed(self.traces.values()):
            uid = tr.input.get("id")
            if tr.input.get("type") == "utterance" and uid != current:
                return uid
        return None

    def _recall(self, nlu: NLUResult, utt: Utterance, lang: str) -> list[Statement]:
        target = nlu.recall_target
        if "allerg" in normalize(utt.text) or "एलर्जी" in utt.text:
            mems = [m for m in self.store.memories() if m["kind"] == "allergy"]
            if not mems:
                return [compose.phrase("no_record", lang, what=ph.name("allergy", lang))]
            return [compose.phrase("memory_recall", lang, ref=f"memory:{m['id']}", value=m["value"]) for m in mems[:3]]
        if target is None:
            mems = self.store.memories()
            if mems:
                return [compose.phrase("memory_recall", lang, ref=f"memory:{m['id']}", value=m["value"]) for m in mems[:3]]
            return [compose.phrase("no_record", lang, what=ph.name("note", lang))]
        if target in {m.value for m in MeasurementType}:
            evs = [e for e in self.store.events(kind=EventKind.MEASUREMENT.value)
                   if e.measurement and e.measurement.type.value == target]
            if not evs:
                return [compose.phrase("no_record", lang, what=ph.name(target, lang))]
            e = evs[0]
            return [compose.phrase("recall_value", lang, ref=f"event:{e.id}", what=ph.name(target, lang),
                                   value=fmt_measurement(e.measurement), when=fmt_time(e.ts))]
        if target == "pain":
            evs = [e for e in self.store.events(kind=EventKind.PAIN.value) if e.pain_score is not None]
            if not evs:
                return [compose.phrase("no_record", lang, what=ph.name("pain_score", lang))]
            e = evs[0]
            return [compose.phrase("recall_value", lang, ref=f"event:{e.id}", what=ph.name("pain_score", lang),
                                   value=f"{e.pain_score}/10", when=fmt_time(e.ts))]
        if target == "medication_taken":
            evs = self.store.events(kind=EventKind.MEDICATION_TAKEN.value)
            if not evs:
                return [compose.phrase("no_record", lang, what=ph.name("medication_taken", lang))]
            e = evs[0]
            return [compose.phrase("recall_value", lang, ref=f"event:{e.id}", what=ph.name("medication_taken", lang),
                                   value=e.medication or "", when=fmt_time(e.ts))]
        if target == "fall":
            evs = self.store.events(kind=EventKind.FALL_REPORTED.value) + self.store.events(kind=EventKind.FALL_DETECTED.value)
            if not evs:
                return [compose.phrase("no_record", lang, what=ph.name("c:fall", lang))]
            evs.sort(key=lambda e: e.ts, reverse=True)
            return [compose.phrase("recall_count", lang, ref=f"event:{evs[0].id}", n=len(evs),
                                   what=ph.name("c:fall", lang), when=fmt_time(evs[0].ts))]
        return [compose.phrase("no_record", lang, what=ph.name("note", lang))]

    def _llm(self, utt: Utterance, lang: str, said: list[Statement], events: list[HealthEvent],
             stored: dict[str, bool], trace: TurnTrace) -> list[Statement]:
        knowledge = {s.ref: s.text for s in said if s.ref and s.ref.startswith("kb:")}
        facts = {f"event:{e.id}": e.summary for e in events if stored.get(e.id)}
        res = llm_mod.enrich(self.llm, lang=lang, user_text=utt.text, already_said=[s.text for s in said],
                             knowledge=knowledge, user_facts=facts, observations={})
        trace.llm = {"call_id": res.call_id, "backend": res.backend, "accepted": res.accepted,
                     "violations": res.violations, "latency_ms": res.latency_ms, "error": res.error}
        self.store.audit("llm", "llm_call", call_id=res.call_id, backend=res.backend, accepted=res.accepted,
                         violations=[v["code"] for v in res.violations], error=res.error, latency_ms=res.latency_ms)
        return res.statements

    # ============================================================ UI / timers
    def ui_action(self, kind: str) -> Optional[TurnTrace]:
        mapping = {"help_now": "user_request_help_now", "im_ok": "user_ok", "cancel": "user_cancel",
                   "resolve": "resolve", "confirm": "user_confirm"}
        if kind not in mapping:
            raise ValueError(kind)
        with self.lock:
            inp = SafetyInput(kind=mapping[kind], source="ui", detail={"reason": "user_request"})  # type: ignore[arg-type]
            return self._system_turn(inp, {"type": "ui", "action": kind})

    def tick(self) -> Optional[TurnTrace]:
        with self.lock:
            inp = self.machine.due_timer(self.clock())
            if inp is None:
                return None
            return self._system_turn(inp, {"type": "timer", "timer": inp.detail.get("timer")})

    def _system_turn(self, inp: SafetyInput, input_desc: dict) -> Optional[TurnTrace]:
        now = self.clock()
        lang = self._lang_for_system_turn()
        trace = TurnTrace(input=input_desc)
        trace.safety_inputs.append(inp)
        hints: dict[str, Any] = {}
        tr = self.machine.feed(inp, now)
        statements: list[Statement] = []
        if tr:
            trace.transitions.append(tr)
            statements = self._execute(tr, lang, hints, trace)
        resp = BaymaxResponse(lang=lang, statements=statements, safety=self.machine.ctx, ui_hints=self._ui_hints(hints))
        trace.response = resp
        self._remember_trace(trace)
        self._emit({"type": "response", "trace_id": trace.id, "response": resp.model_dump(mode="json")})
        return trace

    # ============================================================ perception
    def ingest_body(self, samples: list[BodySample]) -> list[TurnTrace]:
        with self.lock:
            if not self.store.has_consent(ConsentScope.CAMERA) or self.privacy["camera_paused"]:
                return []
            events = self.perception.ingest_body(samples)
            return self._perception_events(events)

    def ingest_audio(self, levels: list[AudioLevel]) -> list[TurnTrace]:
        with self.lock:
            if not self.store.has_consent(ConsentScope.MICROPHONE) or self.privacy["mic_paused"]:
                return []
            return self._perception_events(self.perception.ingest_audio(levels))

    def _perception_events(self, events: list[PerceptionEvent]) -> list[TurnTrace]:
        out: list[TurnTrace] = []
        for ev in events:
            self.observations[ev.id] = ev
            while len(self.observations) > 500:
                self.observations.popitem(last=False)
            self._emit({"type": "observation", "observation": ev.model_dump(mode="json")})
            if ev.kind in ("fall", "lying_inactive"):
                self.store.audit("perception", "perception_event", obs_id=ev.id, kind=ev.kind, model_id=ev.model_id,
                                 confidence=ev.confidence, flags=ev.failure_flags)
                kind = "fall_detected" if ev.kind == "fall" else "lying_inactive"
                inp = SafetyInput(kind=kind, source="perception", confidence=ev.confidence, ref_id=ev.id,  # type: ignore[arg-type]
                                  severity=Severity.HIGH, detail={"evidence": ev.evidence, "flags": ev.failure_flags})
                t = self._system_turn(inp, {"type": "perception", "obs_id": ev.id, "kind": ev.kind})
                if t:
                    out.append(t)
            elif ev.kind == "posture_poor":
                if (self.store.has_consent(ConsentScope.POSTURE_COACHING)
                        and self.machine.ctx.state == SafetyState.MONITORING):
                    lang = self._lang_for_system_turn()
                    st = compose.phrase("posture_tip", lang, ref=f"obs:{ev.id}", model_id=ev.model_id, confidence=ev.confidence)
                    he = HealthEvent(ts=self.clock(), kind=EventKind.POSTURE_ALERT, severity=Severity.LOW, concepts=["posture"],
                                     provenance=Provenance.MODEL, source_ref=ev.id, model_id=ev.model_id,
                                     confidence=ev.confidence, summary="posture reminder", lang=lang)
                    self.store.add_event(he)
                    trace = TurnTrace(input={"type": "perception", "obs_id": ev.id, "kind": ev.kind}, events=[he])
                    trace.response = BaymaxResponse(lang=lang, statements=[st], safety=self.machine.ctx,
                                                    ui_hints=self._ui_hints())
                    self._remember_trace(trace)
                    self._emit({"type": "response", "trace_id": trace.id, "response": trace.response.model_dump(mode="json")})
                    out.append(trace)
        return out

    # ============================================================ provenance
    def resolve_ref(self, ref: str) -> Optional[dict]:
        """Resolve a statement reference to its source record (for the UI and tests)."""
        parts = ref.split("|")
        kind, _, ident = parts[0].partition(":")
        if kind == "kb":
            e = self.kb.get(ident)
            return {"type": "knowledge", "id": e.id, "source": e.source, "review_status": e.review_status,
                    "concepts": e.concepts} if e else None
        if kind == "phrase":
            return {"type": "phrase", "key": ident, "health_guidance": ident in ph.HEALTH_PHRASES} if ident in ph.P else None
        if kind == "event":
            e = self.store.get_event(ident)
            return {"type": "event", **e.model_dump(mode="json")} if e else None
        if kind == "obs":
            o = self.observations.get(ident)
            return {"type": "observation", **o.model_dump(mode="json")} if o else None
        if kind == "utt":
            for tr in self.traces.values():
                if tr.input.get("id") == ident:
                    return {"type": "utterance", "id": ident, "stored_text": None, "intent": tr.nlu.intent.value if tr.nlu else None}
            return {"type": "utterance", "id": ident}
        if kind == "memory":
            for m in self.store.memories(active_only=False):
                if m["id"] == ident:
                    return {"type": "memory", **m}
            return None
        if kind == "llm":
            return {"type": "llm", "call_id": ident, "cites": parts[1:]}
        return None
