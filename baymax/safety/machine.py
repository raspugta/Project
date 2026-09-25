"""Deterministic safety state machine.

`step()` is a pure function: (context, input, policy, now) -> transition.
It is the ONLY component that decides check-ins, countdowns and escalation.
Its inputs (SafetyInput) come from NLU, perception, UI buttons and timers.
There is no input kind for language-model output, so an LLM cannot drive it.

Every transition is identified by a rule id so the audit log can say exactly
why the system did what it did. The full rule table is in docs/SAFETY.md.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from ..contracts import SafetyAction, SafetyContext, SafetyInput, SafetyTransition
from ..ontology import SafetyState as S, Severity


class SafetyPolicy(BaseModel):
    checkin_timeout_s: int = 30          # unanswered "are you okay?" -> countdown
    countdown_s: int = 15                # countdown before alerting contacts
    fall_countdown_s: int = 30           # countdown after a fall where the user can't get up
    high_pain_threshold: int = 8         # offer to alert contacts at or above
    auto_escalate_on_no_response: bool = True
    fall_confidence_threshold: float = 0.6
    inactivity_confidence_threshold: float = 0.6
    escalated_reminder_s: int = 120      # repeat emergency guidance while escalated
    assist_idle_s: int = 180             # assisting flow times out silently


# Check-ins that end quietly on timeout instead of escalating: the user asked
# an informational question that mentioned a red flag ("what if someone has
# chest pain?"). They were asked whether it is happening now; silence means no.
NO_AUTO_ESCALATE_REASONS = {"red_flag_question"}


def _say(key: str, **params) -> SafetyAction:
    return SafetyAction(kind="say", phrase_key=key, params=params)


def _timer(name: str, seconds: int) -> SafetyAction:
    return SafetyAction(kind="start_timer", params={"name": name, "seconds": seconds})


def _log(kind: str, severity: Severity, **detail) -> SafetyAction:
    return SafetyAction(kind="log_event", params={"kind": kind, "severity": severity.value, **detail})


CANCEL_TIMER = SafetyAction(kind="cancel_timer")
STOP_ALARM = SafetyAction(kind="stop_alarm")


def _emergency_guidance(inp: SafetyInput) -> list[SafetyAction]:
    concepts = set(inp.detail.get("concepts", []))
    if "self_harm" in concepts:
        return [_say("crisis_support")]
    if inp.detail.get("subject") == "other":
        return [_say("emergency_other")]
    return [_say("call_emergency")]


def _ctx(state: S, now: float, **kw) -> SafetyContext:
    return SafetyContext(state=state, since=now, **kw)


def step(ctx: SafetyContext, inp: SafetyInput, policy: SafetyPolicy, now: float) -> Optional[SafetyTransition]:
    """Return the transition for `inp`, or None if the input is irrelevant in this state."""
    st = ctx.state
    k = inp.kind
    A: list[SafetyAction] = []

    def out(new: SafetyContext, rule: str) -> SafetyTransition:
        if new.timer_name and new.timer_deadline is None:
            raise AssertionError("timer without deadline")
        return SafetyTransition(before=st, after=new.state, input=inp, actions=A, rule=rule, context=new, ts=now)

    def with_timer(state: S, name: str, seconds: int, **kw) -> SafetyContext:
        A.append(_timer(name, seconds))
        return _ctx(state, now, timer_name=name, timer_deadline=now + seconds, **kw)

    # ---- stale timers are ignored everywhere ---------------------------
    if k == "timer_expired" and inp.detail.get("timer") != ctx.timer_name:
        return None

    # ---- global rules --------------------------------------------------
    if k == "user_request_help_now" and st != S.ESCALATED:
        A += [CANCEL_TIMER, SafetyAction(kind="notify_contacts", params={"level": "emergency", "reason": inp.detail.get("reason", "user_request")}),
              SafetyAction(kind="sound_alarm"), _log("escalation", Severity.CRITICAL, reason="user_request")]
        A += _emergency_guidance(inp)
        new = with_timer(S.ESCALATED, "reminder", policy.escalated_reminder_s, reason="user_request",
                         severity=Severity.CRITICAL, critical=True, trigger_ref=inp.ref_id)
        return out(new, "R-G-HELP-NOW")

    if k == "critical_utterance":
        if st == S.ESCALATED:
            A += _emergency_guidance(inp)
            return out(ctx.model_copy(update={"since": ctx.since}), "R-E-REPEAT")
        if st == S.ESCALATION_COUNTDOWN:
            A += _emergency_guidance(inp)
            return out(ctx.model_copy(update={"critical": True, "severity": Severity.CRITICAL}), "R-X-REPEAT")
        A += [CANCEL_TIMER, _log("emergency", Severity.CRITICAL, **inp.detail)]
        A += _emergency_guidance(inp)
        A.append(_say("countdown", n=policy.countdown_s))
        A.append(SafetyAction(kind="sound_alarm", params={"level": "soft"}))
        new = with_timer(S.ESCALATION_COUNTDOWN, "countdown", policy.countdown_s, reason=inp.detail.get("reason", "critical"),
                         severity=Severity.CRITICAL, critical=True, awaiting="confirm_escalation", trigger_ref=inp.ref_id)
        return out(new, "R-G-CRITICAL")

    # ---- MONITORING ----------------------------------------------------
    if st == S.MONITORING:
        if k == "possible_emergency":
            A += [_say("need_help_q")]
            if inp.detail.get("red_flag_uncertain"):
                A += [_say("check_now")]
            reason = "red_flag_question" if inp.detail.get("red_flag_uncertain") else "possible_emergency"
            new = with_timer(S.CHECK_IN, "checkin", policy.checkin_timeout_s, reason=reason,
                             severity=Severity.HIGH, awaiting="ok_check", trigger_ref=inp.ref_id)
            return out(new, "R-M-POSSIBLE")
        if k == "pain_exclamation":
            A += [_say("pain_heard"), _say("ask_ok"), _say("ask_pain_scale")]
            new = with_timer(S.ASSISTING, "assist", policy.assist_idle_s, reason="pain_exclamation",
                             severity=Severity.MODERATE, awaiting="pain_score", trigger_ref=inp.ref_id)
            return out(new, "R-M-OUCH")
        if k == "pain_report":
            return _pain_report(ctx, inp, policy, now, A, out, with_timer, from_state="M")
        if k == "fall_reported":
            A += [_log("fall_reported", Severity.HIGH), _say("fall_reported_checkin")]
            if inp.detail.get("cannot_get_up") or inp.detail.get("head_injury"):
                A += [_say("countdown", n=policy.fall_countdown_s), SafetyAction(kind="sound_alarm", params={"level": "soft"})]
                new = with_timer(S.ESCALATION_COUNTDOWN, "countdown", policy.fall_countdown_s, reason="fall_cannot_get_up",
                                 severity=Severity.HIGH, awaiting="confirm_escalation", trigger_ref=inp.ref_id)
                return out(new, "R-M-FALL-SEVERE")
            new = with_timer(S.CHECK_IN, "checkin", policy.checkin_timeout_s, reason="fall_reported",
                             severity=Severity.HIGH, awaiting="ok_check", trigger_ref=inp.ref_id)
            return out(new, "R-M-FALL-REPORTED")
        if k == "fall_detected" and inp.confidence >= policy.fall_confidence_threshold:
            A += [_log("fall_detected", Severity.HIGH, confidence=inp.confidence),
                  _say("fall_detected_checkin", n=policy.checkin_timeout_s),
                  SafetyAction(kind="sound_alarm", params={"level": "chime"})]
            new = with_timer(S.CHECK_IN, "checkin", policy.checkin_timeout_s, reason="fall_detected",
                             severity=Severity.HIGH, awaiting="ok_check", trigger_ref=inp.ref_id)
            return out(new, "R-M-FALL-DETECTED")
        if k == "lying_inactive" and inp.confidence >= policy.inactivity_confidence_threshold:
            A += [_log("inactivity", Severity.MODERATE, confidence=inp.confidence), _say("inactivity_checkin")]
            new = with_timer(S.CHECK_IN, "checkin", policy.checkin_timeout_s, reason="lying_inactive",
                             severity=Severity.MODERATE, awaiting="ok_check", trigger_ref=inp.ref_id)
            return out(new, "R-M-INACTIVE")
        return None

    # ---- CHECK_IN ------------------------------------------------------
    if st == S.CHECK_IN:
        if k in ("user_ok", "user_cancel"):
            A += [CANCEL_TIMER, _say("glad_ok"), _log("check_in", Severity.INFO, outcome="user_ok")]
            return out(_ctx(S.MONITORING, now), "R-C-OK")
        if k == "possible_emergency":
            A += [_say("countdown", n=policy.countdown_s), SafetyAction(kind="sound_alarm", params={"level": "soft"})]
            new = with_timer(S.ESCALATION_COUNTDOWN, "countdown", policy.countdown_s, reason=f"{ctx.reason}+not_ok",
                             severity=Severity.HIGH, awaiting="confirm_escalation", trigger_ref=ctx.trigger_ref)
            return out(new, "R-C-NOT-OK")
        if k in ("pain_report", "pain_exclamation") and inp.detail.get("pain_score") is None:
            # keep the check-in (and its no-response timer) running while we ask about the pain
            A += [CANCEL_TIMER, _say("ask_pain_scale")]
            new = with_timer(S.CHECK_IN, "checkin", policy.checkin_timeout_s, reason=ctx.reason, severity=ctx.severity,
                             awaiting="pain_score", trigger_ref=ctx.trigger_ref)
            return out(new, "R-C-PAIN-ASK")
        if k == "pain_report":
            return _pain_report(ctx, inp, policy, now, A, out, with_timer, from_state="C")
        if k == "fall_reported":
            A += [_say("fall_reported_checkin")]
            if inp.detail.get("cannot_get_up") or inp.detail.get("head_injury"):
                A += [_say("countdown", n=policy.fall_countdown_s)]
                new = with_timer(S.ESCALATION_COUNTDOWN, "countdown", policy.fall_countdown_s, reason="fall_cannot_get_up",
                                 severity=Severity.HIGH, awaiting="confirm_escalation", trigger_ref=inp.ref_id)
                return out(new, "R-C-FALL-SEVERE")
            new = with_timer(S.CHECK_IN, "checkin", policy.checkin_timeout_s, reason="fall_reported",
                             severity=Severity.HIGH, awaiting="ok_check", trigger_ref=inp.ref_id)
            return out(new, "R-C-FALL-REPORTED")
        if k == "user_confirm":
            A += [CANCEL_TIMER, SafetyAction(kind="notify_contacts", params={"level": "emergency", "reason": ctx.reason}),
                  _log("escalation", Severity.HIGH, reason=ctx.reason)]
            new = with_timer(S.ESCALATED, "reminder", policy.escalated_reminder_s, reason=ctx.reason,
                             severity=Severity.HIGH, trigger_ref=ctx.trigger_ref)
            return out(new, "R-C-CONFIRM")
        if k == "timer_expired":
            if ctx.reason in NO_AUTO_ESCALATE_REASONS:
                A += [_log("check_in", Severity.INFO, outcome="informational_question_no_reply")]
                return out(_ctx(S.MONITORING, now), "R-C-TIMEOUT-INFO")
            if policy.auto_escalate_on_no_response:
                A += [_say("no_response"), _say("countdown", n=policy.countdown_s), SafetyAction(kind="sound_alarm", params={"level": "loud"})]
                new = with_timer(S.ESCALATION_COUNTDOWN, "countdown", policy.countdown_s, reason=f"{ctx.reason}+no_response",
                                 severity=Severity.HIGH, awaiting="confirm_escalation", trigger_ref=ctx.trigger_ref)
                return out(new, "R-C-TIMEOUT")
            A += [_log("check_in", Severity.MODERATE, outcome="no_response_no_escalation")]
            return out(_ctx(S.MONITORING, now), "R-C-TIMEOUT-NOESC")
        return None

    # ---- ASSISTING -----------------------------------------------------
    if st == S.ASSISTING:
        if k == "pain_report":
            return _pain_report(ctx, inp, policy, now, A, out, with_timer, from_state="A")
        if k == "user_confirm" and ctx.awaiting == "confirm_escalation":
            A += [CANCEL_TIMER, SafetyAction(kind="notify_contacts", params={"level": "urgent", "reason": ctx.reason}),
                  _log("escalation", Severity.HIGH, reason=ctx.reason)]
            new = with_timer(S.ESCALATED, "reminder", policy.escalated_reminder_s, reason=ctx.reason,
                             severity=Severity.HIGH, trigger_ref=ctx.trigger_ref)
            return out(new, "R-A-CONFIRM")
        if k in ("user_cancel", "user_ok"):
            A += [CANCEL_TIMER, _say("glad_ok" if k == "user_ok" else "seek_care_if_worse")]
            return out(_ctx(S.MONITORING, now), "R-A-DECLINE")
        if k == "possible_emergency":
            A += [_say("countdown", n=policy.countdown_s), SafetyAction(kind="sound_alarm", params={"level": "soft"})]
            new = with_timer(S.ESCALATION_COUNTDOWN, "countdown", policy.countdown_s, reason=f"{ctx.reason}+possible_emergency",
                             severity=Severity.HIGH, awaiting="confirm_escalation", trigger_ref=inp.ref_id)
            return out(new, "R-A-POSSIBLE")
        if k in ("fall_detected", "fall_reported", "lying_inactive"):
            # a fall outranks an ongoing pain conversation: behave as from MONITORING
            t = step(_ctx(S.MONITORING, now), inp, policy, now)
            return t.model_copy(update={"before": S.ASSISTING, "rule": t.rule.replace("R-M-", "R-A-")}) if t else None
        if k == "timer_expired":
            return out(_ctx(S.MONITORING, now), "R-A-IDLE")
        return None

    # ---- ESCALATION_COUNTDOWN -------------------------------------------
    if st == S.ESCALATION_COUNTDOWN:
        if k in ("user_cancel", "user_ok"):
            A += [CANCEL_TIMER, STOP_ALARM, _say("cancelled"), _log("escalation", Severity.MODERATE, outcome="cancelled_by_user", reason=ctx.reason)]
            if ctx.critical:
                A.append(_say("cancelled_but_advise"))
            return out(_ctx(S.MONITORING, now), "R-X-CANCEL")
        if k in ("user_confirm", "timer_expired"):
            A += [CANCEL_TIMER, SafetyAction(kind="notify_contacts", params={"level": "emergency", "reason": ctx.reason}),
                  SafetyAction(kind="sound_alarm", params={"level": "loud"}),
                  _log("escalation", Severity.CRITICAL if ctx.critical else Severity.HIGH, reason=ctx.reason,
                       via="confirm" if k == "user_confirm" else "countdown")]
            A += [_say("stay_calm"), _say("call_emergency")]
            new = with_timer(S.ESCALATED, "reminder", policy.escalated_reminder_s, reason=ctx.reason,
                             severity=ctx.severity, critical=ctx.critical, trigger_ref=ctx.trigger_ref)
            return out(new, "R-X-CONFIRM" if k == "user_confirm" else "R-X-TIMEOUT")
        return None

    # ---- ESCALATED -----------------------------------------------------
    if st == S.ESCALATED:
        if k == "resolve":
            A += [CANCEL_TIMER, STOP_ALARM, _say("resolved"), _log("escalation", Severity.INFO, outcome="resolved")]
            return out(_ctx(S.MONITORING, now), "R-E-RESOLVE")
        if k in ("user_ok", "user_cancel"):
            A += [CANCEL_TIMER, STOP_ALARM, SafetyAction(kind="notify_contacts", params={"level": "all_clear", "reason": ctx.reason}),
                  _say("glad_ok"), _log("escalation", Severity.INFO, outcome="user_all_clear")]
            if ctx.critical:
                A.append(_say("cancelled_but_advise"))
            return out(_ctx(S.MONITORING, now), "R-E-ALL-CLEAR")
        if k == "timer_expired":
            A += [_say("call_emergency")]
            new = with_timer(S.ESCALATED, "reminder", policy.escalated_reminder_s, reason=ctx.reason,
                             severity=ctx.severity, critical=ctx.critical, trigger_ref=ctx.trigger_ref,
                             escalation_id=ctx.escalation_id)
            return out(new, "R-E-REMIND")
        return None
    return None


def _pain_report(ctx, inp, policy, now, A, out, with_timer, from_state: str):
    score = inp.detail.get("pain_score")
    A.append(CANCEL_TIMER)
    if score is None:
        A += [_say("ask_pain_scale")]
        new = with_timer(S.ASSISTING, "assist", policy.assist_idle_s, reason="pain",
                         severity=Severity.MODERATE, awaiting="pain_score", trigger_ref=inp.ref_id)
        return out(new, f"R-{from_state}-PAIN-ASK")
    A.append(_say("pain_logged", score=score))
    if score >= policy.high_pain_threshold:
        A += [_say("pain_high_offer")]
        new = with_timer(S.ASSISTING, "assist", policy.assist_idle_s, reason="high_pain",
                         severity=Severity.HIGH, awaiting="confirm_escalation", trigger_ref=inp.ref_id)
        return out(new, f"R-{from_state}-PAIN-HIGH")
    A += [_say("seek_care_if_worse")]
    return out(_ctx(S.MONITORING, now), f"R-{from_state}-PAIN-LOGGED")


class SafetyMachine:
    """Stateful wrapper around the pure `step` function."""

    def __init__(self, policy: SafetyPolicy | None = None) -> None:
        self.policy = policy or SafetyPolicy()
        self.ctx = SafetyContext()

    def feed(self, inp: SafetyInput, now: float) -> Optional[SafetyTransition]:
        tr = step(self.ctx, inp, self.policy, now)
        if tr is not None:
            self._apply(tr, inp, now)
        return tr

    def _apply(self, tr: SafetyTransition, inp: SafetyInput, now: float) -> None:
        self.ctx = tr.context

    def due_timer(self, now: float) -> Optional[SafetyInput]:
        if self.ctx.timer_name and self.ctx.timer_deadline is not None and now >= self.ctx.timer_deadline:
            return SafetyInput(kind="timer_expired", source="timer", ts=now, detail={"timer": self.ctx.timer_name})
        return None
