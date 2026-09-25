"""Intent classifier + health-event extractor (model_id: nlu_lexicon_v1).

Deterministic and fully inspectable. Safety bias: uncertainty about an
emergency never resolves to "ignore" - it resolves to a check-in.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from ..contracts import ConceptMention, LanguageResult, NLUResult, Utterance
from ..ontology import RED_FLAG_CONCEPTS, BodyRegion, Concept, Intent, MeasurementType
from ..language import lexicon as lx
from ..language.text import Match, PhraseIndex, Token, normalize, squash, split_phrases, tokenize
from .extract import extract_measurements, extract_pain_score

MODEL_ID = "nlu_lexicon_v1"

# Tokens that do not count toward negation distance ("I don't HAVE ANY chest pain").
_NEG_SKIP = {"have", "any", "a", "an", "the", "really", "feel", "got", "had", "feeling", "been",
             "koi", "bhi", "hai", "ho", "raha", "rahi", "कोई", "भी", "है",
             "tengo", "ningun", "ninguna", "de", "d'", "du", "la", "le", "habe", "hab", "eine", "einen", "ich"}
_FIRST_PERSON = {"i", "i'm", "im", "main", "mai", "मैं", "yo", "je", "j'ai", "ich", "me", "mujhe", "मुझे"}
_PAIN_LIKE = {Concept.PAIN, Concept.HEADACHE, Concept.SORE_THROAT}
_CANNOT_GET_UP = {"can't get up", "cant get up", "cannot get up", "uth nahi pa raha", "uth nahi pa rahi",
                  "utha nahi ja raha", "उठ नहीं पा रहा", "उठ नहीं पा रही", "no puedo levantarme",
                  "je ne peux pas me relever", "je n'arrive pas a me relever", "komme nicht hoch",
                  "kann nicht aufstehen"}
_MED_STOP = set("at this today now and with for in after before morning evening tonight just already "
                "hoy ahora y con por esta ce matin maintenant et avec pour heute jetzt und mit fur "
                "abhi aaj subah shaam aur".split())
_MED_DROP = set("my the a an some one two mi la el las los une un mon ma mes le les meine mein meinen "
                "die das der maine main mai mujhe meri mera apni apna ki ka ne".split())


# Structural detectors that complement the phrase lexicons.
_DOSAGE_RX = re.compile(
    r"(how (many|much)|what dose|max(imum)? dose|double (my|the) dose|kitni|kitna|cuantas|cuanto|combien|wie viele|wieviel|welche dosis|quelle dose|que dosis)"
    r".{0,40}(tablet|pill|capsule|mg|dose|paracetamol|acetaminophen|ibuprofen|aspirin|insulin|tylenol|advil|goli|dawai|dawa|pastilla|comprime|tablette|medikament|medicament|medicamento)"
    r"|(can|should|may) i (take|double|increase).{0,30}(dose|pills?|tablets?|mg|insulin)")
_EXFIL_RX = re.compile(
    r"(send|share|upload|export|email|e-mail|forward|post|envia|manda|envoie|partage|schick|sende|bhejo|bhej do).{0,40}"
    r"(data|records?|history|information|info|datos|donnees|daten|jaankari)"
    r"|[\w.+-]+@[\w-]+\.[\w.]+|https?://")
_MARKUP_INJECTION_RX = re.compile(r"</?\s*(system|assistant|user|instructions?)\s*>|\[\s*(system|inst)\s*\]|<<\s*sys", re.I)


def _transposition_or_equal(a: str, b: str) -> bool:
    if a == b:
        return True
    if len(a) != len(b) or len(a) < 4:
        return False
    diff = [i for i in range(len(a)) if a[i] != b[i]]
    return len(diff) == 2 and diff[1] == diff[0] + 1 and a[diff[0]] == b[diff[1]] and a[diff[1]] == b[diff[0]]


def _index(table: dict) -> PhraseIndex:
    idx = PhraseIndex()
    for key, by_lang in table.items():
        k = key.value if hasattr(key, "value") else key
        idx.add_lexicon(k, by_lang)
    return idx


def _single(name: str, by_lang: dict[str, str]) -> PhraseIndex:
    idx = PhraseIndex()
    idx.add_lexicon(name, by_lang)
    return idx


@dataclass
class _Ctx:
    tokens: list[Token]
    langs: set[str]
    content: list[Token] = field(default_factory=list)
    idiom_spans: list[tuple[int, int]] = field(default_factory=list)


class NLU:
    model_id = MODEL_ID

    def __init__(self) -> None:
        self.concepts = _index(lx.CONCEPTS)
        self.body = _index(lx.BODY)
        self.cues = {name: _single(name, table) for name, table in {
            "strong": lx.EMERGENCY_STRONG, "help_open": lx.HELP_OPEN, "possible": lx.POSSIBLE_EMERGENCY,
            "idiom": lx.IDIOMS, "fake": lx.FAKE_MARKERS, "ok": lx.USER_OK, "confirm": lx.CONFIRM,
            "cancel": lx.CANCEL, "greeting": lx.GREETING, "goodbye": lx.GOODBYE, "question": lx.QUESTION,
            "diagnosis": lx.DIAGNOSIS_REQUEST, "dosage": lx.DOSAGE_REQUEST, "recall": lx.RECALL,
            "medication": lx.MEDICATION_TAKEN, "negator": lx.NEGATORS, "hypo": lx.HYPOTHETICAL,
            "past": lx.PAST, "other": lx.OTHER_SUBJECT, "injection": lx.INJECTION,
            "private": lx.PRIVATE_REQUEST,
        }.items()}
        self.privacy = _index(lx.PRIVACY)
        self._exact = {name: {normalize(p): lang for lang, s in table.items() for p in split_phrases(s)}
                       for name, table in {"emergency": lx.EMERGENCY_EXACT, "ok": lx.USER_OK_EXACT,
                                           "confirm": lx.CONFIRM, "cancel": lx.CANCEL}.items()}
        self._fillers = {lang: set(split_phrases(s)) for lang, s in lx.FILLERS.items()}
        self._breakers = {w for s in lx.CLAUSE_BREAKERS.values() for w in split_phrases(s)}
        self._excl = {lang: {squash(w) for w in split_phrases(s)} for lang, s in lx.PAIN_EXCLAMATIONS.items()}
        self._excl_all = set().union(*self._excl.values())
        self._help_cont = {w for s in lx.HELP_CONTINUATION.values() for w in split_phrases(s)}
        self._wake = {normalize(w) for w in lx.WAKE_WORDS}
        self._exact_single = {squash(p) for p in self._exact["emergency"] if " " not in p}

    # ------------------------------------------------------------------
    @staticmethod
    def scope(lang: str) -> set[str]:
        s = {lang, "en"}
        if lang in ("hi", "hi-Latn"):
            s |= {"hi", "hi-Latn"}
        return s

    def _find(self, name: str, ctx: _Ctx, all_langs: bool = False) -> list[Match]:
        return self.cues[name].find(ctx.tokens, None if all_langs else ctx.langs)

    def _in_idiom(self, m: Match, ctx: _Ctx) -> bool:
        return any(not (m.end <= a or m.start >= b) for a, b in ctx.idiom_spans)

    def _negated(self, m: Match, ctx: _Ctx, negs: list[Match]) -> bool:
        toks = ctx.tokens
        neg_starts = {n.start: n for n in negs if not (n.start >= m.start and n.end <= m.end)}
        # look back up to 2 counted tokens in the same clause
        counted, i = 0, m.start - 1
        while i >= 0 and counted < 2 and toks[i].clause == toks[m.start].clause:
            if i in neg_starts:
                is_interjection = (i == 0 or toks[i - 1].clause != toks[i].clause) and \
                    i + 1 < len(toks) and toks[i + 1].text in _FIRST_PERSON and toks[i].text in ("no", "nahi", "non", "nein")
                if not is_interjection:
                    return True
            if toks[i].text not in _NEG_SKIP:
                counted += 1
            i -= 1
        # postposed negation (Hindi): "dard nahi hai"
        if ctx.langs & lx.POSTPOSED_NEGATION:
            j, counted = m.end, 0
            while j < len(toks) and counted < 2 and toks[j].clause == toks[m.end - 1].clause:
                if j in neg_starts and neg_starts[j].lang in lx.POSTPOSED_NEGATION:
                    return True
                if toks[j].text not in _NEG_SKIP:
                    counted += 1
                j += 1
        return False

    @staticmethod
    def _before_in_clause(m: Match, cues: list[Match], include_inside: bool = False) -> bool:
        for c in cues:
            if c.clause != m.clause:
                continue
            if c.start < m.start or (include_inside and c.start >= m.start and c.end <= m.end):
                return True
        return False

    # ------------------------------------------------------------------
    def analyze(self, utt: Utterance, lang_result: LanguageResult, lang: str,
                awaiting: Optional[str] = None) -> NLUResult:
        raw = utt.text
        norm = normalize(raw)
        tokens = tokenize(norm, self._breakers)
        ctx = _Ctx(tokens=tokens, langs=self.scope(lang))
        fillers = set().union(*[self._fillers.get(l, set()) for l in ctx.langs]) | self._wake
        ctx.content = [t for t in tokens if t.text not in fillers]
        content_text = " ".join(t.text for t in ctx.content)
        notes: list[str] = []

        ctx.idiom_spans = [(m.start, m.end) for m in self._find("idiom", ctx)]
        negs = self._find("negator", ctx)
        hypos = self._find("hypo", ctx)
        _since = {"since", "desde", "depuis", "seit", "se", "से"}
        pasts = [m for m in self._find("past", ctx) if not (m.start > 0 and tokens[m.start - 1].text in _since)]
        others = self._find("other", ctx)
        is_question = "?" in raw or "¿" in raw

        # ---- concepts --------------------------------------------------
        cmatches = self.concepts.find(tokens, ctx.langs)
        cmatches += [m for m in self.concepts.find(tokens, None)
                     if Concept(m.key) in lx.CROSS_LANGUAGE_CONCEPTS and m.lang not in ctx.langs]
        cmatches = [m for m in cmatches if not self._in_idiom(m, ctx)]
        # drop matches strictly contained in another concept's span
        cmatches = [m for m in cmatches if not any(
            o is not m and o.key != m.key and o.start <= m.start and o.end >= m.end and (o.end - o.start) > (m.end - m.start)
            for o in cmatches)]
        body = [m for m in self.body.find(tokens, ctx.langs) if not self._in_idiom(m, ctx)]
        # "my back is killing me"
        for i in range(len(tokens) - 1):
            if tokens[i].text == "killing" and tokens[i + 1].text == "me" and any(b.clause == tokens[i].clause for b in body):
                cmatches.append(Match(Concept.PAIN.value, "en", i, i + 2, tokens[i].clause, "killing me"))

        mentions: list[ConceptMention] = []
        seen: set[tuple[str, bool]] = set()
        for m in cmatches:
            c = Concept(m.key)
            region = next((BodyRegion(b.key) for b in sorted(body, key=lambda b: abs(b.start - m.start))
                           if b.clause == m.clause), None)
            subject = "other" if self._before_in_clause(m, others, include_inside=True) else "self"
            mention = ConceptMention(
                concept=c,
                negated=self._negated(m, ctx, negs),
                hypothetical=self._before_in_clause(m, hypos),
                past=any(p.clause == m.clause for p in pasts),
                subject=subject,  # type: ignore[arg-type]
                span=" ".join(t.text for t in tokens[m.start:m.end]),
                body_region=region,
            )
            if m.phrase in _CANNOT_GET_UP and not mention.negated:
                notes.append("cannot_get_up")
            key = (c.value, mention.negated)
            if key in seen:
                continue
            seen.add(key)
            mentions.append(mention)
            if c is Concept.PAIN and region in lx.PAIN_REGION_DERIVED and not mention.negated:
                d = lx.PAIN_REGION_DERIVED[region]
                if (d.value, False) not in seen:
                    seen.add((d.value, False))
                    mentions.append(mention.model_copy(update={"concept": d}))
        # a question clause makes red-flag mentions in it hypothetical
        if is_question:
            qclauses = {m.clause for m in self._find("question", ctx)} | {m.clause for m in hypos}
            if qclauses and len({t.clause for t in tokens}) == 1:
                mentions = [mm.model_copy(update={"hypothetical": True}) for mm in mentions]

        affirmed = [m for m in mentions if not m.negated and not m.hypothetical]
        red_affirmed = [m for m in affirmed if m.concept in RED_FLAG_CONCEPTS and not m.past]
        red_uncertain = [m for m in mentions if m.concept in RED_FLAG_CONCEPTS and not m.negated and (m.hypothetical or m.past)]
        negated_only = bool(mentions) and all(m.negated for m in mentions)
        body_regions = list(dict.fromkeys(BodyRegion(b.key) for b in body))

        # ---- cues ------------------------------------------------------
        strong = [m for m in self._find("strong", ctx, all_langs=True)
                  if not self._in_idiom(m, ctx) and not self._negated(m, ctx, negs)]
        n_content = len(ctx.content)
        exact_emergency = content_text in self._exact["emergency"]
        # "help help help", "hlep", "Hilfee": short utterances made only of (misspelt / repeated) help words
        if not exact_emergency and 0 < n_content <= 4:
            words = [squash(t.text) for t in ctx.content if t.text not in ("me", "mujhe", "moi", "mir", "me")]
            if words and all(any(_transposition_or_equal(w, e) for e in self._exact_single) for w in words):
                exact_emergency = True
                notes.append("fuzzy_emergency_match")
        help_open = False
        for m in self._find("help_open", ctx, all_langs=True):
            rest = [t for t in tokens[m.end:] if t.text not in fillers]
            if not rest or rest[0].text not in self._help_cont:
                help_open = help_open or not self._negated(m, ctx, negs)
        possible = [m for m in self._find("possible", ctx) if not self._in_idiom(m, ctx)]
        fake = bool(self._find("fake", ctx))
        injection = bool(self._find("injection", ctx, all_langs=True)) or bool(_MARKUP_INJECTION_RX.search(raw))
        n_content = len(ctx.content)

        ok_phr = [m for m in self._find("ok", ctx) if not self._negated(m, ctx, negs)]
        ok = bool(ok_phr) or content_text in self._exact["ok"]
        cancel_phr = self._find("cancel", ctx)
        cancel_multi = [m for m in cancel_phr if m.end - m.start > 1]
        cancel = bool(cancel_multi) or (n_content <= 3 and bool(cancel_phr)) or content_text in self._exact["cancel"]
        confirm_phr = [m for m in self._find("confirm", ctx) if not self._negated(m, ctx, negs)]
        confirm = content_text in self._exact["confirm"] or (n_content <= 4 and any(m.start <= (ctx.content[0].pos if ctx.content else 0) for m in confirm_phr))
        greeting = n_content <= 5 and any(m.start <= 1 for m in self._find("greeting", ctx))
        goodbye = n_content <= 6 and bool(self._find("goodbye", ctx))
        question_cues = self._find("question", ctx)
        first_content = {t.clause: t.pos for t in reversed(ctx.content)}   # first non-filler token per clause
        question = is_question or any(m.start == 0 or tokens[m.start - 1].clause != m.clause or first_content.get(m.clause) == m.start
                                      for m in question_cues)
        if ctx.langs & {"hi", "hi-Latn"} and question_cues:  # SOV: question words sit mid-sentence
            question = True
        diagnosis_req = bool(self._find("diagnosis", ctx))
        dosage_req = bool(self._find("dosage", ctx)) or bool(_DOSAGE_RX.search(norm))
        recall = bool(self._find("recall", ctx))
        med = self._find("medication", ctx)

        # exclamation: first content token is a pain sound
        excl_set = set().union(*[self._excl.get(l, set()) for l in ctx.langs])
        exclamation = False
        # tokens stripped of fillers already; a lone "ah"/"eh" is hesitation, not a pain sound
        ctx.content = [t for t in ctx.content if t.text not in ("ah", "eh")] or ctx.content
        if ctx.content:
            first = ctx.content[0]
            if squash(first.text) in excl_set:
                nxt = ctx.content[1] if len(ctx.content) > 1 else None
                exclamation = (n_content <= 3 or nxt is None or nxt.clause != first.clause
                               or all(squash(t.text) in excl_set for t in ctx.content))
            elif all(squash(t.text) in excl_set for t in ctx.content):
                exclamation = True
            elif n_content <= 3 and all(squash(t.text) in self._excl_all for t in ctx.content):
                exclamation = True  # pure pain sounds carry no language evidence: accept any language
            # two-token exclamations like "ui maa"
            if not exclamation and len(ctx.content) >= 2:
                two = squash(ctx.content[0].text) + " " + squash(ctx.content[1].text)
                exclamation = two in excl_set and n_content <= 4

        # ---- privacy ---------------------------------------------------
        privacy_action = None
        pm = [m for m in self.privacy.find(tokens, None) if not self._negated(m, ctx, negs)]
        if pm:
            privacy_action = pm[0].key

        # ---- extraction -----------------------------------------------
        measurements, rejects = extract_measurements(norm)
        notes += [f"implausible:{r}" for r in rejects]
        pain_score = extract_pain_score(norm, lang, awaiting == "pain_score", n_content)
        if pain_score is not None:
            measurements = [m for m in measurements if not (m.value == pain_score and m.type in (MeasurementType.HEART_RATE,))]
        medication = self._medication(tokens, med, lang) if med else None
        memory_fact = self._memory(norm, ctx.langs)
        recall_target = self._recall_target(norm, mentions) if recall else None

        # ---- intent decision (ordered, first match wins) ---------------
        intent, conf = Intent.UNKNOWN, 0.3
        secondary: list[Intent] = []
        critical_cue = bool(strong) or exact_emergency or help_open
        if injection:
            notes.append("injection_suspected")
        if self._find("private", ctx, all_langs=True) or _EXFIL_RX.search(norm):
            notes.append("private_data_request")
        if diagnosis_req:
            notes.append("diagnosis_request")
        if dosage_req:
            notes.append("dosage_request")
        if red_uncertain and not red_affirmed:
            notes.append("red_flag_uncertain")

        if red_affirmed or (critical_cue and not fake):
            intent, conf = Intent.EMERGENCY_HELP, 0.95
            if any(m.subject == "other" for m in red_affirmed):
                notes.append("subject_other")
        elif critical_cue and fake:
            intent, conf = Intent.EMERGENCY_HELP, 0.5
            notes += ["fake_marker", "ambiguous_emergency"]
        elif privacy_action:
            intent, conf = Intent.PRIVACY_COMMAND, 0.9
        elif awaiting == "pain_score" and pain_score is not None:
            intent, conf = Intent.PAIN_REPORT, 0.9
        elif awaiting in ("ok_check", "confirm_escalation") and (ok or cancel or confirm):
            if awaiting == "ok_check":
                if ok or (confirm and not cancel):
                    intent, conf = Intent.USER_OK, 0.9
                else:  # "no" to "are you okay?"
                    intent, conf = Intent.EMERGENCY_HELP, 0.6
                    notes += ["negative_to_ok_check", "ambiguous_emergency"]
            else:
                if confirm and not cancel:
                    intent, conf = Intent.CONFIRM, 0.9
                elif ok:
                    intent, conf = Intent.USER_OK, 0.9
                else:
                    intent, conf = Intent.CANCEL, 0.9
        elif recall and question:
            intent, conf = Intent.RECALL_QUERY, 0.85
        elif any(m.concept is Concept.FALL and not m.past for m in affirmed):
            intent, conf = Intent.FALL_REPORT, 0.9
        elif negated_only and n_content <= 6 and not question:
            intent, conf = Intent.USER_OK, 0.7   # "no pain", "dard nahi hai"
        elif cancel_multi or (cancel and n_content <= 3 and not ok):
            intent, conf = Intent.CANCEL, 0.8
        elif ok and not affirmed:
            intent, conf = Intent.USER_OK, 0.85
        elif confirm and n_content <= 3:
            intent, conf = Intent.CONFIRM, 0.7
        elif exclamation and pain_score is None:
            intent, conf = Intent.PAIN_EXCLAMATION, 0.85
        elif possible and not affirmed and not question:
            intent, conf = Intent.EMERGENCY_HELP, 0.6
            notes.append("ambiguous_emergency")
        elif red_uncertain:
            intent, conf = Intent.HEALTH_QUESTION, 0.8
        elif diagnosis_req or dosage_req:
            intent, conf = Intent.HEALTH_QUESTION, 0.85
        elif recall:
            intent, conf = Intent.RECALL_QUERY, 0.85
        elif measurements or any(r for r in rejects):
            intent, conf = Intent.VITAL_REPORT, 0.9
        elif medication:
            intent, conf = Intent.MEDICATION_LOG, 0.85
        elif memory_fact:
            intent, conf = Intent.MEMORY_STATEMENT, 0.85
        elif question and (mentions or body_regions):
            intent, conf = Intent.HEALTH_QUESTION, 0.8
        elif any(m.concept in _PAIN_LIKE for m in affirmed) or pain_score is not None:
            intent, conf = Intent.PAIN_REPORT, 0.85
        elif affirmed:
            intent, conf = Intent.SYMPTOM_REPORT, 0.8
        elif greeting:
            intent, conf = Intent.GREETING, 0.8
        elif goodbye:
            intent, conf = Intent.GOODBYE, 0.8
        elif question:
            intent, conf = Intent.HEALTH_QUESTION, 0.5
        elif possible:
            intent, conf = Intent.EMERGENCY_HELP, 0.6
            notes.append("ambiguous_emergency")

        if possible and intent not in (Intent.EMERGENCY_HELP,) and not question:
            secondary.append(Intent.EMERGENCY_HELP)
            notes.append("ambiguous_emergency")
        if affirmed and intent in (Intent.FALL_REPORT, Intent.EMERGENCY_HELP):
            if any(m.concept in _PAIN_LIKE for m in affirmed) or pain_score is not None:
                secondary.append(Intent.PAIN_REPORT)
        if exclamation and intent is not Intent.PAIN_EXCLAMATION:
            secondary.append(Intent.PAIN_EXCLAMATION)

        return NLUResult(
            utterance_id=utt.id, language=lang_result, intent=intent, intent_confidence=conf,
            secondary_intents=list(dict.fromkeys(secondary)), concepts=mentions, body_regions=body_regions,
            pain_score=pain_score, measurements=measurements, medication=medication, memory_fact=memory_fact,
            recall_target=recall_target, privacy_action=privacy_action, injection_suspected=injection,
            model_id=MODEL_ID, notes=list(dict.fromkeys(notes)),
        )

    # ------------------------------------------------------------------
    def _medication(self, tokens: list[Token], med: list[Match], lang: str) -> Optional[str]:
        m = med[0]
        if lang in ("hi", "hi-Latn") or (m.lang == "de" and m.phrase in ("genommen", "habe genommen", "eingenommen")):
            # name precedes the verb: "maine paracetamol le li", "ich habe Aspirin genommen"
            words = [t.text for t in tokens[max(0, m.start - 4):m.start] if t.clause == m.clause]
            words = [w for w in words if w not in _MED_DROP and w not in ("habe", "ich", "hab", "dawai", "dawa", "goli", "दवा", "दवाई", "गोली", "मैंने", "मैने")]
        else:
            words = []
            for t in tokens[m.end:m.end + 6]:
                if t.clause != m.clause or t.text in _MED_STOP:
                    break
                words.append(t.text)
            while words and words[0] in _MED_DROP:
                words.pop(0)
        words = words[:3]
        return " ".join(words) if words else None

    @staticmethod
    def _memory(norm: str, langs: set[str]) -> Optional[dict[str, str]]:
        for kind, by_lang in lx.MEMORY_PATTERNS.items():
            for lang in langs:
                pat = by_lang.get(lang)
                if not pat:
                    continue
                m = re.search(pat, norm)
                if m:
                    v = next((g for g in m.groups() if g), "").strip()
                    if v and len(v) <= 120:
                        return {"kind": kind, "value": v}
        return None

    @staticmethod
    def _recall_target(norm: str, mentions: list[ConceptMention]) -> Optional[str]:
        from .extract import _cue_positions
        for mtype in ("blood_pressure", "heart_rate", "temperature", "spo2", "glucose", "weight"):
            if _cue_positions(norm, mtype):
                return mtype
        for m in mentions:
            if m.concept is Concept.FALL:
                return "fall"
            if m.concept is Concept.PAIN:
                return "pain"
        if any(w in norm for w in ("medic", "pill", "dawai", "dawa", "दवा", "tablet", "medikament", "pastilla", "comprime")):
            return "medication_taken"
        if "pain" in norm or "dard" in norm or "दर्द" in norm or "dolor" in norm or "douleur" in norm or "schmerz" in norm:
            return "pain"
        return None
