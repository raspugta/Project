"""LLM output guard.

The LLM is treated as an untrusted text generator. Its reply is accepted only
if EVERY statement passes every check; otherwise the whole reply is dropped
and Baymax answers deterministically. Checks are multilingual and err on the
side of rejecting.

Violation codes
  bad_format              reply is not the required JSON
  too_long                more than MAX_STATEMENTS or MAX_CHARS
  wrong_language          statement not in the target language
  unknown_citation        cites an id that was not provided in context
  unsupported_number      a number that appears in no cited source / user text
  dosage                  gives a medicine amount
  unsupported_condition   names a disease/condition not present in cited text or user text
  diagnosis               diagnostic phrasing ("you have ...", "sounds like ...")
  false_observation       claims to see/hear/notice something without an obs: citation
  safety_contradiction    reassures / discourages seeking help
  action_claim            claims Baymax did something (called, alerted, booked)
  pii                     contains phone numbers, e-mails, URLs, ...
  role_break              claims to be a doctor / reveals instructions
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from ..language.detect import detect_language
from ..language.text import normalize
from ..privacy.redact import contains_pii

MAX_STATEMENTS = 3
MAX_CHARS = 320

CONDITIONS = [
    # en
    "heart attack", "stroke", "fracture", "broken bone", "broken", "concussion", "diabetes", "cancer", "tumor",
    "tumour", "infection", "flu", "influenza", "covid", "migraine", "pneumonia", "asthma", "appendicitis",
    "arthritis", "hypertension", "depression", "anxiety disorder", "dehydration", "sprain", "sepsis",
    "meningitis", "angina", "arrhythmia", "vertigo", "gastritis", "ulcer", "kidney stone", "anemia",
    "anaemia", "epilepsy", "allergy", "anaphylaxis", "food poisoning", "hernia", "gout", "sciatica",
    "tendinitis", "bronchitis", "sinusitis", "strep", "uti", "dementia", "parkinson",
    # hi / hi-Latn
    "दिल का दौरा", "स्ट्रोक", "फ्रैक्चर", "हड्डी टूट", "डायबिटीज", "कैंसर", "संक्रमण", "इन्फेक्शन", "माइग्रेन",
    "निमोनिया", "अस्थमा", "मधुमेह", "dil ka daura", "haddi toot", "sugar ki bimari", "infection", "cancer",
    # es
    "infarto", "ataque al corazon", "ictus", "fractura", "conmocion", "diabetes", "infeccion", "gripe",
    "neumonia", "asma", "apendicitis", "migrana", "deshidratacion", "esguince",
    # fr
    "crise cardiaque", "infarctus", "avc", "fracture", "commotion", "diabete", "infection", "grippe",
    "pneumonie", "asthme", "appendicite", "migraine", "deshydratation", "entorse",
    # de
    "herzinfarkt", "schlaganfall", "bruch", "gehirnerschutterung", "diabetes", "krebs", "infektion", "grippe",
    "lungenentzundung", "asthma", "blinddarmentzundung", "migrane", "verstauchung",
]

_DIAGNOSIS = re.compile(
    r"\b(you (probably |likely |most likely |might |may |could |definitely )(have|be having|be suffering from|suffer from)"
    r"|you (have|'ve got|are having|are suffering from|suffer from) (a |an )?(mild |severe |bad |serious )?(case|form|condition|disease|infection|injury|fracture|sprain|disorder|syndrome)"
    r"|sounds like (you have|a|an)|it('s| is) (probably|likely|definitely|most likely) (a |an )?"
    r"|my diagnosis|i diagnose|this is (a |an )?(sign|case) of"
    r"|aapko .{0,25}(bimari|rog|infection|sankraman)|आपको .{0,25}(बीमारी|रोग|इन्फेक्शन|संक्रमण)"
    r"|(usted |tu )?(tienes|tiene|padeces|sufres de)\b|parece (que tienes|un|una)"
    r"|vous avez (probablement |sans doute )?(un|une|de la|du)|c'est (probablement|surement) (un|une)"
    r"|du hast (wahrscheinlich |vermutlich )?(einen|eine|ein)|das ist (wahrscheinlich|vermutlich) (ein|eine))",
    re.IGNORECASE)
_DOSAGE = re.compile(
    r"\d+(?:[.,]\d+)?\s*(mg|milligrams?|mcg|µg|ml|millilit(er|re)s?|g\b|tablets?|pills?|capsules?|drops?|puffs?"
    r"|goli|goliyan|गोली|गोलियां|comprimidos?|pastillas?|comprimes?|gelules?|tabletten?|kapseln?)"
    r"|\b(take|nimm|prenez|toma|tomar|lijiye|लीजिए|le lo)\b.{0,25}\b(twice|three times|every \d|two|deux|dos|zwei|do baar|दो बार)\b",
    re.IGNORECASE)
_OBSERVATION = re.compile(
    r"\b(i (can )?see|i('m| am) (seeing|watching)|i (noticed|notice|saw|heard|detected)|looks like you|you look"
    r"|on (the )?camera|in the video|your face|your posture looks"
    r"|main dekh|maine dekha|मैं देख|मैंने देखा|veo que|he visto|te veo|je vois|j'ai vu|ich sehe|ich habe gesehen)",
    re.IGNORECASE)
_CONTRADICTION = re.compile(
    r"(no need to (call|see|go|worry|panic)|don'?t need to (call|see|go)|do not need to (call|see|go)|not necessary to (call|see)"
    r"|(it'?s|this is|that'?s) not (an emergency|serious)|nothing (serious|to worry)|it'?s (probably )?nothing|just relax"
    r"|(don'?t|do not) call|no need for (a|an) (doctor|ambulance)|you('ll| will) be (fine|ok)|not a big deal"
    r"|koi zaroorat nahi|chinta ki koi baat nahi|ghabrane ki koi baat nahi|कोई ज़रूरत नहीं|चिंता की कोई बात नहीं|घबराने की कोई बात नहीं"
    r"|no hace falta (llamar|ir)|no es (grave|nada)|no te preocupes|pas besoin d'(appeler|aller)|ce n'est (pas grave|rien)"
    r"|ne vous inquietez pas|kein grund zur sorge|nicht schlimm|du musst nicht (anrufen|zum arzt)|keine sorge)",
    re.IGNORECASE)
_ACTION = re.compile(
    r"\b(i('ve| have)? (already |just )?(called|alerted|notified|contacted|booked|scheduled|sent|messaged|dialed|dialled|informed)"
    r"|i('m| am| will) (now )?(calling|call|alert|notify|contact|book)|help is on (the|its) way|ambulance is (coming|on its way)"
    r"|maine .{0,25}(call|alert|bula|bulaya|inform|bhej)|मैंने .{0,25}(कॉल|सूचित|बुला|भेज)"
    r"|(he|hemos) .{0,20}(llamado|avisado|contactado)|(j'ai|nous avons) .{0,25}(appele|prevenu|alerte|contacte)"
    r"|ich habe .{0,30}(angerufen|benachrichtigt|alarmiert|verstandigt|informiert))",
    re.IGNORECASE)
_ROLE = re.compile(
    r"\b(as an ai|as a language model|i am a doctor|i'm a doctor|as your doctor|my instructions|system prompt|developer mode"
    r"|main doctor hoon|मैं डॉक्टर हूं|soy (un |una )?medic[oa]|je suis (un |une )?medecin|ich bin (ein |eine )?arzt)",
    re.IGNORECASE)
_NUM = re.compile(r"\d+(?:[.,]\d+)?")


@dataclass
class GuardVerdict:
    ok: bool
    violations: list[dict[str, Any]] = field(default_factory=list)
    statements: list[dict[str, Any]] = field(default_factory=list)


def _lang_ok(text: str, target: str) -> bool:
    has_deva = bool(re.search(r"[ऀ-ॿ]", text))
    if target == "hi":
        return has_deva
    if has_deva:
        return False
    r = detect_language(text)
    if r.lang != target and r.scores.get(r.lang, 0) > 0 and r.scores.get(target, 0) == 0:
        return False  # evidence for another language and none for the target
    if r.confidence < 0.45:
        return True  # not enough evidence to reject short neutral text
    if target == "hi-Latn":
        return r.lang in ("hi-Latn",) or (r.lang == "en" and r.scores.get("hi-Latn", 0) >= 1.0)
    return r.lang == target


def check(raw_reply: str, *, target_lang: str, context_ids: dict[str, str], user_text: str) -> GuardVerdict:
    """`context_ids` maps each citable id (kb:..., event:..., obs:...) to its text."""
    v: list[dict[str, Any]] = []
    try:
        obj = json.loads(_strip_fences(raw_reply))
        stmts = obj["statements"]
        assert isinstance(stmts, list)
        parsed = [{"text": str(s["text"]).strip(), "cites": [str(c) for c in s.get("cites", [])]} for s in stmts]
    except Exception:
        return GuardVerdict(False, [{"code": "bad_format"}])
    parsed = [p for p in parsed if p["text"]]
    if len(parsed) > MAX_STATEMENTS or any(len(p["text"]) > MAX_CHARS for p in parsed):
        v.append({"code": "too_long"})
    user_norm = normalize(user_text)
    for i, p in enumerate(parsed):
        text = p["text"]
        norm = normalize(text)
        cites = p["cites"]
        bad = [c for c in cites if c not in context_ids]
        if bad:
            v.append({"code": "unknown_citation", "i": i, "ids": bad})
        cited_text = normalize(" ".join(context_ids.get(c, "") for c in cites))
        support = cited_text + " " + user_norm
        if not _lang_ok(text, target_lang):
            v.append({"code": "wrong_language", "i": i})
        for n in _NUM.findall(norm):
            if n not in support:
                v.append({"code": "unsupported_number", "i": i, "value": n})
        if _DOSAGE.search(norm):
            v.append({"code": "dosage", "i": i})
        for term in CONDITIONS:
            t = normalize(term)
            if re.search(r"(?<![\w])" + re.escape(t) + r"(?![\w])", norm) and t not in support:
                v.append({"code": "unsupported_condition", "i": i, "term": term})
                break
        if _DIAGNOSIS.search(norm):
            v.append({"code": "diagnosis", "i": i})
        if _OBSERVATION.search(norm) and not any(c.startswith("obs:") for c in cites):
            v.append({"code": "false_observation", "i": i})
        if _CONTRADICTION.search(norm):
            v.append({"code": "safety_contradiction", "i": i})
        if _ACTION.search(norm):
            v.append({"code": "action_claim", "i": i})
        if contains_pii(text):
            v.append({"code": "pii", "i": i})
        if _ROLE.search(norm):
            v.append({"code": "role_break", "i": i})
    return GuardVerdict(ok=not v, violations=v, statements=parsed if not v else [])


def _strip_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s)
    i, j = s.find("{"), s.rfind("}")
    return s[i:j + 1] if i >= 0 and j > i else s
