"""Synthetic dataset generator.

Combinatorial templates x slot values x languages x perturbations (casing,
lost punctuation, fillers, wake-word prefixes, polite suffixes) produce
labelled cases. The templates are deliberately phrased differently from the
NLU lexicon entries where possible, so this set measures generalisation of the
rule-based NLU rather than echoing its vocabulary. Deterministic for a seed.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

REGIONS = {
    "en": [("knee", "knee"), ("lower back", "back"), ("left shoulder", "shoulder"), ("ankle", "ankle"), ("wrist", "wrist"), ("neck", "neck"), ("stomach", "abdomen")],
    "hi-Latn": [("ghutne", "knee"), ("kamar", "back"), ("kandhe", "shoulder"), ("takhne", "ankle"), ("kalai", "wrist"), ("gardan", "neck"), ("pet", "abdomen")],
    "hi": [("घुटने", "knee"), ("कमर", "back"), ("कंधे", "shoulder"), ("टखने", "ankle"), ("कलाई", "wrist"), ("गर्दन", "neck"), ("पेट", "abdomen")],
    "es": [("rodilla", "knee"), ("espalda", "back"), ("hombro", "shoulder"), ("tobillo", "ankle"), ("muñeca", "wrist"), ("cuello", "neck"), ("estómago", "abdomen")],
    "fr": [("genou", "knee"), ("dos", "back"), ("épaule", "shoulder"), ("cheville", "ankle"), ("poignet", "wrist"), ("cou", "neck"), ("ventre", "abdomen")],
    "de": [("Knie", "knee"), ("Rücken", "back"), ("Schulter", "shoulder"), ("Knöchel", "ankle"), ("Handgelenk", "wrist"), ("Nacken", "neck"), ("Bauch", "abdomen")],
}

PAIN_WITH_SCORE = {
    "en": ["my {r} hurts, I'd say {n} out of 10", "pain in my {r}, about {n}/10", "my {r} is sore, maybe a {n} out of ten", "I have pain in my {r}, {n} out of 10"],
    "hi-Latn": ["mere {r} mein dard hai, {n} out of 10", "{r} mein dard ho raha hai, {n}/10"],
    "hi": ["मेरे {r} में दर्द है, 10 में से {n}", "{r} में दर्द हो रहा है, {n}/10"],
    "es": ["me duele la {r}, un {n} de 10", "tengo dolor en el {r}, {n}/10"],
    "fr": ["j'ai mal au {r}, {n} sur 10", "douleur au {r}, environ {n}/10"],
    "de": ["mein {r} tut weh, {n} von 10", "Schmerzen im {r}, etwa {n}/10"],
}
PAIN_NO_SCORE = {
    "en": ["my {r} really hurts", "I've got a sore {r}", "there's a sharp pain in my {r}", "my {r} aches"],
    "hi-Latn": ["mere {r} mein bahut dard hai", "{r} dukh raha hai"],
    "hi": ["मेरे {r} में बहुत दर्द है", "{r} में दर्द है"],
    "es": ["me duele mucho la {r}", "tengo dolor de {r}"],
    "fr": ["j'ai très mal au {r}", "mon {r} me fait mal"],
    "de": ["mein {r} tut sehr weh", "ich habe Schmerzen im {r}"],
}
EMERGENCY = {
    "en": ["please help me, my chest hurts so much", "I can't breathe properly, help", "call an ambulance, my wife collapsed",
           "my husband is having a seizure", "I think my mother is having a stroke, her face is drooping", "I'm bleeding a lot and it won't stop",
           "I took too many pills", "someone help, he's not breathing", "chest pain, spreading to my arm"],
    "hi-Latn": ["bachao, seene mein bahut dard hai", "mujhe saans nahi aa rahi, madad karo", "ambulance bulao, papa behosh ho gaye",
                "khoon ruk nahi raha", "meri maa ko daura pad raha hai"],
    "hi": ["बचाओ, सीने में बहुत दर्द है", "मुझे सांस नहीं आ रही", "एम्बुलेंस बुलाओ, पापा बेहोश हो गए", "खून रुक नहीं रहा"],
    "es": ["ayúdenme, me duele mucho el pecho", "no puedo respirar bien", "llamen a una ambulancia, mi madre se desmayó", "mi hijo tiene convulsiones"],
    "fr": ["au secours, j'ai une douleur à la poitrine", "je n'arrive pas à respirer", "appelez une ambulance, mon père ne respire plus", "elle a une crise d'épilepsie"],
    "de": ["Hilfe, ich habe starke Brustschmerzen", "ich bekomme keine Luft", "ruft einen Krankenwagen, mein Vater ist bewusstlos", "er hat einen Krampfanfall"],
}
FALL = {
    "en": ["I just fell in the kitchen", "I tripped over the rug and fell", "I slipped on the stairs", "I fell off the chair"],
    "hi-Latn": ["main kitchen mein gir gaya", "main seedhiyon par fisal gayi", "main gir gayi"],
    "hi": ["मैं रसोई में गिर गया", "मैं सीढ़ियों पर फिसल गई"],
    "es": ["me caí en el baño", "me resbalé en la escalera"],
    "fr": ["je suis tombé dans la cuisine", "j'ai glissé dans la salle de bain"],
    "de": ["ich bin in der Küche hingefallen", "ich bin auf der Treppe ausgerutscht"],
}
VITALS = {
    "en": [("my blood pressure is {s}/{d}", "bp"), ("BP {s} over {d}", "bp"), ("pulse is {hr}", "hr"), ("heart rate {hr} bpm", "hr"),
           ("temperature {t} degrees", "t"), ("sugar {g} mg/dl", "g"), ("oxygen {o}%", "o")],
    "hi-Latn": [("mera bp {s}/{d} hai", "bp"), ("meri nadi {hr} hai", "hr"), ("sugar {g} hai", "g")],
    "hi": [("मेरा बीपी {s}/{d} है", "bp"), ("बुखार {t} डिग्री है", "t")],
    "es": [("mi presión es {s}/{d}", "bp"), ("mi pulso es {hr}", "hr"), ("tengo fiebre de {t} grados", "t")],
    "fr": [("ma tension est de {s}/{d}", "bp"), ("mon pouls est à {hr}", "hr"), ("j'ai {t} de fièvre", "t")],
    "de": [("mein Blutdruck ist {s}/{d}", "bp"), ("mein Puls ist {hr}", "hr"), ("ich habe {t} Grad Fieber", "t")],
}
NEG_CHEST = {
    "en": ["I don't have any chest pain", "no chest pain at all", "there is no pain in my chest"],
    "hi-Latn": ["seene mein koi dard nahi hai"],
    "hi": ["सीने में कोई दर्द नहीं है"],
    "es": ["no tengo ningún dolor en el pecho"],
    "fr": ["je n'ai aucune douleur à la poitrine"],
    "de": ["ich habe keine Schmerzen in der Brust"],
}
OUCH = {"en": ["ouch", "ow", "owwww", "ouch!"], "hi-Latn": ["aah", "uff", "ahhh"], "hi": ["आह", "उफ़"], "es": ["ay", "¡ay!", "ayyy"],
        "fr": ["aïe", "ouille"], "de": ["aua", "autsch", "au!"]}
GREET = {"en": ["hi there", "good morning Baymax", "hello"], "hi-Latn": ["namaste", "namaskar baymax"], "hi": ["नमस्ते", "नमस्कार"],
         "es": ["buenos días", "hola Baymax"], "fr": ["bonjour", "salut Baymax"], "de": ["guten Morgen", "hallo"]}
QUESTION_BURN = {"en": ["how should I treat a small burn?", "what do I do about a burn on my finger?"],
                 "hi": ["जलने का इलाज कैसे करें?"], "es": ["¿cómo trato una quemadura?"], "fr": ["comment soigner une brûlure ?"],
                 "de": ["wie behandle ich eine Verbrennung?"], "hi-Latn": ["haath jal gaya, kya karun?"]}

FILLERS = {"en": ["um ", "uh ", "so ", ""], "hi-Latn": ["arre ", "yaar ", ""], "hi": ["अरे ", ""], "es": ["eh ", "oye ", ""],
           "fr": ["euh ", ""], "de": ["äh ", "also ", ""]}


def _perturb(rng: random.Random, text: str, lang: str) -> str:
    t = text
    r = rng.random()
    if r < 0.2:
        t = t.lower()
    elif r < 0.3:
        t = t.upper() if lang != "hi" else t
    if rng.random() < 0.4:
        t = t.rstrip("!?.")
    if rng.random() < 0.3:
        t = rng.choice(FILLERS[lang]) + t
    return t


def generate(seed: int = 2026, per_template: int = 2) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []

    def add(cat: str, lang: str, text: str, expect: dict, setup: list[str] | None = None):
        out.append({"id": f"syn-{len(out):04d}", "source": "synthetic", "category": cat, "lang": lang,
                    "text": _perturb(rng, text, lang), "setup": setup or [], "expect": expect})

    for lang in REGIONS:
        for tmpl in PAIN_WITH_SCORE[lang]:
            for _ in range(per_template):
                word, _region = rng.choice(REGIONS[lang])
                n = rng.randint(1, 7)
                add("pain_score", lang, tmpl.format(r=word, n=n), {"intent": "pain_report", "pain_score": n, "lang": lang, "escalate": False})
        for tmpl in PAIN_NO_SCORE[lang]:
            word, _ = rng.choice(REGIONS[lang])
            add("pain_no_score", lang, tmpl.format(r=word), {"intent": "pain_report", "state": "assisting", "lang": lang, "escalate": False})
        for text in EMERGENCY[lang]:
            add("emergency", lang, text, {"intent": "emergency_help", "escalate": True, "lang": lang})
        for text in FALL[lang]:
            add("fall", lang, text, {"intent": "fall_report", "state_min": "check_in", "lang": lang})
        for tmpl, kind in VITALS[lang]:
            for _ in range(per_template):
                vals = {"s": rng.randint(95, 175), "d": rng.randint(55, 105), "hr": rng.randint(48, 130),
                        "t": round(rng.uniform(36.2, 39.8), 1), "g": rng.randint(70, 260), "o": rng.randint(88, 100)}
                if vals["d"] >= vals["s"]:
                    vals["d"] = vals["s"] - 30
                text = tmpl.format(**vals)
                if lang in ("fr", "de", "es") and kind == "t":
                    text = text.replace(".", ",")
                m = {"bp": {"type": "blood_pressure", "value": vals["s"], "value2": vals["d"]},
                     "hr": {"type": "heart_rate", "value": vals["hr"]}, "t": {"type": "temperature", "value": vals["t"]},
                     "g": {"type": "glucose", "value": vals["g"]}, "o": {"type": "spo2", "value": vals["o"]}}[kind]
                add("vital", lang, text, {"intent": "vital_report", "measurements": [m], "lang": lang})
        for text in NEG_CHEST[lang]:
            add("negation", lang, text, {"escalate": False, "not_affirmed": ["chest_pain"]})
        for text in OUCH[lang]:
            add("ouch", lang, text, {"intent": "pain_exclamation", "state": "assisting"})
        for text in GREET[lang]:
            add("greeting", lang, text, {"intent": "greeting", "lang": lang})
        for text in QUESTION_BURN[lang]:
            add("question", lang, text, {"refs": ["kb:kb.burn"], "lang": lang, "escalate": False})
        # slot filling: "ouch" then a bare number
        for _ in range(per_template):
            n = rng.randint(0, 7)
            add("slot_fill", lang, str(n), {"intent": "pain_report", "pain_score": n, "state": "monitoring"},
                setup=[rng.choice(OUCH[lang])])
    return out


def write(path: Path, seed: int = 2026) -> int:
    cases = generate(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    return len(cases)
