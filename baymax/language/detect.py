"""Language identifier (model_id: lang_id_v1).

Deterministic: Unicode script evidence + distinctive function words + the
NLU lexicons themselves. Returns a LanguageResult restricted to
SUPPORTED_LANGS. See the model card in baymax/models/registry.py.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Optional

from ..contracts import LanguageResult
from ..ontology import SUPPORTED_LANGS
from . import lexicon as lx
from .text import normalize, split_phrases, tokenize

MODEL_ID = "lang_id_v1"

_FUNCTION_WORDS: dict[str, str] = {
    "en": "the|i|my|is|am|are|it|and|to|of|in|have|has|me|you|what|how|this|that|not|don't|can't|please|was|feel|with|on|for|a|an|i'm|im|it's|do|did|pain|yes|hurts|fell|help|should|can|your|be|been|very|just|about|there|here|when",
    "hi-Latn": "hai|hain|mujhe|mera|meri|mere|nahi|nahin|kya|ho|raha|rahi|gaya|gayi|bahut|dard|aap|kar|karo|ke|ki|ka|ko|bhi|yeh|ye|woh|wo|hoon|hu|tha|thi|chahiye|theek|thik|bachao|madad|accha|acha|haan|ji|mein|kuch|abhi|kaise|kyun|aur|lekin|toh|hua|hui|lag|laga|lagi|sab|kal|aaj|pe|par|wala|wali|diya|liya|li|le|kaha|bata|batao|mai|hum|tum|apna|apni|jaldi",
    "es": "el|la|los|las|de|que|y|en|un|una|me|mi|es|estoy|esta|tengo|duele|por|para|con|muy|se|lo|al|del|pero|como|yo|dolor|mucho|favor|soy|he|puedo|hay|ayuda|cuando|donde|tu|te|mis|ese|esa|esto|hoy|deja|dejar|quiero|necesito|puedes|estas|eres|tiene|tienes|ahora|aqui|bien|mal|todo|nada|algo|porque|cual|hola|gracias|escuchar|mirar|hacer|hago|siento|creo|mas|ya|tambien|sin|sobre|entre|desde|hasta",
    "fr": "le|la|les|de|des|du|et|je|j'ai|suis|est|un|une|pas|ne|me|mon|ma|mes|que|qui|pour|avec|dans|au|aux|mal|ca|c'est|vous|tu|oui|non|tres|ai|sur|moi|merci|bonjour|aide|n'est|il|elle|ce|cette|j'arrive|m'aider|peux|veux|fait|faire|avoir|etre|mais|plus|tout|rien|quelque|chose|depuis|aussi|bien|va|vais|arrete|ecouter|regarder|salut",
    "de": "der|die|das|und|ich|bin|ist|nicht|ein|eine|mein|meine|mir|mich|zu|mit|auf|habe|hab|es|wir|ja|nein|bitte|hilfe|tut|weh|kein|keine|schmerzen|danke|was|wie|auch|sehr|den|dem|von|im|geht|gut|heute|kann|hat|hallo|seit|noch|schon|aber|oder|wenn|dass|jetzt|hier|etwas|nichts|alles|mehr|viel|wieder|hore|horen|schauen",
}

_DIACRITICS = {
    "es": "ñ¿¡áíóú",
    "fr": "èêàçùûôîœëï",
    "de": "äöüß",
}
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_LATIN = re.compile(r"[A-Za-zÀ-ɏ]")


@lru_cache(maxsize=1)
def _exclamations() -> dict[str, set[str]]:
    from .text import squash
    out: dict[str, set[str]] = {}
    for lang, phrases in lx.PAIN_EXCLAMATIONS.items():
        for w in split_phrases(phrases):
            out.setdefault(squash(w), set()).add(lang)
    return out


@lru_cache(maxsize=1)
def _vocab() -> dict[str, dict[str, float]]:
    """word -> {lang: weight}. Words shared by several languages are split."""
    owners: dict[str, set[str]] = {}

    def add(lang: str, phrases: str, _weight: float = 1.0) -> None:
        for ph in split_phrases(phrases):
            for t in tokenize(ph):
                w = t.text.rstrip("*")
                if len(w) >= 1:
                    owners.setdefault(w, set()).add(lang)

    for lang, words in _FUNCTION_WORDS.items():
        add(lang, words)
    for table in (lx.CONCEPTS, lx.BODY):
        for by_lang in table.values():
            for lang, phrases in by_lang.items():
                if lang != "hi":
                    add(lang, phrases)
    tables = [lx.EMERGENCY_STRONG, lx.EMERGENCY_EXACT, lx.HELP_OPEN, lx.POSSIBLE_EMERGENCY, lx.USER_OK,
              lx.USER_OK_EXACT, lx.CONFIRM, lx.CANCEL, lx.GREETING, lx.GOODBYE, lx.PAIN_EXCLAMATIONS, lx.QUESTION,
              lx.NEGATORS, lx.HYPOTHETICAL, lx.RECALL, lx.MEDICATION_TAKEN, lx.DIAGNOSIS_REQUEST, lx.DOSAGE_REQUEST,
              lx.OTHER_SUBJECT, lx.PAST, lx.CLAUSE_BREAKERS, lx.FILLERS, lx.IDIOMS, *lx.PRIVACY.values()]
    for table in tables:
        for lang, phrases in table.items():
            if lang != "hi":
                add(lang, phrases)
    wake = {t.text for w in lx.WAKE_WORDS for t in tokenize(normalize(w))} | {"hey", "ok", "okay"}
    vocab: dict[str, dict[str, float]] = {}
    for w, langs in owners.items():
        if w in wake and w not in ("hey", "ok", "okay"):
            continue  # the assistant's name is not evidence of any language
        vocab[w] = {l: 1.0 / len(langs) for l in langs}
    return vocab


def detect_language(
    text: str,
    prior: Optional[str] = None,
    stt_language: Optional[str] = None,
) -> LanguageResult:
    scores = {l: 0.0 for l in SUPPORTED_LANGS}
    deva = len(_DEVANAGARI.findall(text))
    latin = len(_LATIN.findall(text))
    if deva and deva >= latin:
        scores["hi"] += 3.0 + deva / 4.0
    lower = text.lower()
    for lang, chars in _DIACRITICS.items():
        n = sum(lower.count(c) for c in chars)
        scores[lang] += 1.5 * n
    norm = normalize(text)
    tokens = [t.text for t in tokenize(norm)]
    vocab = _vocab()
    from .text import squash
    excl = _exclamations()
    if tokens and len(tokens) <= 3 and all(squash(t) in excl for t in tokens):
        # a bare pain sound ("au!", "aïe", "ayyy"): only exclamation vocabulary is evidence
        vocab = {squash(t): {l: 1.0 / len(excl[squash(t)]) for l in excl[squash(t)]} for t in tokens}
        tokens = [squash(t) for t in tokens]
    evidence_tokens = 0
    exclusive = {l: 0 for l in SUPPORTED_LANGS}
    for tok in tokens:
        if _DEVANAGARI.search(tok):
            continue
        w = vocab.get(tok)
        if w:
            evidence_tokens += 1
            for l, v in w.items():
                scores[l] += v
            if len(w) == 1:
                exclusive[next(iter(w))] += 1
    if stt_language:
        code = _map_stt_lang(stt_language, text)
        if code:
            scores[code] += 2.0
    total = sum(scores.values())
    if total <= 0:
        lang = prior if prior in SUPPORTED_LANGS else "en"
        return LanguageResult(lang=lang, confidence=0.0, model_id=MODEL_ID, scores=scores)
    # ties: more exclusive evidence wins, then the session's language
    lang = max(scores, key=lambda k: (round(scores[k], 6), exclusive[k], k == prior))
    conf = scores[lang] / total
    # Short texts carry little evidence.
    n_evidence = evidence_tokens + (3 if deva else 0) + (2 if stt_language else 0)
    conf *= min(1.0, max(n_evidence, 1) / 3.0)
    return LanguageResult(lang=lang, confidence=round(conf, 3), model_id=MODEL_ID,
                          scores={k: round(v, 3) for k, v in scores.items()})


def _map_stt_lang(code: str, text: str) -> Optional[str]:
    c = code.lower().split("-")[0]
    if c == "hi":
        return "hi" if _DEVANAGARI.search(text) else "hi-Latn"
    return c if c in SUPPORTED_LANGS else None


def resolve_language(result: LanguageResult, session_lang: Optional[str],
                     preferred: Optional[str], threshold: float = 0.5) -> str:
    """Pick the response language for this turn.

    An explicit user preference always wins. Otherwise a confident detection
    wins, else the session's last confident language, else English.
    """
    if preferred and preferred != "auto" and preferred in SUPPORTED_LANGS:
        return preferred
    if result.confidence >= threshold:
        return result.lang
    if session_lang in SUPPORTED_LANGS:
        return session_lang  # type: ignore[return-value]
    return result.lang if result.confidence > 0 else "en"
