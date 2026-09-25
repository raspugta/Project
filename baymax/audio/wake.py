"""Wake-word matcher on transcripts (model_id: wake_word_v1)."""
from __future__ import annotations

from dataclasses import dataclass

from ..language import lexicon as lx
from ..language.text import normalize, tokenize

MODEL_ID = "wake_word_v1"
_VARIANTS = sorted({normalize(w).replace(" ", "") for w in lx.WAKE_WORDS} | {"baymax"}, key=len, reverse=True)
_PREFIXES = {"hey", "ok", "okay", "hi", "hello", "arre", "oye", "he", "hallo"}


def _lev(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


@dataclass
class WakeResult:
    found: bool
    remainder: str
    confidence: float


def detect_wake(text: str) -> WakeResult:
    toks = [t.text for t in tokenize(normalize(text))]
    best = (0.0, -1, 0)
    for i in range(len(toks)):
        for span in (1, 2):
            if i + span > len(toks):
                continue
            cand = "".join(toks[i:i + span])
            if len(cand) < 4:
                continue
            for v in _VARIANTS:
                d = _lev(cand, v)
                score = 1 - d / max(len(v), len(cand))
                if score > best[0]:
                    best = (score, i, span)
    score, i, span = best
    if score >= 0.8:
        start = i - 1 if i > 0 and toks[i - 1] in _PREFIXES else i
        rest = toks[:start] + toks[i + span:]
        # keep the original text for the remainder when the wake word was at the start
        return WakeResult(True, " ".join(rest), round(score, 3))
    return WakeResult(False, text, round(score, 3))
