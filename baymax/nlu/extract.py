"""Structured extraction of measurements and pain scores from user text.

Everything returned here is USER-provided data: the exact source span is kept
in `Measurement.raw`. Values outside physiological plausibility ranges are
returned separately as rejects and are never stored or "corrected".
"""
from __future__ import annotations

import re
from typing import Optional

from ..contracts import Measurement
from ..ontology import MEASUREMENT_PLAUSIBLE, MEASUREMENT_UNITS, MeasurementType as MT
from ..language import lexicon as lx
from ..language.text import split_phrases

_DEVA_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_NUM = r"(\d{1,3}(?:[.,]\d{1,2})?)"


def ascii_digits(text: str) -> str:
    return text.translate(_DEVA_DIGITS)


def _cue_positions(norm: str, mtype: str) -> list[int]:
    out: list[int] = []
    for cue in split_phrases(lx.MEASURE_CUES[mtype]):
        if cue.isascii():
            pat = r"(?<![a-z])" + re.escape(cue) + r"(?![a-z])"
            out += [m.end() for m in re.finditer(pat, norm)]
        else:
            start = 0
            while (i := norm.find(cue, start)) >= 0:
                out.append(i + len(cue))
                start = i + 1
    return sorted(out)


def _num_after(norm: str, pos: int, window: int = 28) -> Optional[re.Match[str]]:
    seg = norm[pos:pos + window]
    m = re.search(_NUM, seg)
    if not m:
        return None
    # do not jump over another sentence
    if re.search(r"[.!?;]\s", seg[:m.start()]):
        return None
    return m


def _f(s: str) -> float:
    return float(s.replace(",", "."))


def _plausible(t: MT, v: float) -> bool:
    lo, hi = MEASUREMENT_PLAUSIBLE[t]
    return lo <= v <= hi


def extract_measurements(norm_text: str) -> tuple[list[Measurement], list[str]]:
    """Returns (measurements, rejects). rejects are 'type:raw' strings."""
    norm = ascii_digits(norm_text)
    found: list[Measurement] = []
    rejects: list[str] = []
    used: list[tuple[int, int]] = []

    def overlaps(a: int, b: int) -> bool:
        return any(not (b <= x or a >= y) for x, y in used)

    def add(t: MT, v: float, raw: str, span: tuple[int, int], v2: Optional[float] = None) -> None:
        if overlaps(*span):
            return
        used.append(span)
        if not _plausible(t, v) or (v2 is not None and not (20 <= v2 <= 180 and v2 < v)):
            rejects.append(f"{t.value}:{raw.strip()}")
            return
        found.append(Measurement(type=t, value=round(v, 1), value2=v2, unit=MEASUREMENT_UNITS[t], raw=raw.strip()))

    # Blood pressure "120/80", "120 over 80"
    bp_cue = bool(_cue_positions(norm, "blood_pressure"))
    for m in re.finditer(r"(?<!\d)(\d{2,3})\s*(?:/|over|sur|uber|por|से|se)\s*(\d{2,3})(?!\d)", norm):
        sys_, dia = float(m.group(1)), float(m.group(2))
        if dia == 10 and not bp_cue:
            continue  # "7/10" is a pain score
        if bp_cue or (70 <= sys_ <= 260 and 30 <= dia <= 150 and sys_ > dia):
            add(MT.BLOOD_PRESSURE, sys_, m.group(0), m.span(), v2=dia)

    # Explicit units
    for m in re.finditer(_NUM + r"\s*(?:bpm|beats per minute|beats|schlage|latidos|battements)", norm):
        add(MT.HEART_RATE, _f(m.group(1)), m.group(0), m.span())
    for m in re.finditer(_NUM + r"\s*(?:°\s*)?(c|f|celsius|fahrenheit|degrees?|grad|degres?|grados?|डिग्री|degree)(?![a-z])", norm):
        v = _f(m.group(1))
        unit = m.group(2)
        if unit in ("f", "fahrenheit") or (unit not in ("c", "celsius") and v > 50):
            v = (v - 32) * 5 / 9
        add(MT.TEMPERATURE, v, m.group(0), m.span())
    for m in re.finditer(_NUM + r"\s*(mg/dl|mg dl|mg|mmol/l|mmol)(?![a-z])", norm):
        v = _f(m.group(1))
        if not _cue_positions(norm, "glucose"):
            continue  # "400 mg" is a medicine dose, not glucose
        if m.group(2).startswith("mmol"):
            v = v * 18.0
        add(MT.GLUCOSE, v, m.group(0), m.span())
    for m in re.finditer(_NUM + r"\s*(kg|kgs|kilos?|kilograms?|lbs?|pounds?)(?![a-z])", norm):
        v = _f(m.group(1))
        if m.group(2).startswith(("lb", "pound")):
            v = v * 0.4536
        add(MT.WEIGHT, v, m.group(0), m.span())
    for m in re.finditer(_NUM + r"\s*%", norm):
        if _cue_positions(norm, "spo2"):
            add(MT.SPO2, _f(m.group(1)), m.group(0), m.span())

    # Cue followed by a bare number
    for mtype, t in (("heart_rate", MT.HEART_RATE), ("temperature", MT.TEMPERATURE),
                     ("spo2", MT.SPO2), ("glucose", MT.GLUCOSE), ("weight", MT.WEIGHT)):
        for pos in _cue_positions(norm, mtype):
            m = _num_after(norm, pos)
            if not m:
                continue
            a, b = pos + m.start(), pos + m.end()
            if overlaps(a, b):
                continue
            v = _f(m.group(1))
            if t is MT.TEMPERATURE and v > 50:
                v = (v - 32) * 5 / 9
            raw = re.split(r"[.!?;,]", norm[max(0, pos - 24):b])[-1]
            add(t, v, raw, (a, b))
    for mtype, t in (("temperature", MT.TEMPERATURE), ("heart_rate", MT.HEART_RATE), ("glucose", MT.GLUCOSE)):
        for pos in _cue_positions(norm, mtype):
            cue_start = pos - 1
            while cue_start > 0 and norm[cue_start - 1] not in " ,.;!?":
                cue_start -= 1
            seg = norm[max(0, cue_start - 14):cue_start]
            m = re.search(_NUM + r"\s*(?:de|d'|of|grad|degrees?|°)?\s*$", seg)
            if not m:
                continue
            a = max(0, cue_start - 14) + m.start(1)
            b = a + len(m.group(1))
            if overlaps(a, b):
                continue
            v = _f(m.group(1))
            if t is MT.TEMPERATURE and v > 50:
                v = (v - 32) * 5 / 9
            add(t, v, norm[a:pos], (a, b))
    return found, rejects


_PAIN_CUES = r"(?:pain|hurts?|dard|dolor|douleur|schmerz\w*|दर्द|mal)"
_OUT_OF_10 = r"(?:/|out of|of|sur|von|de|में से|mein se|me se|में)"
_TEN = r"(?:10|ten|dix|diez|zehn|das|दस)"


def extract_pain_score(norm_text: str, lang: str, awaiting_score: bool, n_content_tokens: int) -> Optional[int]:
    norm = ascii_digits(norm_text)
    m = re.search(r"(?<![\d/])(10|[0-9])\s*" + _OUT_OF_10 + r"\s*" + _TEN + r"(?!\d)", norm)
    if m:
        return int(m.group(1))
    # Hindi order: "10 में से 3"
    m = re.search(r"(?:10|दस|das)\s*(?:में से|mein se|me se)\s*(10|[0-9])(?!\d)", norm)
    if m:
        return int(m.group(1))
    m = re.search(_PAIN_CUES + r"[^\d.!?]{0,18}?(?<![\d/])(10|[0-9])(?![\d/.,]\d)(?!\s*(?:%|bpm|kg|mg|°))", norm)
    if m:
        return int(m.group(1))
    words = lx.NUMBER_WORDS.get(lang, {}) | lx.NUMBER_WORDS["en"]
    m = re.search(r"(?<![\w])(" + "|".join(sorted(map(re.escape, words), key=len, reverse=True)) + r")\s*" + _OUT_OF_10 + r"\s*" + _TEN + r"(?!\d)", norm)
    if m:
        return words[m.group(1)]
    if awaiting_score:
        m = re.search(r"(?<![\d/])(10|[0-9])(?![\d])(?!\s*" + _OUT_OF_10 + r"\s*10)", norm)
        if m:
            return int(m.group(1))
        if n_content_tokens <= 4:
            for tok in re.split(r"\s+", norm):
                tok = tok.strip(".,!?")
                if tok in words:
                    return words[tok]
    return None
