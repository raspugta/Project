"""PII redaction applied to anything leaving the core (LLM context, alerts, logs)."""
from __future__ import annotations

import re

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")),
    ("URL", re.compile(r"https?://\S+|www\.\S+")),
    ("CARD", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("PHONE", re.compile(r"(?<![\w/])\+?\d[\d\s().-]{6,}\d(?![\w/])")),
    ("ID", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
]


_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _is_pii(label: str, s: str) -> bool:
    if label == "PHONE":
        return sum(c.isdigit() for c in s) >= 8 and not _DATE.search(s)
    return True


def redact(text: str) -> str:
    out = text
    for label, pat in _PATTERNS:
        out = pat.sub(lambda m, l=label: f"[{l}]" if _is_pii(l, m.group(0)) else m.group(0), out)
    return out


def contains_pii(text: str) -> list[str]:
    return [label for label, pat in _PATTERNS if any(_is_pii(label, m.group(0)) for m in pat.finditer(text))]
