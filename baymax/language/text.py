"""Text normalisation, tokenisation and lexicon phrase matching."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Optional

_CLAUSE_PUNCT = set(".,!?;:¿¡…।\n")
_SPLIT = re.compile(r"[\s.,!?;:¿¡…।\"“”«»()\[\]{}<>/\\|+=~`^@#$%&]+")
_NUKTA = "़"
_CHANDRABINDU = "ँ"
_ANUSVARA = "ं"


def _is_latin_letter(ch: str) -> bool:
    return ch.isalpha() and ord(ch) < 0x250


def normalize(text: str) -> str:
    """Lowercase; fold Latin accents; fold Devanagari nukta/chandrabindu;
    unify apostrophes; hyphens become spaces; ß -> ss."""
    t = text.replace("ß", "ss").replace("ẞ", "ss")
    t = t.replace("’", "'").replace("‘", "'").replace("`", "'").replace("´", "'")
    t = t.replace("-", " ").replace("‐", " ").replace("–", " ").replace("—", " ")
    t = unicodedata.normalize("NFKD", t)
    out: list[str] = []
    last_base = ""
    for ch in t:
        if unicodedata.category(ch) == "Mn" and _is_latin_letter(last_base):
            continue
        if ch == _NUKTA:
            continue
        if ch == _CHANDRABINDU:
            ch = _ANUSVARA
        out.append(ch)
        if unicodedata.category(ch) not in ("Mn", "Mc"):
            last_base = ch
    t = unicodedata.normalize("NFC", "".join(out)).lower()
    t = t.replace("œ", "oe").replace("æ", "ae")
    return re.sub(r"[ \t]+", " ", t).strip()


@dataclass(frozen=True)
class Token:
    text: str
    pos: int
    clause: int


def tokenize(norm_text: str, clause_breakers: Iterable[str] = ()) -> list[Token]:
    """Split normalised text into tokens; clause index increments at clause
    punctuation and at clause-breaking conjunctions ("but", "lekin", ...)."""
    breakers = set(clause_breakers)
    tokens: list[Token] = []
    clause = 0
    pos = 0
    last = 0
    for m in _SPLIT.finditer(norm_text + " "):
        word = norm_text[last:m.start()].strip("'")
        sep = m.group(0)
        if word:
            if word in breakers and tokens:
                clause += 1
            tokens.append(Token(word, pos, clause))
            pos += 1
        if any(c in _CLAUSE_PUNCT for c in sep):
            clause += 1
        last = m.end()
    return tokens


def squash(word: str) -> str:
    """Collapse repeated characters: 'ouuuch' -> 'ouch', 'owww' -> 'ow'."""
    return re.sub(r"(.)\1+", r"\1", word)


@dataclass(frozen=True)
class Match:
    key: str
    lang: str
    start: int
    end: int          # exclusive token index
    clause: int
    phrase: str


@dataclass(frozen=True)
class _Phrase:
    parts: tuple[str, ...]
    key: str
    lang: str
    raw: str


def split_phrases(s: str) -> list[str]:
    return [p.strip() for p in s.split("|") if p.strip()]


# Tokens that may appear between the words of a multi-word phrase without
# breaking the match: "I (just) fell", "I (suddenly) can't breathe".
GAP_TOKENS = frozenset({"just", "really", "suddenly", "accidentally", "actually", "totally", "kind", "of",
                        "abhi", "bas", "achanak", "अभी", "अचानक", "bas", "ya", "ja", "doch", "gerade", "plotzlich",
                        "de", "nuevo", "soudain", "vraiment"})


class PhraseIndex:
    """Index of multi-token phrases, keyed by their first token."""

    def __init__(self) -> None:
        self._by_first: dict[str, list[_Phrase]] = {}
        self._prefix: list[_Phrase] = []

    def add(self, key: str, lang: str, phrase: str) -> None:
        parts = tuple(t.text for t in tokenize(phrase))
        if not parts:
            return
        p = _Phrase(parts, key, lang, phrase)
        if parts[0].endswith("*"):
            self._prefix.append(p)
        else:
            self._by_first.setdefault(parts[0], []).append(p)

    def add_lexicon(self, key: str, by_lang: dict[str, str]) -> None:
        for lang, phrases in by_lang.items():
            for ph in split_phrases(phrases):
                self.add(key, lang, ph)

    @classmethod
    def _match_at(cls, parts: tuple[str, ...], tokens: list[Token], i: int) -> Optional[int]:
        """Match parts starting at token i, allowing at most one gap token between parts."""
        j = i
        gaps = 0
        for k, part in enumerate(parts):
            if j >= len(tokens):
                return None
            if k > 0 and not cls._part_eq(part, tokens[j].text) and tokens[j].text in GAP_TOKENS and gaps == 0 \
                    and part not in GAP_TOKENS:
                gaps += 1
                j += 1
                if j >= len(tokens):
                    return None
            if not cls._part_eq(part, tokens[j].text):
                return None
            if k > 0 and tokens[j].clause != tokens[i].clause:
                return None
            j += 1
        return j

    @staticmethod
    def _part_eq(part: str, tok: str) -> bool:
        if part.endswith("*"):
            return tok.startswith(part[:-1])
        return part == tok

    def find(self, tokens: list[Token], langs: Iterable[str] | None = None) -> list[Match]:
        allowed = set(langs) if langs is not None else None
        out: list[Match] = []
        n = len(tokens)
        for i, tok in enumerate(tokens):
            cands = list(self._by_first.get(tok.text, ()))
            cands += [p for p in self._prefix if tok.text.startswith(p.parts[0][:-1])]
            for p in cands:
                if allowed is not None and p.lang not in allowed:
                    continue
                end = self._match_at(p.parts, tokens, i)
                if end is not None:
                    out.append(Match(p.key, p.lang, i, end, tok.clause, p.raw))
        return _keep_longest(out)


def _keep_longest(matches: list[Match]) -> list[Match]:
    """For matches of the same key that overlap, keep the longest."""
    matches = sorted(matches, key=lambda m: (-(m.end - m.start), m.start))
    kept: list[Match] = []
    for m in matches:
        if any(k.key == m.key and not (m.end <= k.start or m.start >= k.end) for k in kept):
            continue
        kept.append(m)
    return sorted(kept, key=lambda m: m.start)


@lru_cache(maxsize=None)
def word_set(phrases: str) -> frozenset[str]:
    return frozenset(split_phrases(phrases))
