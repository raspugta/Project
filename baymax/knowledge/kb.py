"""Structured Baymax knowledge base + retrieval (model_id: retrieval_bm25_v1).

Knowledge entries are curated data, not model output. Retrieval ranks them by
BM25 over the entry text in the response language plus a strong boost for
matching ontology concepts, so a Hindi question about "सीने में दर्द" finds
kb.chest_pain through the concept id even if wording differs.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional

from pydantic import BaseModel

from ..contracts import KnowledgeHit
from ..ontology import SUPPORTED_LANGS, Concept
from ..language.text import normalize, tokenize

MODEL_ID = "retrieval_bm25_v1"
KB_PATH = Path(__file__).parent / "data" / "kb.json"
CONCEPT_BOOST = 3.0
MIN_SCORE = 1.0


class KBEntry(BaseModel):
    id: str
    concepts: list[str]
    red_flag: bool
    source: str
    review_status: str
    title: dict[str, str]
    text: dict[str, str]
    keywords: dict[str, str] = {}


class KnowledgeBase:
    def __init__(self, path: Path = KB_PATH) -> None:
        raw = json.loads(path.read_text(encoding="utf-8"))
        self.version: str = raw["version"]
        self.notice: str = raw["notice"]
        self.entries: dict[str, KBEntry] = {e["id"]: KBEntry(**e) for e in raw["entries"]}
        self._index: dict[str, _BM25] = {}
        for lang in SUPPORTED_LANGS:
            docs = {eid: _terms(e.text.get(lang, "") + " " + e.title.get(lang, "") + " " + e.keywords.get(lang, ""))
                    for eid, e in self.entries.items()}
            self._index[lang] = _BM25(docs)

    def validate(self) -> list[str]:
        problems = []
        valid_concepts = {c.value for c in Concept}
        for e in self.entries.values():
            for lang in SUPPORTED_LANGS:
                if not e.text.get(lang):
                    problems.append(f"{e.id}: missing text for {lang}")
                if not e.title.get(lang):
                    problems.append(f"{e.id}: missing title for {lang}")
            for c in e.concepts:
                if c not in valid_concepts:
                    problems.append(f"{e.id}: unknown concept {c}")
            if not e.source:
                problems.append(f"{e.id}: missing source")
        return problems

    def get(self, entry_id: str) -> Optional[KBEntry]:
        return self.entries.get(entry_id)

    def by_concept(self, concept: str) -> list[KBEntry]:
        return [e for e in self.entries.values() if concept in e.concepts]

    def retrieve(self, query: str, lang: str, concepts: Iterable[str] = (), k: int = 2) -> list[KnowledgeHit]:
        concepts = set(concepts)
        scores = self._index[lang].score(_terms(query))
        ranked = []
        for eid, e in self.entries.items():
            s = scores.get(eid, 0.0) + CONCEPT_BOOST * len(concepts & set(e.concepts))
            if s >= MIN_SCORE:
                ranked.append((s, eid))
        ranked.sort(reverse=True)
        return [self.hit(eid, lang, score) for score, eid in ranked[:k]]

    def hit(self, entry_id: str, lang: str, score: float = 0.0) -> KnowledgeHit:
        e = self.entries[entry_id]
        return KnowledgeHit(entry_id=e.id, title=e.title[lang], text=e.text[lang], score=round(score, 3),
                            concepts=e.concepts, source=e.source, review_status=e.review_status)


def _terms(text: str) -> list[str]:
    return [t.text for t in tokenize(normalize(text)) if len(t.text) > 2 or not t.text.isascii()]


class _BM25:
    def __init__(self, docs: dict[str, list[str]], k1: float = 1.4, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.tf = {d: Counter(t) for d, t in docs.items()}
        self.len = {d: len(t) for d, t in docs.items()}
        self.avg = (sum(self.len.values()) / len(self.len)) if self.len else 1.0
        df: Counter[str] = Counter()
        for t in docs.values():
            df.update(set(t))
        n = len(docs)
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}

    def score(self, q: list[str]) -> dict[str, float]:
        out: dict[str, float] = {}
        for d, tf in self.tf.items():
            s = 0.0
            for w in set(q):
                f = tf.get(w, 0)
                if f:
                    s += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[d] / self.avg))
            if s:
                out[d] = s
        return out


@lru_cache(maxsize=1)
def default_kb() -> KnowledgeBase:
    return KnowledgeBase()
