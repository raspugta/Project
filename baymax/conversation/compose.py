"""Deterministic response composition with provenance.

Every sentence is created through one of these constructors, which is what
makes each statement traceable: phrase-table sentences carry `phrase:<key>`,
knowledge sentences carry `kb:<id>`, user-data echoes carry `event:<id>` /
`utt:<id>`, and model-inference sentences carry `obs:<id>` + model id +
confidence.
"""
from __future__ import annotations

from typing import Any, Optional

from ..contracts import KnowledgeHit, Statement
from ..ontology import Provenance
from ..language import phrases as ph


def phrase(key: str, lang: str, *, ref: Optional[str] = None, model_id: Optional[str] = None,
           confidence: Optional[float] = None, **params: Any) -> Statement:
    text = ph.say(key, lang, **params)
    if key in ph.INFERENCE_PHRASES and ref and ref.startswith("obs:"):
        return Statement(text=text, provenance=Provenance.MODEL, ref=ref, model_id=model_id,
                         confidence=confidence, health_related=True)
    if key in ph.USER_DATA_PHRASES:
        return Statement(text=text, provenance=Provenance.USER, ref=ref or f"phrase:{key}", health_related=True)
    if key in ph.HEALTH_PHRASES:
        return Statement(text=text, provenance=Provenance.KNOWLEDGE, ref=f"phrase:{key}", health_related=True)
    if key in ph.INFERENCE_PHRASES:  # inference phrase without an observation id (e.g. heard "ouch")
        return Statement(text=text, provenance=Provenance.MODEL, ref=ref or f"phrase:{key}", model_id=model_id,
                         confidence=confidence, health_related=True)
    return Statement(text=text, provenance=Provenance.SYSTEM, ref=f"phrase:{key}")


def knowledge(hit: KnowledgeHit) -> Statement:
    return Statement(text=hit.text, provenance=Provenance.KNOWLEDGE, ref=f"kb:{hit.entry_id}", health_related=True)
