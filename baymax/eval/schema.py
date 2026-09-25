"""Evaluation dataset schema (see datasets/SCHEMA.md).

One JSON object per line. A case runs `setup` utterances (and `seed` ones that
populate the health database) on a fresh engine, then the case `text`, and
checks the resulting trace against `expect`.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

from ..ontology import SUPPORTED_LANGS, Concept, Intent, SafetyState


class Expect(BaseModel):
    intent: Optional[str] = None
    intent_in: list[str] = []
    lang: Optional[str] = None                      # expected RESPONSE language
    concepts: list[str] = []                        # must be present and affirmed
    negated: list[str] = []                         # must be present and negated
    not_affirmed: list[str] = []                    # must not be affirmed
    pain_score: Optional[int] = None
    measurements: list[dict] = []                   # {"type", "value", "value2"?}
    state: Optional[str] = None                     # exact safety state after the turn
    state_min: Optional[str] = None                 # at least this state rank
    state_max: Optional[str] = None                 # at most this state rank
    escalate: Optional[bool] = None                 # True: reach >= escalation_countdown; False: must not
    refs: list[str] = []                            # response must contain a statement whose ref starts with each
    no_refs: list[str] = []                         # ... and none starting with these
    must_not_match: list[str] = []                  # regexes that must not match the response text
    event_kinds: list[str] = []                     # health events that must be created
    no_llm_statements: bool = False

    @field_validator("intent")
    @classmethod
    def _intent(cls, v):
        if v is not None:
            Intent(v)
        return v

    @field_validator("intent_in")
    @classmethod
    def _intents(cls, v):
        for x in v:
            Intent(x)
        return v

    @field_validator("concepts", "negated", "not_affirmed")
    @classmethod
    def _concepts(cls, v):
        for x in v:
            Concept(x)
        return v

    @field_validator("state", "state_min", "state_max")
    @classmethod
    def _state(cls, v):
        if v is not None:
            SafetyState(v)
        return v

    @field_validator("lang")
    @classmethod
    def _lang(cls, v):
        if v is not None and v not in SUPPORTED_LANGS:
            raise ValueError(f"unsupported language {v}")
        return v


class Case(BaseModel):
    id: str
    source: Literal["curated", "synthetic", "adversarial"]
    category: str
    lang: str
    text: str
    setup: list[str] = []
    seed: list[str] = []
    modality: Literal["text", "speech"] = "text"
    wake_word: bool = True
    llm_reply: Optional[str] = None                 # scripted LLM reply for this turn (attack injection)
    expect: Expect = Field(default_factory=Expect)
    tags: list[str] = []
    note: str = ""

    @field_validator("lang")
    @classmethod
    def _lang(cls, v):
        if v not in SUPPORTED_LANGS:
            raise ValueError(f"unsupported language {v}")
        return v


class GuardCase(BaseModel):
    """A candidate LLM output for the output guard (llm_attack_outputs.jsonl)."""

    id: str
    category: str
    target_lang: str
    user_text: str
    context: dict[str, str] = {}
    output: str
    expect: Literal["block", "allow"]
    codes: list[str] = []                           # at least one of these violation codes when blocked
