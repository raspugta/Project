"""System interfaces.

Every stage of the pipeline consumes and produces one of these typed
records. Stages never pass free-form dicts to each other. Each record that is
the output of a model carries `model_id` + `confidence` so any downstream
statement can be traced back to the exact component that produced it.
"""
from __future__ import annotations

import time
import uuid
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from .ontology import (
    BodyRegion,
    Concept,
    EventKind,
    Intent,
    MeasurementType,
    Modality,
    Provenance,
    SafetyState,
    Severity,
)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def now() -> float:
    return time.time()


# --------------------------------------------------------------------------
# Inputs from sensors
# --------------------------------------------------------------------------

class Utterance(BaseModel):
    """One user turn from STT or the keyboard."""

    id: str = Field(default_factory=lambda: new_id("utt"))
    text: str
    modality: Modality = Modality.TEXT
    stt_model_id: Optional[str] = None        # e.g. "browser_webspeech", "faster_whisper:small"
    stt_confidence: Optional[float] = None    # 0..1 as reported by the STT backend (None = unknown)
    stt_language: Optional[str] = None        # language reported by the STT backend, if any
    ts: float = Field(default_factory=now)
    wake_word: bool = False                   # was the wake word present


class BodySample(BaseModel):
    """One frame of body-position features computed in the browser.

    Raw camera frames never leave the browser; only these numbers do.
    Coordinates are normalised to the image (0,0 top-left .. 1,1 bottom-right).
    """

    t: float                                   # seconds (client clock, monotonic within a session)
    present: bool
    source: Literal["pose", "motion"] = "pose"
    center_y: Optional[float] = None           # hip/torso centre (pose) or blob centroid (motion)
    center_x: Optional[float] = None
    head_y: Optional[float] = None
    bbox_w: Optional[float] = None
    bbox_h: Optional[float] = None
    torso_angle: Optional[float] = None        # deg from vertical; 0 upright, 90 horizontal (pose only)
    neck_forward: Optional[float] = None       # deg of ear ahead of shoulder (pose only)
    shoulder_tilt: Optional[float] = None      # deg (pose only)
    visibility: float = 1.0                    # mean landmark visibility / blob quality 0..1
    motion: Optional[float] = None             # motion energy 0..1
    persons: int = 1


class AudioLevel(BaseModel):
    t: float
    rms_db: float        # dBFS, typically -90..0


# --------------------------------------------------------------------------
# Model outputs
# --------------------------------------------------------------------------

class ConceptMention(BaseModel):
    concept: Concept
    negated: bool = False
    hypothetical: bool = False            # question / conditional ("what if ...")
    past: bool = False                    # explicitly in the past ("yesterday")
    subject: Literal["self", "other", "unknown"] = "self"
    span: str = ""
    body_region: Optional[BodyRegion] = None


class Measurement(BaseModel):
    type: MeasurementType
    value: float
    value2: Optional[float] = None        # diastolic for BP
    unit: str
    raw: str                              # the exact user text span it was read from


class LanguageResult(BaseModel):
    lang: str
    confidence: float
    model_id: str = "lang_id_v1"
    scores: dict[str, float] = {}


class NLUResult(BaseModel):
    """Output of the intent + extraction models for one utterance."""

    utterance_id: str
    language: LanguageResult
    intent: Intent
    intent_confidence: float
    secondary_intents: list[Intent] = []
    concepts: list[ConceptMention] = []
    body_regions: list[BodyRegion] = []
    pain_score: Optional[int] = None
    measurements: list[Measurement] = []
    medication: Optional[str] = None
    memory_fact: Optional[dict[str, str]] = None   # {"kind": "allergy", "value": "penicillin"}
    recall_target: Optional[str] = None            # MeasurementType / EventKind value being asked about
    privacy_action: Optional[str] = None           # pause_camera, pause_mic, resume, forget_last, delete_all
    injection_suspected: bool = False
    model_id: str = "nlu_lexicon_v1"
    notes: list[str] = []


class PerceptionEvent(BaseModel):
    """Output of a perception model (fall/activity/posture/acoustic)."""

    id: str = Field(default_factory=lambda: new_id("obs"))
    kind: Literal["fall", "lying_inactive", "posture_poor", "activity", "loud_impact", "absent_long"]
    model_id: str
    confidence: float
    ts: float = Field(default_factory=now)
    evidence: dict[str, Any] = {}
    failure_flags: list[str] = []      # e.g. ["low_visibility", "multiple_people"]


# --------------------------------------------------------------------------
# Safety
# --------------------------------------------------------------------------

SafetyInputKind = Literal[
    "critical_utterance",       # explicit emergency / red-flag concept affirmed
    "possible_emergency",       # ambiguous emergency language -> check in
    "pain_exclamation",
    "pain_report",
    "fall_reported",
    "fall_detected",
    "lying_inactive",
    "user_ok",
    "user_confirm",
    "user_cancel",
    "user_request_help_now",    # UI big red button / explicit "call now"
    "resolve",                  # UI: situation resolved
    "timer_expired",
    "symptom_report",
]


class SafetyInput(BaseModel):
    """The ONLY way to drive the safety machine. Produced from NLU, perception,
    UI buttons and timers. There is deliberately no kind for LLM output."""

    kind: SafetyInputKind
    source: Literal["nlu", "perception", "ui", "timer"]
    ts: float = Field(default_factory=now)
    confidence: float = 1.0
    severity: Severity = Severity.INFO
    detail: dict[str, Any] = {}
    ref_id: Optional[str] = None           # utterance / observation id that caused it


class SafetyAction(BaseModel):
    kind: Literal[
        "say",                 # emit a deterministic directive (phrase key)
        "start_timer",
        "cancel_timer",
        "notify_contacts",
        "sound_alarm",
        "stop_alarm",
        "log_event",
    ]
    phrase_key: Optional[str] = None
    params: dict[str, Any] = {}


class SafetyContext(BaseModel):
    state: SafetyState = SafetyState.MONITORING
    since: float = Field(default_factory=now)
    reason: Optional[str] = None
    severity: Severity = Severity.INFO
    critical: bool = False
    timer_name: Optional[str] = None
    timer_deadline: Optional[float] = None
    awaiting: Optional[Literal["ok_check", "pain_score", "confirm_escalation"]] = None
    trigger_ref: Optional[str] = None
    escalation_id: Optional[str] = None


class SafetyTransition(BaseModel):
    before: SafetyState
    after: SafetyState
    input: SafetyInput
    actions: list[SafetyAction]
    rule: str                       # id of the transition rule that fired
    context: SafetyContext          # the resulting safety context
    ts: float = Field(default_factory=now)


# --------------------------------------------------------------------------
# Knowledge + conversation
# --------------------------------------------------------------------------

class KnowledgeHit(BaseModel):
    entry_id: str
    title: str
    text: str                  # in the response language
    score: float
    concepts: list[str]
    source: str
    review_status: str


class Statement(BaseModel):
    """One sentence Baymax says, with its provenance."""

    text: str
    provenance: Provenance
    ref: Optional[str] = None          # kb:<id> | event:<id> | obs:<id> | phrase:<key> | llm:<call id>
    model_id: Optional[str] = None
    confidence: Optional[float] = None
    health_related: bool = False


class BaymaxResponse(BaseModel):
    id: str = Field(default_factory=lambda: new_id("rsp"))
    lang: str
    statements: list[Statement]
    safety: SafetyContext
    ts: float = Field(default_factory=now)
    ui_hints: dict[str, Any] = {}

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.statements)


class HealthEvent(BaseModel):
    id: str = Field(default_factory=lambda: new_id("evt"))
    kind: EventKind
    ts: float = Field(default_factory=now)
    severity: Severity = Severity.INFO
    concepts: list[str] = []
    body_regions: list[str] = []
    pain_score: Optional[int] = None
    measurement: Optional[Measurement] = None
    medication: Optional[str] = None
    summary: str = ""                     # language-neutral structured summary (never shown as a fact)
    provenance: Provenance
    source_ref: Optional[str] = None      # utterance id / observation id
    model_id: Optional[str] = None
    confidence: Optional[float] = None
    lang: Optional[str] = None


class TurnTrace(BaseModel):
    """Full trace of one pipeline run: what each stage saw and produced."""

    id: str = Field(default_factory=lambda: new_id("trace"))
    input: dict[str, Any]
    nlu: Optional[NLUResult] = None
    safety_inputs: list[SafetyInput] = []
    transitions: list[SafetyTransition] = []
    knowledge: list[KnowledgeHit] = []
    events: list[HealthEvent] = []
    llm: Optional[dict[str, Any]] = None
    response: Optional[BaymaxResponse] = None
