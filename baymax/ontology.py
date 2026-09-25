"""BAYMAX ontology.

This module is the single source of truth for every concept the system can
reason about. Perception, NLU, safety, knowledge, storage and evaluation all
speak in these identifiers. Nothing outside this file may invent a new
concept id; adding a concept means adding it here (and to the lexicons and
knowledge base, which are checked for completeness by tests).
"""
from __future__ import annotations

from enum import Enum


class Lang(str, Enum):
    """Languages Baymax can understand AND answer in.

    Baymax never answers in a language that is not in this list; every
    deterministic phrase and knowledge entry must exist in all of them.
    """

    EN = "en"
    HI = "hi"            # Hindi, Devanagari script
    HI_LATN = "hi-Latn"  # Hindi written in Latin script ("Hinglish")
    ES = "es"
    FR = "fr"
    DE = "de"


SUPPORTED_LANGS: tuple[str, ...] = tuple(l.value for l in Lang)


class Provenance(str, Enum):
    """Where a statement came from. Every emitted statement carries one."""

    USER = "user_provided"              # user said/typed/entered it
    KNOWLEDGE = "baymax_knowledge"      # curated Baymax knowledge base / protocol
    MODEL = "model_inference"           # output of an identified model (with confidence)
    LLM = "llm_conversational"          # free text produced by an LLM (never a health fact)
    SYSTEM = "system"                   # non-health UI/dialog phrasing (greetings, prompts)


class Modality(str, Enum):
    SPEECH = "speech"
    TEXT = "text"
    VISION = "vision"
    AUDIO_EVENT = "audio_event"
    UI = "ui"
    TIMER = "timer"


class Intent(str, Enum):
    EMERGENCY_HELP = "emergency_help"        # explicit call for help / emergency
    PAIN_EXCLAMATION = "pain_exclamation"    # "ouch", "aah", "aïe"
    PAIN_REPORT = "pain_report"              # "my knee hurts, about 6"
    FALL_REPORT = "fall_report"              # "I fell"
    SYMPTOM_REPORT = "symptom_report"        # "I feel dizzy"
    VITAL_REPORT = "vital_report"            # "BP is 130/85"
    MEDICATION_LOG = "medication_log"        # "I took my inhaler"
    HEALTH_QUESTION = "health_question"      # "what do I do for a burn?"
    RECALL_QUERY = "recall_query"            # "what was my blood pressure yesterday?"
    USER_OK = "user_ok"                      # "I'm fine"
    CONFIRM = "confirm"                      # "yes"
    CANCEL = "cancel"                        # "cancel", "don't call"
    PRIVACY_COMMAND = "privacy_command"      # "stop watching", "forget that"
    MEMORY_STATEMENT = "memory_statement"    # "I'm allergic to penicillin"
    GREETING = "greeting"
    GOODBYE = "goodbye"
    UNKNOWN = "unknown"


class Concept(str, Enum):
    """Clinical-ish concepts Baymax recognises. NOT diagnoses: these are the
    user's reported experiences or observable events."""

    PAIN = "pain"
    HEADACHE = "headache"
    CHEST_PAIN = "chest_pain"
    BREATHING_DIFFICULTY = "breathing_difficulty"
    STROKE_SIGNS = "stroke_signs"
    SEVERE_BLEEDING = "severe_bleeding"
    BLEEDING = "bleeding"
    UNCONSCIOUS = "unconscious"
    SEIZURE = "seizure"
    SELF_HARM = "self_harm"
    ANAPHYLAXIS_SIGNS = "anaphylaxis_signs"
    POISONING = "poisoning"
    HEAD_INJURY = "head_injury"
    FALL = "fall"
    DIZZINESS = "dizziness"
    NAUSEA = "nausea"
    VOMITING = "vomiting"
    FEVER = "fever"
    COUGH = "cough"
    SORE_THROAT = "sore_throat"
    FATIGUE = "fatigue"
    RASH = "rash"
    BURN = "burn"
    CUT = "cut"
    SPRAIN = "sprain"
    SWELLING = "swelling"
    NUMBNESS = "numbness"
    CONFUSION = "confusion"
    ANXIETY = "anxiety"
    INSOMNIA = "insomnia"
    DIARRHEA = "diarrhea"
    POSTURE = "posture"


# Concepts that on their own (when affirmed, first-person or third-person,
# present tense) always drive the safety machine toward escalation.
RED_FLAG_CONCEPTS: frozenset[Concept] = frozenset({
    Concept.CHEST_PAIN,
    Concept.BREATHING_DIFFICULTY,
    Concept.STROKE_SIGNS,
    Concept.SEVERE_BLEEDING,
    Concept.UNCONSCIOUS,
    Concept.SEIZURE,
    Concept.SELF_HARM,
    Concept.ANAPHYLAXIS_SIGNS,
    Concept.POISONING,
})


class BodyRegion(str, Enum):
    HEAD = "head"
    FACE = "face"
    EYE = "eye"
    EAR = "ear"
    MOUTH = "mouth"
    THROAT = "throat"
    NECK = "neck"
    SHOULDER = "shoulder"
    CHEST = "chest"
    ABDOMEN = "abdomen"
    BACK = "back"
    ARM = "arm"
    ELBOW = "elbow"
    WRIST = "wrist"
    HAND = "hand"
    HIP = "hip"
    LEG = "leg"
    KNEE = "knee"
    ANKLE = "ankle"
    FOOT = "foot"


class MeasurementType(str, Enum):
    BLOOD_PRESSURE = "blood_pressure"   # value=systolic, value2=diastolic, mmHg
    HEART_RATE = "heart_rate"           # bpm
    TEMPERATURE = "temperature"         # canonical °C
    SPO2 = "spo2"                       # %
    GLUCOSE = "glucose"                 # canonical mg/dL
    WEIGHT = "weight"                   # canonical kg
    PAIN_SCORE = "pain_score"           # 0..10


# Plausibility ranges (canonical units). Values outside are rejected as
# extraction errors rather than stored; they are never "corrected".
MEASUREMENT_PLAUSIBLE: dict[MeasurementType, tuple[float, float]] = {
    MeasurementType.BLOOD_PRESSURE: (50, 260),   # systolic
    MeasurementType.HEART_RATE: (25, 250),
    MeasurementType.TEMPERATURE: (30.0, 44.0),
    MeasurementType.SPO2: (50, 100),
    MeasurementType.GLUCOSE: (20, 700),
    MeasurementType.WEIGHT: (2, 400),
    MeasurementType.PAIN_SCORE: (0, 10),
}

MEASUREMENT_UNITS: dict[MeasurementType, str] = {
    MeasurementType.BLOOD_PRESSURE: "mmHg",
    MeasurementType.HEART_RATE: "bpm",
    MeasurementType.TEMPERATURE: "°C",
    MeasurementType.SPO2: "%",
    MeasurementType.GLUCOSE: "mg/dL",
    MeasurementType.WEIGHT: "kg",
    MeasurementType.PAIN_SCORE: "/10",
}


class EventKind(str, Enum):
    """Health-event taxonomy stored in the personal health-event database."""

    PAIN = "pain"
    SYMPTOM = "symptom"
    FALL_REPORTED = "fall_reported"
    FALL_DETECTED = "fall_detected"
    MEASUREMENT = "measurement"
    MEDICATION_TAKEN = "medication_taken"
    EMERGENCY = "emergency"
    POSTURE_ALERT = "posture_alert"
    INACTIVITY = "inactivity"
    CHECK_IN = "check_in"
    ESCALATION = "escalation"
    NOTE = "note"


class Severity(str, Enum):
    INFO = "info"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


SEVERITY_ORDER = {s: i for i, s in enumerate(Severity)}


class SafetyState(str, Enum):
    """States of the deterministic safety machine (see baymax/safety/machine.py)."""

    MONITORING = "monitoring"                  # normal; listening for triggers
    CHECK_IN = "check_in"                      # asked "are you okay?", awaiting reply w/ timeout
    ASSISTING = "assisting"                    # guided care flow (non-emergency)
    ESCALATION_COUNTDOWN = "escalation_countdown"  # will notify contacts unless cancelled
    ESCALATED = "escalated"                    # contacts notified / emergency guidance active


SAFETY_STATE_RANK = {
    SafetyState.MONITORING: 0,
    SafetyState.ASSISTING: 1,
    SafetyState.CHECK_IN: 2,
    SafetyState.ESCALATION_COUNTDOWN: 3,
    SafetyState.ESCALATED: 4,
}


class ConsentScope(str, Enum):
    CAMERA = "camera_processing"                 # analyse webcam locally
    MICROPHONE = "microphone_processing"         # listen / transcribe
    CLOUD_SPEECH = "cloud_speech_recognition"    # browser STT that may use a vendor cloud
    STORE_TRANSCRIPTS = "store_transcripts"      # keep conversation text
    STORE_HEALTH_EVENTS = "store_health_events"  # keep structured health events
    LONG_TERM_MEMORY = "long_term_memory"        # remember facts (allergies, meds, ...)
    EMERGENCY_CONTACTS = "emergency_contact_sharing"  # send alerts to contacts
    SHARE_HEALTH_IN_ALERTS = "share_health_details_in_alerts"
    CLOUD_LLM = "cloud_llm"                      # send redacted context to a hosted LLM
    POSTURE_COACHING = "posture_coaching"


CONSENT_DESCRIPTIONS: dict[ConsentScope, str] = {
    ConsentScope.CAMERA: "Analyse the webcam on this device to detect falls, inactivity and posture. Frames never leave the browser; only body-position numbers are sent to the local Baymax server.",
    ConsentScope.MICROPHONE: "Listen through the microphone so you can talk to Baymax and so safety words like 'help' or 'ouch' are recognised.",
    ConsentScope.CLOUD_SPEECH: "Allow the browser's built-in speech recognition. In some browsers (e.g. Chrome) this sends audio to the browser vendor's servers. Without it, only local speech recognition is used.",
    ConsentScope.STORE_TRANSCRIPTS: "Keep the text of conversations on this device.",
    ConsentScope.STORE_HEALTH_EVENTS: "Keep a timeline of health events (pain, symptoms, measurements, falls) on this device.",
    ConsentScope.LONG_TERM_MEMORY: "Remember facts you tell Baymax (allergies, medications, notes) across sessions.",
    ConsentScope.EMERGENCY_CONTACTS: "Send alerts to your configured emergency contacts when escalation is triggered.",
    ConsentScope.SHARE_HEALTH_IN_ALERTS: "Include the triggering health details (e.g. 'reported chest pain') in alerts to contacts. Otherwise alerts only say that Baymax needs someone to check on you.",
    ConsentScope.CLOUD_LLM: "Send a redacted conversation context to a hosted language model for more natural replies. Safety decisions never depend on it.",
    ConsentScope.POSTURE_COACHING: "Give occasional posture reminders based on camera analysis.",
}

# Consents that are ON by default for a new profile: none. Baymax asks.
DEFAULT_CONSENTS: dict[ConsentScope, bool] = {s: False for s in ConsentScope}
