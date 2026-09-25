"""Model registry.

Every component that infers something the user did not directly state is a
"model" and must be registered here with a card describing its input,
output, how its confidence score is defined, and its known failure modes.
Components refer to themselves by `model_id`; unregistered ids fail tests.
"""
from __future__ import annotations

from pydantic import BaseModel


class ModelCard(BaseModel):
    model_id: str
    name: str
    kind: str                     # rule-based | kinematic | statistical | neural (external) | llm
    runs: str                     # where it executes
    input: str
    output: str
    confidence: str               # definition of the confidence score
    failure_modes: list[str]
    mitigations: list[str]
    safety_critical: bool
    may_trigger_safety: bool      # can its output become a SafetyInput?


REGISTRY: dict[str, ModelCard] = {}


def register(card: ModelCard) -> ModelCard:
    REGISTRY[card.model_id] = card
    return card


def get(model_id: str) -> ModelCard:
    base = model_id.split(":")[0]
    return REGISTRY[base]


register(ModelCard(
    model_id="lang_id_v1",
    name="Language identifier",
    kind="rule-based (script detection + stop-word/lexicon scoring)",
    runs="local server (Python)",
    input="Utterance text (UTF-8), optional STT-reported language, session language prior",
    output="LanguageResult{lang in SUPPORTED_LANGS, confidence, per-language scores}",
    confidence="Share of the winning language's evidence among all evidence, shrunk toward 0 for texts under 3 tokens. Below 0.5 the session's last confident language is used instead.",
    failure_modes=[
        "Very short utterances ('ok', 'no', '7') carry no language evidence",
        "Code-switching (Hinglish mixing English words) can tilt toward English",
        "Proper nouns and medication names are language-neutral noise",
        "Languages outside the supported set are forced to the nearest supported one",
    ],
    mitigations=[
        "Session language prior for low-confidence turns",
        "Explicit language preference in settings overrides detection",
        "Dedicated Hinglish marker lexicon",
    ],
    safety_critical=False,
    may_trigger_safety=False,
))

register(ModelCard(
    model_id="nlu_lexicon_v1",
    name="Intent + health-event extractor",
    kind="rule-based (multilingual lexicons, negation/hypothetical scopes, regex measurement parsers)",
    runs="local server (Python)",
    input="Utterance text + detected language + dialog context (what Baymax is awaiting)",
    output="NLUResult{intent, intent_confidence, concepts[+negated/hypothetical/subject], body regions, pain score, measurements, medication, memory fact, privacy action, injection flag}",
    confidence="Lexicon match strength: 0.95 for exact multi-word safety phrases, 0.8 for single strong keywords, 0.6 for weak/ambiguous keywords, 0.3 for fallback UNKNOWN.",
    failure_modes=[
        "Unseen paraphrases of emergencies (lexical coverage gap)",
        "Sarcasm / idioms ('this traffic is killing me')",
        "Negation outside the 3-token scope window",
        "STT errors that corrupt safety keywords",
        "Numbers from unrelated context read as measurements",
    ],
    mitigations=[
        "Safety-biased: ambiguous emergency language maps to CHECK_IN, never to 'ignore'",
        "Idiom allow-list tested in adversarial set",
        "Measurements require a type cue AND a plausible range",
        "Emergency recall is a hard release gate (must be 1.0 on curated+adversarial)",
    ],
    safety_critical=True,
    may_trigger_safety=True,
))

register(ModelCard(
    model_id="fall_kinematic_v1",
    name="Fall detector",
    kind="kinematic (windowed descent velocity + post-event posture + stillness)",
    runs="local server (Python) on body features computed in the browser",
    input="Stream of BodySample (pose landmarks or motion blob features) at 5-15 Hz",
    output="PerceptionEvent{kind='fall', confidence, evidence{drop, peak_velocity, lying, stillness}, failure_flags}",
    confidence="Weighted sum: 0.45*descent score + 0.35*post-fall horizontal/low score + 0.2*post-fall stillness, times a visibility factor. Emitted when >= 0.6.",
    failure_modes=[
        "Person falls outside the camera view (no post-event evidence)",
        "Fast but intentional movements (plopping onto a sofa or floor)",
        "Occlusion by furniture hides the lying posture",
        "Multiple people: track switches between persons",
        "Pets or moving objects in motion-only mode",
        "Low light degrades landmark visibility",
    ],
    mitigations=[
        "A detected fall only causes CHECK_IN (ask); escalation needs no response within timeout",
        "Visibility factor + failure flags reported to UI",
        "Synthetic scenario suite incl. sit/lie/bend/leave-frame negatives with FP gate",
        "Loud-impact audio event can corroborate but never trigger alone",
    ],
    safety_critical=True,
    may_trigger_safety=True,
))

register(ModelCard(
    model_id="activity_v1",
    name="Activity / inactivity classifier",
    kind="rule-based on BodySample window",
    runs="local server (Python)",
    input="BodySample stream",
    output="Activity label in {absent, upright_still, moving, lying}; PerceptionEvent 'lying_inactive' after configured minutes lying with low motion",
    confidence="Fraction of window samples agreeing with the label, times mean visibility.",
    failure_modes=[
        "Sleeping on a sofa in view looks like lying inactive",
        "Person partially out of frame",
    ],
    mitigations=[
        "Lying-inactive only triggers a check-in, with a long configurable threshold",
        "Can be disabled per user",
    ],
    safety_critical=True,
    may_trigger_safety=True,
))

register(ModelCard(
    model_id="posture_v1",
    name="Posture analyser",
    kind="geometric (neck forward angle, shoulder tilt) on pose landmarks",
    runs="local server (Python)",
    input="BodySample stream with pose features",
    output="PerceptionEvent 'posture_poor' when neck-forward angle exceeds threshold for a sustained window",
    confidence="Fraction of window samples over threshold, times visibility.",
    failure_modes=[
        "Camera placed high/low skews angles",
        "Motion-only mode has no landmarks (analyser inactive)",
    ],
    mitigations=["Only produces low-severity coaching tips; requires posture consent"],
    safety_critical=False,
    may_trigger_safety=False,
))

register(ModelCard(
    model_id="impact_audio_v1",
    name="Loud impact detector",
    kind="signal threshold (dB jump over rolling baseline)",
    runs="local server (Python) on RMS levels computed in the browser",
    input="AudioLevel stream (dBFS every ~100 ms). No raw audio.",
    output="PerceptionEvent 'loud_impact'",
    confidence="min(1, (jump_db - 20)/20)",
    failure_modes=["Door slams, dropped objects, TV"],
    mitigations=["Never triggers safety alone; only corroborates a fall event"],
    safety_critical=False,
    may_trigger_safety=False,
))

register(ModelCard(
    model_id="wake_word_v1",
    name="Wake-word matcher",
    kind="fuzzy string match on transcripts",
    runs="local server (Python)",
    input="Transcript text",
    output="(wake: bool, remainder text, confidence)",
    confidence="1 - normalised edit distance of best matching token window to a wake variant",
    failure_modes=["STT mis-hears the name", "Name said in unrelated context"],
    mitigations=["Safety phrases bypass the wake word entirely"],
    safety_critical=False,
    may_trigger_safety=False,
))

register(ModelCard(
    model_id="browser_webspeech",
    name="Browser speech recognition (Web Speech API)",
    kind="neural (external, browser vendor)",
    runs="browser; may send audio to the vendor cloud",
    input="Microphone audio",
    output="Transcript + confidence (vendor-defined)",
    confidence="Vendor-reported; frequently 0 or missing, treated as unknown",
    failure_modes=["Requires network in Chrome", "Language must be preset", "Accents / noisy rooms"],
    mitigations=["Requires explicit cloud_speech_recognition consent", "Keyboard input always available"],
    safety_critical=False,
    may_trigger_safety=False,
))

register(ModelCard(
    model_id="faster_whisper",
    name="Local Whisper STT (faster-whisper)",
    kind="neural (external, runs locally)",
    runs="local server, CPU/GPU",
    input="16 kHz mono PCM WAV segment (voice-activity segmented in the browser)",
    output="Transcript, detected language, confidence",
    confidence="exp(mean avg_logprob of segments), clipped to 0..1",
    failure_modes=["Hallucinated text on silence/noise", "Slow on CPU for large models"],
    mitigations=["Browser-side VAD sends only speech segments", "no_speech_prob filter", "Transcripts shown to user"],
    safety_critical=False,
    may_trigger_safety=False,
))

register(ModelCard(
    model_id="llm",
    name="Conversational language model (optional)",
    kind="llm (Anthropic API or local Ollama; disabled by default)",
    runs="local (Ollama) or cloud (requires cloud_llm consent)",
    input="Redacted context: safety directive (read-only), retrieved knowledge with ids, user-provided facts with ids, the user's text, target language",
    output="JSON list of statements, each citing kb:/event:/obs: ids or 'none'",
    confidence="Not used. LLM text is never treated as a fact; uncited sentences are labelled llm_conversational and must pass the output guard.",
    failure_modes=[
        "Hallucinated medical facts, diagnoses, doses or measurements",
        "Claims of observations it never had",
        "Contradicting / softening safety instructions",
        "Following injected instructions",
        "Answering in the wrong language",
    ],
    mitigations=[
        "Cannot produce SafetyInput (no code path)",
        "Deterministic safety statements are always emitted and cannot be removed",
        "Output guard rejects the whole LLM reply on any violation and falls back to deterministic composition",
        "Adversarial LLM corpus with block-rate release gate",
    ],
    safety_critical=False,
    may_trigger_safety=False,
))

register(ModelCard(
    model_id="retrieval_bm25_v1",
    name="Knowledge retrieval",
    kind="BM25 over language-specific entry text + concept-id boosting",
    runs="local server (Python)",
    input="Query text, language, affirmed concept ids",
    output="Ranked KnowledgeHit list",
    confidence="BM25 score + 3.0 per matching concept; entries below 1.0 are dropped",
    failure_modes=["Question about a topic outside the knowledge base"],
    mitigations=["Out-of-KB questions get an explicit 'I don't have verified information' reply"],
    safety_critical=False,
    may_trigger_safety=False,
))
