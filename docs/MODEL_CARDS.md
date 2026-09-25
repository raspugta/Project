# Model cards

_Generated from `baymax/models/registry.py` by `scripts/gen_docs.py`._

Every component that infers something the user did not directly state is registered here. Only components marked **may trigger safety** can produce `SafetyInput`s; the LLM cannot.

## Language identifier — `lang_id_v1`

- **Kind:** rule-based (script detection + stop-word/lexicon scoring)
- **Runs:** local server (Python)
- **Input:** Utterance text (UTF-8), optional STT-reported language, session language prior
- **Output:** LanguageResult{lang in SUPPORTED_LANGS, confidence, per-language scores}
- **Confidence:** Share of the winning language's evidence among all evidence, shrunk toward 0 for texts under 3 tokens. Below 0.5 the session's last confident language is used instead.
- **Safety-critical:** no · **may trigger safety:** no

**Failure modes**

- Very short utterances ('ok', 'no', '7') carry no language evidence
- Code-switching (Hinglish mixing English words) can tilt toward English
- Proper nouns and medication names are language-neutral noise
- Languages outside the supported set are forced to the nearest supported one

**Mitigations**

- Session language prior for low-confidence turns
- Explicit language preference in settings overrides detection
- Dedicated Hinglish marker lexicon

## Intent + health-event extractor — `nlu_lexicon_v1`

- **Kind:** rule-based (multilingual lexicons, negation/hypothetical scopes, regex measurement parsers)
- **Runs:** local server (Python)
- **Input:** Utterance text + detected language + dialog context (what Baymax is awaiting)
- **Output:** NLUResult{intent, intent_confidence, concepts[+negated/hypothetical/subject], body regions, pain score, measurements, medication, memory fact, privacy action, injection flag}
- **Confidence:** Lexicon match strength: 0.95 for exact multi-word safety phrases, 0.8 for single strong keywords, 0.6 for weak/ambiguous keywords, 0.3 for fallback UNKNOWN.
- **Safety-critical:** yes · **may trigger safety:** yes

**Failure modes**

- Unseen paraphrases of emergencies (lexical coverage gap)
- Sarcasm / idioms ('this traffic is killing me')
- Negation outside the 3-token scope window
- STT errors that corrupt safety keywords
- Numbers from unrelated context read as measurements

**Mitigations**

- Safety-biased: ambiguous emergency language maps to CHECK_IN, never to 'ignore'
- Idiom allow-list tested in adversarial set
- Measurements require a type cue AND a plausible range
- Emergency recall is a hard release gate (must be 1.0 on curated+adversarial)

## Fall detector — `fall_kinematic_v1`

- **Kind:** kinematic (windowed descent velocity + post-event posture + stillness)
- **Runs:** local server (Python) on body features computed in the browser
- **Input:** Stream of BodySample (pose landmarks or motion blob features) at 5-15 Hz
- **Output:** PerceptionEvent{kind='fall', confidence, evidence{drop, peak_velocity, lying, stillness}, failure_flags}
- **Confidence:** Weighted sum: 0.45*descent score + 0.35*post-fall horizontal/low score + 0.2*post-fall stillness, times a visibility factor. Emitted when >= 0.6.
- **Safety-critical:** yes · **may trigger safety:** yes

**Failure modes**

- Person falls outside the camera view (no post-event evidence)
- Fast but intentional movements (plopping onto a sofa or floor)
- Occlusion by furniture hides the lying posture
- Multiple people: track switches between persons
- Pets or moving objects in motion-only mode
- Low light degrades landmark visibility

**Mitigations**

- A detected fall only causes CHECK_IN (ask); escalation needs no response within timeout
- Visibility factor + failure flags reported to UI
- Synthetic scenario suite incl. sit/lie/bend/leave-frame negatives with FP gate
- Loud-impact audio event can corroborate but never trigger alone

## Activity / inactivity classifier — `activity_v1`

- **Kind:** rule-based on BodySample window
- **Runs:** local server (Python)
- **Input:** BodySample stream
- **Output:** Activity label in {absent, upright_still, moving, lying}; PerceptionEvent 'lying_inactive' after configured minutes lying with low motion
- **Confidence:** Fraction of window samples agreeing with the label, times mean visibility.
- **Safety-critical:** yes · **may trigger safety:** yes

**Failure modes**

- Sleeping on a sofa in view looks like lying inactive
- Person partially out of frame

**Mitigations**

- Lying-inactive only triggers a check-in, with a long configurable threshold
- Can be disabled per user

## Posture analyser — `posture_v1`

- **Kind:** geometric (neck forward angle, shoulder tilt) on pose landmarks
- **Runs:** local server (Python)
- **Input:** BodySample stream with pose features
- **Output:** PerceptionEvent 'posture_poor' when neck-forward angle exceeds threshold for a sustained window
- **Confidence:** Fraction of window samples over threshold, times visibility.
- **Safety-critical:** no · **may trigger safety:** no

**Failure modes**

- Camera placed high/low skews angles
- Motion-only mode has no landmarks (analyser inactive)

**Mitigations**

- Only produces low-severity coaching tips; requires posture consent

## Loud impact detector — `impact_audio_v1`

- **Kind:** signal threshold (dB jump over rolling baseline)
- **Runs:** local server (Python) on RMS levels computed in the browser
- **Input:** AudioLevel stream (dBFS every ~100 ms). No raw audio.
- **Output:** PerceptionEvent 'loud_impact'
- **Confidence:** min(1, (jump_db - 20)/20)
- **Safety-critical:** no · **may trigger safety:** no

**Failure modes**

- Door slams, dropped objects, TV

**Mitigations**

- Never triggers safety alone; only corroborates a fall event

## Wake-word matcher — `wake_word_v1`

- **Kind:** fuzzy string match on transcripts
- **Runs:** local server (Python)
- **Input:** Transcript text
- **Output:** (wake: bool, remainder text, confidence)
- **Confidence:** 1 - normalised edit distance of best matching token window to a wake variant
- **Safety-critical:** no · **may trigger safety:** no

**Failure modes**

- STT mis-hears the name
- Name said in unrelated context

**Mitigations**

- Safety phrases bypass the wake word entirely

## Browser speech recognition (Web Speech API) — `browser_webspeech`

- **Kind:** neural (external, browser vendor)
- **Runs:** browser; may send audio to the vendor cloud
- **Input:** Microphone audio
- **Output:** Transcript + confidence (vendor-defined)
- **Confidence:** Vendor-reported; frequently 0 or missing, treated as unknown
- **Safety-critical:** no · **may trigger safety:** no

**Failure modes**

- Requires network in Chrome
- Language must be preset
- Accents / noisy rooms

**Mitigations**

- Requires explicit cloud_speech_recognition consent
- Keyboard input always available

## Local Whisper STT (faster-whisper) — `faster_whisper`

- **Kind:** neural (external, runs locally)
- **Runs:** local server, CPU/GPU
- **Input:** 16 kHz mono PCM WAV segment (voice-activity segmented in the browser)
- **Output:** Transcript, detected language, confidence
- **Confidence:** exp(mean avg_logprob of segments), clipped to 0..1
- **Safety-critical:** no · **may trigger safety:** no

**Failure modes**

- Hallucinated text on silence/noise
- Slow on CPU for large models

**Mitigations**

- Browser-side VAD sends only speech segments
- no_speech_prob filter
- Transcripts shown to user

## Conversational language model (optional) — `llm`

- **Kind:** llm (Anthropic API or local Ollama; disabled by default)
- **Runs:** local (Ollama) or cloud (requires cloud_llm consent)
- **Input:** Redacted context: safety directive (read-only), retrieved knowledge with ids, user-provided facts with ids, the user's text, target language
- **Output:** JSON list of statements, each citing kb:/event:/obs: ids or 'none'
- **Confidence:** Not used. LLM text is never treated as a fact; uncited sentences are labelled llm_conversational and must pass the output guard.
- **Safety-critical:** no · **may trigger safety:** no

**Failure modes**

- Hallucinated medical facts, diagnoses, doses or measurements
- Claims of observations it never had
- Contradicting / softening safety instructions
- Following injected instructions
- Answering in the wrong language

**Mitigations**

- Cannot produce SafetyInput (no code path)
- Deterministic safety statements are always emitted and cannot be removed
- Output guard rejects the whole LLM reply on any violation and falls back to deterministic composition
- Adversarial LLM corpus with block-rate release gate

## Knowledge retrieval — `retrieval_bm25_v1`

- **Kind:** BM25 over language-specific entry text + concept-id boosting
- **Runs:** local server (Python)
- **Input:** Query text, language, affirmed concept ids
- **Output:** Ranked KnowledgeHit list
- **Confidence:** BM25 score + 3.0 per matching concept; entries below 1.0 are dropped
- **Safety-critical:** no · **may trigger safety:** no

**Failure modes**

- Question about a topic outside the knowledge base

**Mitigations**

- Out-of-KB questions get an explicit 'I don't have verified information' reply
