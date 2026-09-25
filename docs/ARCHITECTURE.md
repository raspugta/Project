# Architecture

```
 Browser (web/)                                    Local server (baymax/, 127.0.0.1)
 ─────────────────────────────                     ──────────────────────────────────────────────────────────────
 webcam ─► pose landmarks (MediaPipe,              BodySample ─► perception/pipeline.py
           optional) or motion blob                              fall_kinematic_v1 · activity_v1 · posture_v1
           → BodySample numbers ───── WS ─────────►             │ PerceptionEvent(model_id, confidence, flags)
 mic ────► RMS levels ─────────────── WS ────────► AudioLevel ─► impact_audio_v1 (corroboration only)
        └► energy VAD → WAV ── POST /api/stt ───► audio/stt.py (faster-whisper, local)
        └► (or Web Speech API, with consent)                     │ transcript
 keyboard ─────────────── utterance ─ WS ────────► audio/wake.py (wake word) ─┐
                                                                            ▼
                                                   engine.py ── language/detect.py  (lang_id_v1)
                                                             ── nlu/model.py        (nlu_lexicon_v1: intent, concepts,
                                                                                      negation, hypothetical, past, subject,
                                                                                      pain score, measurements, medication,
                                                                                      memory facts, privacy commands, injection)
                                                             ── SafetyInput ─► safety/machine.py (pure, deterministic)
                                                                                 └► actions: say / timers / alarm /
                                                                                    notify ─► safety/escalation.py
                                                             ── store/db.py (consent-gated events, memory, audit chain)
                                                             ── knowledge/kb.py (curated KB + BM25/concept retrieval)
                                                             ── conversation/compose.py (provenance-tagged sentences)
                                                             ── conversation/llm.py + guard.py (optional, last, guarded)
 face / captions / TTS / alarm / overlay ◄── WS ── BaymaxResponse(statements[text, provenance, ref], safety, ui_hints)
```

## Separation of concerns

| Layer | Package | Decides | Never does |
|---|---|---|---|
| Perception | `perception/`, `web/sensors.js` | body features → fall / inactivity / posture events with confidence | speak, store frames, escalate |
| Speech | `audio/`, `web/voice.js` | transcripts, wake word | interpret meaning |
| Inference (NLU) | `nlu/`, `language/` | intent, concepts, extraction | decide safety actions |
| Safety | `safety/` | check-ins, countdowns, alerts | generate free text |
| Knowledge | `knowledge/` | which curated entry answers a concept/question | invent content |
| Conversation | `conversation/` | how to phrase the response | change safety state |
| Memory / DB | `store/`, `privacy/` | what is persisted (consent-gated), audit | store raw audio/video |

## Interfaces (`baymax/contracts.py`)

`Utterance`, `BodySample`, `AudioLevel` → `LanguageResult`, `NLUResult`, `PerceptionEvent`
→ `SafetyInput` → `SafetyTransition(before, after, rule, actions, context)` → `HealthEvent`, `KnowledgeHit`
→ `Statement(text, provenance, ref, model_id, confidence, health_related)` → `BaymaxResponse`.
Every run of the pipeline produces a `TurnTrace` with each stage's output (visible in the UI via **Why?**).

## Traceability

Each `Statement.ref` resolves (`Engine.resolve_ref`, `GET /api/ref`) to its source:

| ref | resolves to | provenance |
|---|---|---|
| `event:<id>` | the stored health event created from the user's words | `user_provided` |
| `memory:<id>` | a remembered fact the user stated | `user_provided` |
| `utt:<id>` | the utterance itself (when storage consent is off) | `user_provided` / `model_inference` |
| `kb:<id>` | curated knowledge entry, with source and review status | `baymax_knowledge` |
| `phrase:<key>` | deterministic phrase-table entry (protocol text) | `baymax_knowledge` / `system` |
| `obs:<id>` | perception event with model id, confidence, evidence, failure flags | `model_inference` |
| `llm:<call>` `|kb:…` | an LLM call and the ids it cited | `llm_conversational` |

The evaluation harness checks, for every response, that every ref resolves, that knowledge/phrase text is
exactly the curated text in the response language, and that every number is grounded
(user input, curated text, stored records, policy values).

## Language

Language is identified per turn (`lang_id_v1`); low-confidence turns (e.g. "7", "ok") reuse the session
language; an explicit preference overrides detection. Every deterministic sentence exists in all six
languages, so the response language always equals the user's language. The UI chrome switches too.
Code-switching (e.g. Hinglish) is handled by matching the detected language + English, and red-flag
concepts / emergency phrases are matched across **all** languages.

## Runtime

- Python 3.10+, FastAPI + uvicorn, SQLite (WAL). No GPU needed.
- Optional: `faster-whisper` for local STT; MediaPipe Pose (vendored with `scripts/fetch_models.sh`,
  otherwise loaded from CDN, otherwise motion-only mode); Ollama or Anthropic API for conversational polish.
- The engine is single-process and lock-protected; a 0.5 s ticker drives safety timers.
