# BAYMAX OS

A local, webcam + microphone personal health and care companion — built safety-first.

Baymax listens for "help", "ouch", pain, falls and symptoms in **six languages** (English, Hindi, Hinglish,
Spanish, French, German) and always answers in the language you spoke. It watches for falls and long
inactivity through the webcam **without video ever leaving the browser**, keeps a consented personal health
timeline, answers first-aid questions from a curated knowledge base, and escalates to your emergency contacts
through a **deterministic safety state machine that no language model can influence**.

Every sentence it says is traceable to one of: what **you** said · curated **Baymax knowledge** · an
identified **model inference** (with confidence) · or a guarded **LLM** conversational line. Click any chip in
the conversation to see the source; click **Why?** to see the whole pipeline trace.

> Baymax is not a medical device and does not diagnose. The knowledge base has not been clinically reviewed.
> In an emergency, call your local emergency number.

## Quick start

```bash
pip install -r requirements.txt            # fastapi, uvicorn, websockets, pydantic, httpx
pip install faster-whisper                 # optional: local speech recognition
scripts/fetch_models.sh                    # optional: vendor the pose model for fully offline camera analysis
python -m baymax.server.app                # → http://127.0.0.1:8765
```

Open the page in Chrome/Edge/Firefox, pick your language and consents, then turn on the camera and/or mic.
Type or talk. Try: *"ouch"* → *"7"*, *"I fell and can't get up"*, *"मुझे सीने में दर्द हो रहा है"*,
*"my blood pressure is 140 over 90"* → *"what was my blood pressure?"*, *"what should I do for a burn?"*,
*"stop watching"*. Set up contacts under **Emergency setup** (test mode is on until you switch it off).

```bash
python -m baymax.eval                      # full evaluation + release gates → reports/eval_latest.md
python -m pytest                           # ~700 tests (safety machine, guard, privacy, datasets, API)
python scripts/gen_docs.py                 # regenerate docs/ONTOLOGY.md and docs/MODEL_CARDS.md from code
```

## What's inside

| | |
|---|---|
| **Perception** (`web/sensors.js`, `baymax/perception/`) | MediaPipe pose landmarks (or a model-free motion-blob fallback) → body features → kinematic fall detector, activity/inactivity classifier, posture analyser, loud-impact corroboration. Each emits `PerceptionEvent(model_id, confidence, evidence, failure_flags)`. |
| **Speech** (`web/voice.js`, `baymax/audio/`) | Energy VAD → local Whisper, or browser speech with explicit consent; wake word "Baymax" (safety words bypass it); barge-in; echo suppression; per-language TTS voices; alarm tones. |
| **Language + NLU** (`baymax/language/`, `baymax/nlu/`) | Language ID; multilingual lexicons with negation, hypothetical, past-tense and third-person scoping; idiom suppression; pain scores (digits and number words), vitals with units and plausibility ranges, medications, memory facts, privacy commands, injection/exfiltration detection. |
| **Safety** (`baymax/safety/`) | Pure deterministic state machine (monitoring → check-in → assisting → countdown → escalated) with 30+ named rules; configurable timers; webhook / e-mail / console escalation with dry-run default and minimal-disclosure alerts in each contact's language. See [docs/SAFETY.md](docs/SAFETY.md). |
| **Knowledge** (`baymax/knowledge/`) | 22 curated first-aid entries × 6 languages with sources and review status; BM25 + ontology-concept retrieval. |
| **Conversation** (`baymax/conversation/`) | Provenance-tagged deterministic composition; optional LLM (Ollama local / Anthropic cloud) that can only *append* at most two lines, only outside emergencies, behind an output guard that rejects diagnoses, doses, ungrounded numbers, invented observations, reassurance, action claims, PII, role breaks and wrong-language output. |
| **Memory + privacy** (`baymax/store/`, `baymax/privacy/`) | Consent-gated SQLite timeline and long-term memory; revoking consent deletes data; export; delete-all; "forget that"; PII redaction; hash-chained tamper-evident audit log. See [docs/PRIVACY.md](docs/PRIVACY.md). |
| **UI** (`web/`) | Companion face with moods, live captions, provenance chips + source drawer, pipeline trace, full-screen countdown/emergency overlay synced to server timers, timeline with charts, consent, emergency setup, model cards, audit verification, eval report. Localised in all six languages. |
| **Evaluation** (`baymax/eval/`, `datasets/`) | 183 curated + 86 adversarial + 186 synthetic + held-out synthetic cases, 62 malicious/benign LLM outputs, 780 perception scenarios, response invariants (grounded numbers, resolvable refs, language consistency, no leaks) and 20 release gates. See [docs/EVALUATION.md](docs/EVALUATION.md). |

Design documents: [ARCHITECTURE](docs/ARCHITECTURE.md) · [ONTOLOGY](docs/ONTOLOGY.md) · [SAFETY](docs/SAFETY.md) ·
[MODEL CARDS](docs/MODEL_CARDS.md) · [EVALUATION](docs/EVALUATION.md) · [PRIVACY](docs/PRIVACY.md) ·
[DATASET SCHEMA](datasets/SCHEMA.md).

## Current evaluation

All 20 release gates pass (`reports/eval_latest.md`): emergency recall **127/127**, false escalations
**0/173**, adversarial **86/86**, output guard **52/52** attacks blocked (0/10 benign blocked), **0** attack
outputs reaching the user with a malicious LLM wired into the engine, **0** ungrounded numbers or leaks,
fall detection **99 %** recall / **0 %** FP on held-out synthetic scenarios and **91 %** / 0 % under stress.
First-run numbers (before fixes) and the limits of these numbers are in
[docs/EVALUATION.md](docs/EVALUATION.md) — notably, perception is evaluated on synthetic trajectories, not
real video, and the test sets were written by the same author as the system.

## Configuration

See `baymax.env.example` (server address, data directory, Whisper model, optional LLM backend, SMTP).
Safety timers, emergency number and contacts are set in the UI and stored locally.

## Project layout

```
baymax/            ontology.py · contracts.py · engine.py
  language/        detect.py (lang_id_v1) · lexicon.py · phrases.py (6-language phrase table) · text.py
  nlu/             model.py (nlu_lexicon_v1) · extract.py (measurements, pain scores)
  perception/      pipeline.py (fall_kinematic_v1, activity_v1, posture_v1, impact_audio_v1)
  audio/           stt.py (faster-whisper) · wake.py
  safety/          machine.py (state machine) · escalation.py
  knowledge/       kb.py · data/kb.json
  conversation/    compose.py · llm.py · guard.py
  store/ privacy/  db.py (events, memory, consent, audit chain) · redact.py
  models/          registry.py (model cards)
  eval/            harness.py · criteria.py · synth.py · perception_synth.py · schema.py · __main__.py
  server/          app.py (FastAPI + WebSocket)
web/               index.html · app.js · sensors.js · voice.js · i18n.js · styles.css
datasets/          curated/ · adversarial/ · synthetic/ · SCHEMA.md
docs/  tests/  scripts/  reports/
```
