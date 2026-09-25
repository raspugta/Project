# Privacy, consent and audit

## Data flow

| Data | Where it goes | Stored? |
|---|---|---|
| Camera frames | stay in the browser tab | never |
| Body-position numbers (hip height, torso angle, bbox, motion) | local server over WebSocket | no (only derived events) |
| Microphone audio | browser; speech segments → local server for Whisper STT | never |
| Microphone RMS level | local server (impact detector) | no |
| Browser speech recognition (optional) | the browser vendor (e.g. Google for Chrome) | vendor's policy — requires `cloud_speech_recognition` consent |
| Transcripts | local SQLite | only with `store_transcripts` |
| Health events (pain, symptoms, measurements, falls, medications) | local SQLite | only with `store_health_events` |
| Remembered facts (allergies, notes) | local SQLite | only with `long_term_memory` |
| Emergency / escalation records | local SQLite | always, structured only, no conversation text |
| Alerts | your contacts' channels | only with `emergency_contact_sharing`; health reason only with `share_health_details_in_alerts` |
| LLM context (optional) | local Ollama, or Anthropic API | cloud only with `cloud_llm`; PII-redacted, no contact data |

The server binds to `127.0.0.1` by default. Data lives in `./data/baymax.db` (override with `BAYMAX_DATA`).

## Consent

All scopes default to **off**; onboarding asks. Revoking a storage consent **deletes** what it covered
(and records that in the audit log). Consents are checked inside the store and the engine, not only in the UI.

## User controls

- Voice: "stop watching", "stop listening", "privacy mode", "resume", "forget that" (deletes records made
  from the previous utterance). "Delete all my data" by voice only points to Settings — destructive actions
  need a typed confirmation.
- UI: pause-all-sensors button, per-scope toggles, remembered-facts list with *Forget*, per-event delete on
  the timeline, JSON export of everything, delete-all (type `DELETE`).
- Continuous listening only processes speech that starts with the wake word "Baymax" — except safety speech
  (help, pain sounds, falls, answers to a check-in), which always gets through.

## Audit log

Append-only table where each entry's hash is `sha256(previous_hash + canonical_json(entry))`.
`GET /api/audit/verify` (UI: System → Verify integrity) recomputes the chain and reports the first broken
entry. Entries hold ids, rule ids, decisions and model/guard outcomes — **never transcript text**.
Logged: consent changes, safety transitions (rule, input, source, confidence), escalations and channel
results, perception events that fed safety, every turn's intent/provenance summary, LLM calls (accepted or
rejected with violation codes), privacy commands, exports and deletions.

## Not (yet) implemented

- Encryption at rest of the SQLite file (use OS disk encryption).
- Multi-user profiles / authentication on the local server (it assumes a single trusted local user).
- Automatic retention periods (currently manual deletion).
