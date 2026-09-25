# BAYMAX ontology

_Generated from `baymax/ontology.py`, the lexicons and the knowledge base by `scripts/gen_docs.py`. Do not edit by hand._

## Languages

Baymax understands and answers only in: `en`, `hi`, `hi-Latn`, `es`, `fr`, `de`. Every phrase, knowledge entry and lexicon exists in all of them (enforced by tests).

## Provenance

| Value | Meaning |
|---|---|
| `user_provided` | the user said/typed/entered it |
| `baymax_knowledge` | curated Baymax knowledge or safety protocol text |
| `model_inference` | output of an identified model, with model id and confidence |
| `llm_conversational` | free text from the optional LLM (never a health fact; guarded) |
| `system` | non-health dialogue text (greetings, prompts) or Baymax's own records |

## Intents

`emergency_help`, `pain_exclamation`, `pain_report`, `fall_report`, `symptom_report`, `vital_report`, `medication_log`, `health_question`, `recall_query`, `user_ok`, `confirm`, `cancel`, `privacy_command`, `memory_statement`, `greeting`, `goodbye`, `unknown`

## Concepts

Red-flag concepts (affirmed, present tense) always drive the safety machine to an alert countdown.

| Concept | Red flag | Knowledge entries | Example phrases (en) |
|---|---|---|---|
| `pain` |  | kb.pain_scale | pain, painful, hurts, hurt |
| `headache` |  | kb.headache | headache, migraine |
| `chest_pain` | **yes** | kb.chest_pain | chest pain, pain in my chest, chest hurts, chest is hurting |
| `breathing_difficulty` | **yes** | kb.breathing | can't breathe, cant breathe, cannot breathe, can not breathe |
| `stroke_signs` | **yes** | kb.stroke | face drooping, face is drooping, drooping face, face droop |
| `severe_bleeding` | **yes** | kb.severe_bleeding | bleeding a lot, bleeding heavily, heavy bleeding, won't stop bleeding |
| `bleeding` |  | kb.minor_cut | bleeding, i'm bleeding, bleeds |
| `unconscious` | **yes** | kb.emergency_general | unconscious, passed out, not responding, unresponsive |
| `seizure` | **yes** | kb.seizure | seizure, having a fit, convulsing, convulsions |
| `self_harm` | **yes** | kb.self_harm | kill myself, end my life, want to die, suicide |
| `anaphylaxis_signs` | **yes** | kb.anaphylaxis | throat is swelling, throat swelling, throat closing, throat is closing |
| `poisoning` | **yes** | kb.poisoning | overdose, overdosed, poisoned, swallowed bleach |
| `head_injury` |  | kb.head_injury | hit my head, banged my head, bumped my head, head injury |
| `fall` |  | kb.after_fall | i fell, fell down, i've fallen, i have fallen |
| `dizziness` |  | kb.dizziness | dizzy, dizziness, lightheaded, light headed |
| `nausea` |  | kb.stomach_upset | nausea, nauseous, feel sick, feeling sick |
| `vomiting` |  | kb.stomach_upset | vomit, vomiting, vomited, threw up |
| `fever` |  | kb.fever | fever, feverish, high temperature, temperature is high |
| `cough` |  |  | cough, coughing, coughed |
| `sore_throat` |  |  | sore throat, throat hurts, throat is sore |
| `fatigue` |  |  | tired, exhausted, fatigue, fatigued |
| `rash` |  |  | rash, hives, itchy, itching |
| `burn` |  | kb.burn | burn, burned, burnt, scalded |
| `cut` |  | kb.minor_cut | cut myself, cut my, a cut, deep cut |
| `sprain` |  | kb.sprain | sprain, sprained, twisted my, rolled my ankle |
| `swelling` |  | kb.sprain | swollen, swelling, swelled |
| `numbness` |  | kb.stroke | numb, numbness, tingling, pins and needles |
| `confusion` |  | kb.stroke | confused, disoriented, don't know where i am |
| `anxiety` |  | kb.anxiety | anxious, anxiety, panic, panic attack |
| `insomnia` |  |  | can't sleep, cant sleep, insomnia, couldn't sleep |
| `diarrhea` |  | kb.stomach_upset | diarrhea, diarrhoea, loose motion, loose motions |
| `posture` |  | kb.posture | posture, slouching, slouch |

Pain in a region derives a specific concept: pain + `chest` → `chest_pain`, pain + `head` → `headache`, pain + `throat` → `sore_throat`

## Body regions

`head`, `face`, `eye`, `ear`, `mouth`, `throat`, `neck`, `shoulder`, `chest`, `abdomen`, `back`, `arm`, `elbow`, `wrist`, `hand`, `hip`, `leg`, `knee`, `ankle`, `foot`

## Measurements

| Type | Canonical unit | Plausible range (outside = rejected, never corrected) |
|---|---|---|
| `blood_pressure` | mmHg | 50 – 260 |
| `heart_rate` | bpm | 25 – 250 |
| `temperature` | °C | 30.0 – 44.0 |
| `spo2` | % | 50 – 100 |
| `glucose` | mg/dL | 20 – 700 |
| `weight` | kg | 2 – 400 |
| `pain_score` | /10 | 0 – 10 |

## Health-event taxonomy

`pain`, `symptom`, `fall_reported`, `fall_detected`, `measurement`, `medication_taken`, `emergency`, `posture_alert`, `inactivity`, `check_in`, `escalation`, `note`

Severity: `info` < `low` < `moderate` < `high` < `critical`

## Safety states

`monitoring` → `check_in` → `assisting` → `escalation_countdown` → `escalated` (see docs/SAFETY.md)

## Consent scopes (all default OFF)

| Scope | What it allows |
|---|---|
| `camera_processing` | Analyse the webcam on this device to detect falls, inactivity and posture. Frames never leave the browser; only body-position numbers are sent to the local Baymax server. |
| `microphone_processing` | Listen through the microphone so you can talk to Baymax and so safety words like 'help' or 'ouch' are recognised. |
| `cloud_speech_recognition` | Allow the browser's built-in speech recognition. In some browsers (e.g. Chrome) this sends audio to the browser vendor's servers. Without it, only local speech recognition is used. |
| `store_transcripts` | Keep the text of conversations on this device. |
| `store_health_events` | Keep a timeline of health events (pain, symptoms, measurements, falls) on this device. |
| `long_term_memory` | Remember facts you tell Baymax (allergies, medications, notes) across sessions. |
| `emergency_contact_sharing` | Send alerts to your configured emergency contacts when escalation is triggered. |
| `share_health_details_in_alerts` | Include the triggering health details (e.g. 'reported chest pain') in alerts to contacts. Otherwise alerts only say that Baymax needs someone to check on you. |
| `cloud_llm` | Send a redacted conversation context to a hosted language model for more natural replies. Safety decisions never depend on it. |
| `posture_coaching` | Give occasional posture reminders based on camera analysis. |

## Deterministic phrases

56 phrase keys × 6 languages. Health-guidance phrases (provenance `baymax_knowledge`): `call_emergency`, `cancelled_but_advise`, `crisis_support`, `emergency_other`, `fall_reported_checkin`, `no_dosage`, `not_doctor`, `posture_tip`, `seek_care_if_worse`, `stay_calm`.
