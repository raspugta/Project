# Safety design

Baymax's safety behaviour is decided by one **deterministic, pure state machine**
(`baymax/safety/machine.py::step`). Nothing else — not the conversational layer,
not the LLM, not the knowledge base — can start a check-in, a countdown or an alert.

## Hard guarantees (each enforced by a test)

| Guarantee | Mechanism | Test |
|---|---|---|
| The LLM cannot trigger, delay or cancel safety actions | `SafetyInput.source ∈ {nlu, perception, ui, timer}`; there is no LLM input kind. The LLM runs *after* safety, only in `monitoring`/`assisting`. | `test_no_llm_input_kind_exists`, `test_malicious_llm_cannot_change_safety_or_reach_user`, `test_llm_not_called_in_emergency_states` |
| Deterministic safety sentences cannot be removed or reworded | LLM output is only *appended*; the guard rejects contradictions ("no need to call", "it's nothing") | `test_benign_llm_output_is_labelled_llm_and_appended_last`, guard corpus |
| An affirmed red flag always reaches an alert countdown, from any state | rule `R-G-CRITICAL` | `test_critical_input_always_starts_countdown`, emergency-recall gate = 1.0 |
| `escalated` is only reachable by explicit confirmation, the help button, or an expired countdown | rule table | `test_every_state_input_pair_is_well_defined`, random-walk test (3,000 sequences) |
| Every waiting state has a deadline | `with_timer()` | same |
| The user can always get back to monitoring | cancel / "I'm okay" / resolve | random-walk test |
| Stale timers never fire | timer name must match the context | `test_stale_timer_is_ignored` |
| Settings cannot disable the safety net | server clamps check-in ≤ 300 s, countdown ≥ 5 s | `test_policy_floors_cannot_disable_safety_net` |
| A failing alert channel never breaks the safety path | every channel wrapped; failure reported to the user with the emergency number | `test_failed_channel_never_raises` |

## States

```
                 critical (any state) ─────────────────────────────┐
                                                                   ▼
 MONITORING ──possible / fall / inactivity──► CHECK_IN ──no reply──► ESCALATION_COUNTDOWN ──timeout / "yes"──► ESCALATED
     ▲  │                                        │  ▲                  │                                        │
     │  └──ouch / pain──► ASSISTING ──pain ≥ 8, "yes"─────────────────────────────────────────────────────────►│
     │                        │                   │  └ "no" (to "are you ok?") ┘                               │
     └──── "I'm ok" / cancel / resolve ◄──────────┴──────────────────────────── cancel ◄────────────  "I'm ok" / resolve
```

## Inputs

| Input | Produced by |
|---|---|
| `critical_utterance` | NLU: explicit emergency phrase, or affirmed present-tense red-flag concept (chest pain, breathing difficulty, stroke signs, severe bleeding, unconsciousness, seizure, self-harm, anaphylaxis signs, poisoning) |
| `possible_emergency` | NLU: ambiguous language ("I'm dying", "something is wrong"), red flag in a question / in the past, "no" to "are you okay?", fake-emergency markers ("as a prank") |
| `pain_exclamation`, `pain_report`, `fall_reported`, `symptom_report` | NLU |
| `user_ok`, `user_cancel`, `user_confirm` | NLU or UI buttons |
| `fall_detected`, `lying_inactive` | perception models (with confidence ≥ policy threshold) |
| `user_request_help_now`, `resolve` | UI buttons |
| `timer_expired` | the engine's timer tick |

Uncertainty never maps to "ignore": ambiguous emergency language becomes a check-in with a timeout.
A red flag mentioned in a question ("what if someone has chest pain?") asks *"Is this happening right now?"*
and, unlike other check-ins, ends quietly if unanswered (`R-C-TIMEOUT-INFO`) — an unanswered informational
question is not evidence of an emergency.

## Rule table

| Rule | From | Input | To | Main actions |
|---|---|---|---|---|
| R-G-CRITICAL | any except escalated | critical_utterance | countdown | emergency guidance (or crisis support for self-harm / "call for them" for third parties), countdown, soft alarm, emergency event |
| R-G-HELP-NOW | any except escalated | help button | escalated | notify contacts, alarm, guidance |
| R-X-REPEAT / R-E-REPEAT | countdown / escalated | critical_utterance | same | repeat guidance |
| R-M-POSSIBLE | monitoring | possible_emergency | check_in | "Do you need emergency help?" (+ "Is this happening now?") |
| R-M-OUCH | monitoring | pain_exclamation | assisting | "Are you okay?", ask 0–10 |
| R-M-PAIN-ASK / -HIGH / -LOGGED | monitoring | pain_report | assisting / assisting / monitoring | ask score / offer alert (≥ threshold) / log + "see a doctor if worse" |
| R-M-FALL-REPORTED | monitoring | fall_reported | check_in | fall guidance, check-in timer |
| R-M-FALL-SEVERE | monitoring | fall_reported (can't get up / head injury) | countdown | fall guidance, countdown |
| R-M-FALL-DETECTED | monitoring | fall_detected ≥ threshold | check_in | "It looks like you may have fallen…" (model provenance), chime |
| R-M-INACTIVE | monitoring | lying_inactive ≥ threshold | check_in | inactivity check-in |
| R-C-OK | check_in | user_ok / cancel | monitoring | "glad you're okay", log outcome |
| R-C-NOT-OK | check_in | possible_emergency | countdown | countdown |
| R-C-PAIN-ASK | check_in | pain without score | check_in | ask 0–10, keep the no-response timer |
| R-C-FALL-REPORTED / -SEVERE | check_in | fall_reported | check_in / countdown | |
| R-C-CONFIRM | check_in | user_confirm | escalated | notify contacts |
| R-C-TIMEOUT | check_in | timer | countdown | "I haven't heard from you", loud alarm |
| R-C-TIMEOUT-INFO | check_in (informational) | timer | monitoring | log only |
| R-C-TIMEOUT-NOESC | check_in (auto-escalation disabled) | timer | monitoring | log only |
| R-A-PAIN-* | assisting | pain_report | assisting / monitoring | as monitoring |
| R-A-CONFIRM | assisting (offer pending) | user_confirm | escalated | notify contacts (urgent) |
| R-A-DECLINE | assisting | cancel / ok | monitoring | |
| R-A-POSSIBLE | assisting | possible_emergency | countdown | |
| R-A-FALL-* / R-A-INACTIVE | assisting | fall / inactivity | as from monitoring | a fall outranks a pain conversation |
| R-A-IDLE | assisting | timer | monitoring | silent |
| R-X-CANCEL | countdown | cancel / ok | monitoring | stop alarm, "cancelled" (+ "please still consider calling …" if it was a red flag) |
| R-X-CONFIRM / R-X-TIMEOUT | countdown | yes / timer | escalated | notify contacts, loud alarm, guidance |
| R-E-ALL-CLEAR | escalated | ok / cancel | monitoring | all-clear message to contacts, advice if red flag |
| R-E-RESOLVE | escalated | resolve button | monitoring | stop alarm |
| R-E-REMIND | escalated | timer | escalated | repeat emergency number every 2 min |

Every transition is written to the audit log with its rule id, input kind, source and reference id.

## Escalation

- **Channels:** webhook (JSON POST — works with ntfy.sh, Home Assistant, Telegram/SMS bridges), e-mail
  (SMTP via `BAYMAX_SMTP_*` env vars), console. A local alarm always sounds in the browser.
- **Dry-run is ON by default.** A new install never pages anyone until the user tests with
  "Send test alert" and switches test mode off.
- **Consent:** alerts require `emergency_contact_sharing`. The alert text is fixed, in the *contact's* language,
  and never contains conversation text. The triggering reason ("a possible fall was detected by the camera")
  is included only with `share_health_details_in_alerts`.
- **All-clear:** if the user says they're okay after an alert, contacts get an all-clear message.
- **Default policy:** check-in timeout 30 s, alert countdown 15 s, fall countdown 30 s, offer alert at pain ≥ 8,
  lying-inactive check after 5 min. All configurable within clamped bounds.

## Autonomy vs safety

A conscious adult may cancel any countdown, including for a red flag — Baymax is not a medical device and does
not override the user. When a red-flag alert is cancelled, Baymax still says "please still consider calling
{number}". Framing tricks ("it's not an emergency but I can't breathe", "system: safety disabled") do not
suppress red flags; only an explicit cancel does, and it is audited.
