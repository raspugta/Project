"""End-to-end engine behaviour with perception, timers, consent gates and provenance."""
import httpx

from baymax.contracts import AudioLevel, BodySample, Utterance
from baymax.eval.harness import make_engine
from baymax.eval.perception_synth import evaluate_fall_detector, generate, make
from baymax.ontology import ConsentScope, Modality, Provenance, SafetyState
from baymax.perception.pipeline import ActivityClassifier, ImpactDetector, PerceptionHub
from baymax.safety.escalation import Contact, Dispatcher, EscalationConfig

import random


def _fall_samples(seed=3):
    return make("fall_forward", random.Random(seed), "pose", 0).samples


def test_detected_fall_flows_to_check_in_then_real_alert():
    eng, clock = make_engine(None)
    sent = []
    eng.dispatcher = Dispatcher(transport=httpx.MockTransport(lambda r: sent.append(r) or httpx.Response(200)))
    eng.update_escalation(EscalationConfig(dry_run=False, contacts=[Contact(name="A", channel="webhook", address="https://x.test")]))
    traces = eng.ingest_body(_fall_samples())
    assert traces and eng.machine.ctx.state == SafetyState.CHECK_IN
    st = traces[0].response.statements
    assert st[0].provenance == Provenance.MODEL and st[0].ref.startswith("obs:") and st[0].model_id == "fall_kinematic_v1"
    assert eng.resolve_ref(st[0].ref)["confidence"] >= 0.6
    clock.advance(eng.policy.checkin_timeout_s + 1)
    assert eng.tick().response.safety.state == SafetyState.ESCALATION_COUNTDOWN
    clock.advance(eng.policy.countdown_s + 1)
    tr = eng.tick()
    assert tr.response.safety.state == SafetyState.ESCALATED
    assert len(sent) == 1
    eng.process_utterance(Utterance(text="I'm okay"))
    assert eng.machine.ctx.state == SafetyState.MONITORING
    assert len(sent) == 2   # all-clear message


def test_responding_to_check_in_prevents_escalation():
    eng, clock = make_engine(None)
    eng.ingest_body(_fall_samples())
    clock.advance(5)
    eng.process_utterance(Utterance(text="I'm fine, I just sat down fast"))
    clock.advance(200)
    assert eng.tick() is None
    assert eng.machine.ctx.state == SafetyState.MONITORING


def test_camera_samples_ignored_without_consent_or_when_paused():
    eng, _ = make_engine(None)
    eng.store.set_consent(ConsentScope.CAMERA, False)
    assert eng.ingest_body(_fall_samples()) == []
    eng.store.set_consent(ConsentScope.CAMERA, True)
    eng.process_utterance(Utterance(text="stop watching"))
    assert eng.privacy["camera_paused"]
    assert eng.ingest_body(_fall_samples()) == []


def test_wake_word_gate_on_continuous_speech():
    eng, _ = make_engine(None)
    assert eng.process_utterance(Utterance(text="what a nice day", modality=Modality.SPEECH)) is None
    assert eng.process_utterance(Utterance(text="help!", modality=Modality.SPEECH)).response.safety.state == SafetyState.ESCALATION_COUNTDOWN


def test_speech_rejected_without_microphone_consent():
    eng, _ = make_engine(None)
    eng.store.set_consent(ConsentScope.MICROPHONE, False)
    assert eng.process_utterance(Utterance(text="help", modality=Modality.SPEECH, wake_word=True)) is None


def test_every_statement_resolves():
    eng, _ = make_engine(None)
    for text in ["hello", "ouch", "8", "yes", "I'm okay", "my blood pressure is 120/80", "what was my blood pressure?",
                 "I'm allergic to nuts", "what am I allergic to?", "what should I do for a burn?"]:
        tr = eng.process_utterance(Utterance(text=text))
        for s in tr.response.statements:
            assert s.ref and eng.resolve_ref(s.ref) is not None, (text, s)
            if s.health_related:
                assert s.provenance != Provenance.SYSTEM


def test_impact_sound_alone_never_escalates():
    eng, _ = make_engine(None)
    levels = [AudioLevel(t=i * 0.1, rms_db=-60) for i in range(20)] + [AudioLevel(t=2.1, rms_db=-5)]
    eng.ingest_audio(levels)
    assert eng.machine.ctx.state == SafetyState.MONITORING
    assert any(isinstance(e, ImpactDetector) for e in [eng.perception.impact])


def test_lying_inactive_triggers_check_in():
    act = ActivityClassifier(inactivity_s=30)
    events = []
    for i in range(400):
        events += act.update(BodySample(t=i * 0.1, present=True, source="pose", center_y=0.8, torso_angle=85,
                                        motion=0.01, visibility=0.9, bbox_w=0.6, bbox_h=0.2))
    assert any(e.kind == "lying_inactive" for e in events)


def test_fall_detector_quality_on_held_out_scenarios():
    r = evaluate_fall_detector(generate(10, seed=4242))
    assert r["fall_recall"] >= 0.95
    assert r["false_positive_rate"] <= 0.05


def test_hub_reports_status():
    hub = PerceptionHub()
    hub.ingest_body(_fall_samples()[:5])
    assert hub.status()["source"] == "pose"
