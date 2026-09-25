"""Consent enforcement, audit-log integrity, data export/deletion and escalation delivery."""
import json
import sqlite3

import httpx

from baymax.contracts import HealthEvent, Utterance
from baymax.engine import Engine
from baymax.eval.harness import make_engine
from baymax.ontology import ConsentScope, EventKind, Provenance
from baymax.privacy.redact import redact
from baymax.safety.escalation import Contact, Dispatcher, EscalationConfig, render_alert
from baymax.store.db import Store


def test_nothing_is_stored_without_consent():
    store = Store()
    eng = Engine(store)
    eng.process_utterance(Utterance(text="my knee hurts, 6 out of 10"))
    eng.process_utterance(Utterance(text="I'm allergic to penicillin"))
    assert store.events() == []
    assert store.memories() == []
    assert store.utterances() == []


def test_emergency_records_are_kept_without_transcripts():
    store = Store()
    eng = Engine(store)
    eng.process_utterance(Utterance(text="I have chest pain"))
    kinds = {e.kind for e in store.events()}
    assert EventKind.EMERGENCY in kinds
    assert store.utterances() == []


def test_revoking_consent_deletes_covered_data():
    store = Store()
    store.set_consent(ConsentScope.STORE_HEALTH_EVENTS, True)
    store.add_event(HealthEvent(kind=EventKind.PAIN, provenance=Provenance.USER, pain_score=3))
    assert store.events()
    store.set_consent(ConsentScope.STORE_HEALTH_EVENTS, False)
    assert store.events() == []
    assert any(a["action"] == "data_deleted_on_revocation" for a in store.audit_log())


def test_audit_chain_detects_tampering(tmp_path):
    path = tmp_path / "b.db"
    store = Store(path)
    for i in range(5):
        store.audit("test", "thing", i=i)
    assert store.verify_audit()["ok"]
    con = sqlite3.connect(path)
    con.execute("UPDATE audit SET detail = ? WHERE seq = 3", (json.dumps({"i": 999}),))
    con.commit()
    con.close()
    v = Store(path).verify_audit()
    assert not v["ok"] and v["broken_at"] == 3


def test_audit_never_contains_transcript_text():
    eng, _ = make_engine(None)
    eng.process_utterance(Utterance(text="my secret diary says my knee hurts"))
    blob = json.dumps(eng.store.audit_log(1000))
    assert "secret diary" not in blob


def test_export_and_delete_all():
    eng, _ = make_engine(None)
    eng.process_utterance(Utterance(text="my blood pressure is 130/80"))
    exported = eng.store.export_all()
    assert exported["events"]
    counts = eng.store.delete_all()
    assert counts["events"] >= 1
    assert eng.store.events() == []
    assert eng.store.audit_log(1)[0]["action"] == "all_data_deleted"


def test_forget_last_utterance():
    eng, _ = make_engine(None)
    eng.process_utterance(Utterance(text="my blood pressure is 150/95"))
    assert eng.store.events()
    eng.process_utterance(Utterance(text="forget that"))
    assert eng.store.events() == []


def test_redaction():
    out = redact("mail me at a.b@c.org or call +91 98765 43210, see http://x.y")
    assert "[EMAIL]" in out and "[PHONE]" in out and "[URL]" in out
    assert redact("bp 120/80 on 2026-09-25") == "bp 120/80 on 2026-09-25"


# ------------------------------------------------------------------ escalation
def _transport(log):
    def handler(req: httpx.Request) -> httpx.Response:
        log.append(json.loads(req.content))
        return httpx.Response(200)
    return httpx.MockTransport(handler)


def test_dispatch_requires_consent_and_defaults_to_dry_run():
    d = Dispatcher()
    cfg = EscalationConfig(contacts=[Contact(name="A", channel="webhook", address="https://x.test/hook")])
    assert cfg.dry_run is True
    assert d.dispatch(cfg, "emergency", "critical", contacts_consent=False, share_details=False).status == "no_consent"
    assert d.dispatch(cfg, "emergency", "critical", contacts_consent=True, share_details=False).status == "dry_run"
    assert d.dispatch(EscalationConfig(), "emergency", "x", True, False).status == "no_contacts"


def test_webhook_delivery_and_minimal_disclosure():
    log = []
    d = Dispatcher(transport=_transport(log))
    cfg = EscalationConfig(dry_run=False, user_display_name="Ravi",
                           contacts=[Contact(name="A", channel="webhook", address="https://x.test/hook", lang="hi")])
    r = d.dispatch(cfg, "emergency", "fall_detected+no_response", contacts_consent=True, share_details=False)
    assert r.status == "sent"
    text = log[0]["text"]
    assert "Ravi" in text and "बेमैक्स" in text           # contact's own language
    assert "गिर" not in text                               # no health details without consent
    d.dispatch(cfg, "emergency", "fall_detected+no_response", contacts_consent=True, share_details=True)
    assert "गिर" in log[1]["text"]


def test_failed_channel_never_raises():
    def boom(req):
        raise httpx.ConnectError("down")
    d = Dispatcher(transport=httpx.MockTransport(boom))
    cfg = EscalationConfig(dry_run=False, contacts=[Contact(name="A", channel="webhook", address="https://x.test")])
    r = d.dispatch(cfg, "emergency", "critical", True, False)
    assert r.status == "failed" and not r.results[0].ok


def test_alert_text_never_contains_conversation():
    txt = render_alert("emergency", "critical", "en", "Sam", share_details=True)
    assert "Sam" in txt and "emergency symptoms" in txt
