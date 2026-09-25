"""HTTP + WebSocket API."""
from fastapi.testclient import TestClient

from baymax.engine import Engine
from baymax.server.app import create_app
from baymax.store.db import Store


def client():
    store = Store()
    app = create_app(store=store, engine=Engine(store))
    return TestClient(app), store


def test_state_and_consent_roundtrip():
    c, store = client()
    with c:
        s = c.get("/api/state").json()
        assert s["consents"]["camera_processing"] is False
        r = c.post("/api/consent", json={"scope": "camera_processing", "granted": True})
        assert r.status_code == 200 and r.json()["camera_processing"] is True
        assert c.post("/api/consent", json={"scope": "nope", "granted": True}).status_code == 400


def test_websocket_utterance_roundtrip():
    c, _ = client()
    with c, c.websocket_connect("/ws") as ws:
        ws.send_json({"type": "utterance", "text": "I have chest pain", "modality": "text"})
        for _ in range(10):
            m = ws.receive_json()
            if m["type"] == "response":
                break
        assert m["response"]["safety"]["state"] == "escalation_countdown"
        assert any(st["ref"] == "kb:kb.chest_pain" for st in m["response"]["statements"])
        tr = c.get(f"/api/trace/{m['trace_id']}").json()
        assert tr["nlu"]["intent"] == "emergency_help"
        ws.send_json({"type": "ui", "action": "cancel"})
        for _ in range(10):
            m = ws.receive_json()
            if m["type"] == "response":
                break
        assert m["response"]["safety"]["state"] == "monitoring"


def test_policy_floors_cannot_disable_safety_net():
    c, _ = client()
    with c:
        p = c.post("/api/policy", json={"checkin_timeout_s": 100000, "countdown_s": 0}).json()
        assert p["checkin_timeout_s"] <= 300 and p["countdown_s"] >= 5


def test_delete_all_requires_typed_confirmation():
    c, _ = client()
    with c:
        assert c.post("/api/delete_all", json={}).status_code == 400
        assert c.post("/api/delete_all", json={"confirm": "DELETE"}).status_code == 200


def test_escalation_config_validation():
    c, _ = client()
    with c:
        bad = c.post("/api/escalation", json={"contacts": [{"name": "x", "channel": "webhook", "address": "not-a-url"}]})
        assert bad.status_code == 400
        ok = c.post("/api/escalation", json={"emergency_number": "911", "contacts": [
            {"name": "x", "channel": "webhook", "address": "https://ntfy.sh/topic", "lang": "es"}]})
        assert ok.json()["emergency_number"] == "911"


def test_audit_verify_and_models_endpoints():
    c, _ = client()
    with c:
        assert c.get("/api/audit/verify").json()["ok"]
        ids = {m["model_id"] for m in c.get("/api/models").json()["models"]}
        assert {"nlu_lexicon_v1", "fall_kinematic_v1", "llm"} <= ids
        assert c.get("/api/ref", params={"ref": "kb:kb.burn"}).json()["type"] == "knowledge"
        assert c.get("/api/ref", params={"ref": "kb:nonexistent"}).status_code == 404
