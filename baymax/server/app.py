"""Local web server: serves the UI and bridges browser sensors to the engine.

Binds to 127.0.0.1 by default. The browser sends only derived features
(body-position numbers, audio RMS levels) and transcripts / speech segments;
video frames never leave the browser.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from anyio import to_thread
from fastapi import Body, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .. import __version__
from ..audio.stt import LocalSTT
from ..audio.wake import detect_wake
from ..contracts import AudioLevel, BodySample, Utterance
from ..conversation.llm import backend_from_env
from ..engine import Engine
from ..knowledge.kb import default_kb
from ..language import phrases as ph
from ..models.registry import REGISTRY
from ..ontology import CONSENT_DESCRIPTIONS, SUPPORTED_LANGS, ConsentScope, Modality
from ..safety.escalation import Contact, EscalationConfig
from ..store.db import Store

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "web"
DATA_DIR = Path(os.environ.get("BAYMAX_DATA", ROOT / "data"))
REPORTS = ROOT / "reports"


class ConsentBody(BaseModel):
    scope: str
    granted: bool


def create_app(store: Optional[Store] = None, engine: Optional[Engine] = None) -> FastAPI:
    store = store or Store(DATA_DIR / "baymax.db")
    engine = engine or Engine(store, llm_backend=backend_from_env())
    stt = LocalSTT()
    clients: set[asyncio.Queue] = set()
    loop_holder: dict[str, asyncio.AbstractEventLoop] = {}

    def broadcast(msg: dict) -> None:
        loop = loop_holder.get("loop")
        if not loop:
            return
        for q in list(clients):
            loop.call_soon_threadsafe(q.put_nowait, msg)

    engine.listeners.append(broadcast)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        loop_holder["loop"] = asyncio.get_running_loop()

        async def ticker() -> None:
            while True:
                await asyncio.sleep(0.5)
                try:
                    await to_thread.run_sync(engine.tick)
                except Exception as e:  # never let the safety timer loop die
                    store.audit("system", "tick_error", error=str(e)[:200])

        async def status() -> None:
            while True:
                await asyncio.sleep(1.0)
                broadcast({"type": "status", "safety": engine.machine.ctx.model_dump(mode="json"),
                           "perception": engine.perception.status(), "privacy": engine.privacy,
                           "server_time": time.time()})

        tasks = [asyncio.create_task(ticker()), asyncio.create_task(status())]
        yield
        for tk in tasks:
            tk.cancel()

    app = FastAPI(title="BAYMAX OS", version=__version__, lifespan=lifespan)
    app.state.engine = engine
    app.state.store = store


    # ----------------------------------------------------------------- UI
    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(WEB / "index.html")

    app.mount("/static", StaticFiles(directory=WEB), name="static")

    # ------------------------------------------------------------ realtime
    @app.websocket("/ws")
    async def ws(sock: WebSocket) -> None:
        await sock.accept()
        q: asyncio.Queue = asyncio.Queue()
        clients.add(q)

        async def pump() -> None:
            while True:
                msg = await q.get()
                await sock.send_text(json.dumps(msg, ensure_ascii=False, default=str))

        pump_task = asyncio.create_task(pump())
        try:
            while True:
                msg = json.loads(await sock.receive_text())
                t = msg.get("type")
                if t == "utterance":
                    await to_thread.run_sync(_handle_utterance, engine, msg)
                elif t == "body":
                    samples = [BodySample(**s) for s in msg.get("samples", [])][:100]
                    await to_thread.run_sync(engine.ingest_body, samples)
                elif t == "audio":
                    levels = [AudioLevel(**a) for a in msg.get("levels", [])][:100]
                    await to_thread.run_sync(engine.ingest_audio, levels)
                elif t == "ui":
                    await to_thread.run_sync(engine.ui_action, msg.get("action", ""))
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            pump_task.cancel()
            clients.discard(q)

    # ----------------------------------------------------------------- REST
    @app.get("/api/state")
    def state() -> dict:
        return {
            "version": __version__,
            "profile": engine.profile.model_dump(),
            "policy": engine.policy.model_dump(),
            "consents": store.consents(),
            "consent_descriptions": {s.value: d for s, d in CONSENT_DESCRIPTIONS.items()},
            "safety": engine.machine.ctx.model_dump(mode="json"),
            "privacy": engine.privacy,
            "escalation": engine.escalation.model_dump(),
            "stt": {"local_available": stt.available, "model_id": stt.model_id if stt.available else None, "error": stt.error},
            "llm": {"backend": engine.llm.name if engine.llm else None, "cloud": bool(engine.llm and engine.llm.is_cloud)},
            "languages": list(SUPPORTED_LANGS),
            "session_lang": engine.session_lang,
            "onboarded": store.get_setting("onboarded", False),
        }

    @app.post("/api/consent")
    def set_consent(b: ConsentBody) -> dict:
        try:
            scope = ConsentScope(b.scope)
        except ValueError:
            raise HTTPException(400, "unknown scope")
        store.set_consent(scope, b.granted, source="ui")
        return store.consents()

    @app.post("/api/onboarded")
    def onboarded() -> dict:
        store.set_setting("onboarded", True)
        return {"ok": True}

    @app.post("/api/profile")
    def set_profile(b: dict = Body(...)) -> dict:
        allowed = {"user_name", "preferred_lang", "require_wake_word", "inactivity_minutes", "llm_enabled"}
        if b.get("preferred_lang") not in (None, "auto", *SUPPORTED_LANGS):
            raise HTTPException(400, "unsupported language")
        return engine.update_profile(**{k: v for k, v in b.items() if k in allowed}).model_dump()

    @app.post("/api/policy")
    def set_policy(b: dict = Body(...)) -> dict:
        p = engine.policy.model_dump()
        for k, v in b.items():
            if k in p:
                p[k] = v
        # hard floors: a misconfiguration must not disable the safety net
        p["checkin_timeout_s"] = max(10, min(300, int(p["checkin_timeout_s"])))
        p["countdown_s"] = max(5, min(120, int(p["countdown_s"])))
        p["fall_countdown_s"] = max(5, min(180, int(p["fall_countdown_s"])))
        return engine.update_policy(**p).model_dump()

    @app.post("/api/escalation")
    def set_escalation(b: dict = Body(...)) -> dict:
        contacts = [Contact(**c) for c in b.get("contacts", [])]
        for c in contacts:
            if c.channel == "webhook" and not c.address.startswith(("http://", "https://")):
                raise HTTPException(400, f"contact {c.name}: webhook address must be an http(s) URL")
            if c.lang not in SUPPORTED_LANGS:
                raise HTTPException(400, f"contact {c.name}: unsupported language")
        cfg = EscalationConfig(emergency_number=str(b.get("emergency_number", engine.escalation.emergency_number))[:16],
                               dry_run=bool(b.get("dry_run", engine.escalation.dry_run)), contacts=contacts,
                               user_display_name=str(b.get("user_display_name", engine.escalation.user_display_name))[:60])
        return engine.update_escalation(cfg).model_dump()

    @app.post("/api/escalation/test")
    def test_escalation() -> dict:
        res = engine.dispatcher.dispatch(engine.escalation, "test", "", store.has_consent(ConsentScope.EMERGENCY_CONTACTS),
                                         False, force_send=True)
        store.audit("user", "test_alert_sent", status=res.status)
        return res.model_dump()

    @app.get("/api/timeline")
    def timeline(days: float = 30, kind: Optional[str] = None) -> dict:
        evs = store.events(since=time.time() - days * 86400, kind=kind, limit=2000)
        return {"events": [e.model_dump(mode="json") for e in evs]}

    @app.delete("/api/events/{event_id}")
    def delete_event(event_id: str) -> dict:
        return {"deleted": store.delete_event(event_id)}

    @app.get("/api/memories")
    def memories() -> dict:
        return {"memories": store.memories()}

    @app.delete("/api/memories/{memory_id}")
    def delete_memory(memory_id: str) -> dict:
        return {"deleted": store.forget_memory(memory_id)}

    @app.get("/api/export")
    def export() -> Response:
        data = json.dumps(store.export_all(), ensure_ascii=False, indent=2, default=str)
        return Response(data, media_type="application/json",
                        headers={"Content-Disposition": "attachment; filename=baymax-export.json"})

    @app.post("/api/delete_all")
    def delete_all(b: dict = Body(...)) -> dict:
        if b.get("confirm") != "DELETE":
            raise HTTPException(400, "type DELETE to confirm")
        return {"deleted": store.delete_all(actor="user")}

    @app.get("/api/audit")
    def audit(limit: int = 100) -> dict:
        return {"entries": store.audit_log(limit=min(limit, 1000))}

    @app.get("/api/audit/verify")
    def audit_verify() -> dict:
        return store.verify_audit()

    @app.get("/api/trace/{trace_id}")
    def trace(trace_id: str) -> dict:
        tr = engine.traces.get(trace_id)
        if not tr:
            raise HTTPException(404)
        return tr.model_dump(mode="json")

    @app.get("/api/ref")
    def ref(ref: str) -> dict:
        r = engine.resolve_ref(ref)
        if r is None:
            raise HTTPException(404, "unresolvable reference")
        return r

    @app.get("/api/models")
    def models() -> dict:
        return {"models": [c.model_dump() for c in REGISTRY.values()]}

    @app.get("/api/kb")
    def kb(lang: str = "en") -> dict:
        k = default_kb()
        lang = lang if lang in SUPPORTED_LANGS else "en"
        return {"version": k.version, "notice": k.notice,
                "entries": [{"id": e.id, "title": e.title[lang], "text": e.text[lang], "concepts": e.concepts,
                             "source": e.source, "review_status": e.review_status, "red_flag": e.red_flag}
                            for e in k.entries.values()]}

    @app.get("/api/names/{lang}")
    def names(lang: str) -> dict:
        if lang not in SUPPORTED_LANGS:
            raise HTTPException(404)
        return {k: v[lang] for k, v in ph.NAMES.items()}

    @app.get("/api/eval/latest")
    def eval_latest() -> JSONResponse:
        p = REPORTS / "eval_latest.json"
        if not p.exists():
            raise HTTPException(404, "no evaluation report yet; run `python -m baymax.eval`")
        return JSONResponse(json.loads(p.read_text()))

    @app.post("/api/stt")
    async def stt_endpoint(request: Request, lang: str = "auto") -> dict:
        if not store.has_consent(ConsentScope.MICROPHONE):
            raise HTTPException(403, "microphone consent required")
        if not stt.available:
            raise HTTPException(501, stt.error or "local STT unavailable")
        wav = await request.body()
        if len(wav) > 16000 * 2 * 20 + 44:
            raise HTTPException(413, "segment too long")
        r = await to_thread.run_sync(stt.transcribe, wav, lang)
        return {"text": r.text, "language": r.language, "confidence": r.confidence, "model_id": r.model_id,
                "no_speech_prob": r.no_speech_prob}

    return app


def _handle_utterance(engine: Engine, msg: dict) -> None:
    text = str(msg.get("text", ""))[:1000].strip()
    if not text:
        return
    modality = Modality.SPEECH if msg.get("modality") == "speech" else Modality.TEXT
    wake = False
    if modality == Modality.SPEECH:
        w = detect_wake(text)
        wake = w.found
        if w.found:
            text = w.remainder.strip() or text
    utt = Utterance(text=text, modality=modality, stt_model_id=msg.get("stt_model_id"),
                    stt_confidence=msg.get("stt_confidence"), stt_language=msg.get("stt_language"), wake_word=wake)
    if wake and not w.remainder.strip():
        # bare wake word: acknowledge without running the pipeline
        from ..conversation import compose
        from ..contracts import BaymaxResponse
        lang = engine._lang_for_system_turn()
        resp = BaymaxResponse(lang=lang, statements=[compose.phrase("listening", lang)], safety=engine.machine.ctx,
                              ui_hints=engine._ui_hints({"wake": True}))
        engine._emit({"type": "response", "trace_id": None, "response": resp.model_dump(mode="json")})
        return
    engine.process_utterance(utt)


def main() -> None:
    import uvicorn

    host = os.environ.get("BAYMAX_HOST", "127.0.0.1")
    port = int(os.environ.get("BAYMAX_PORT", "8765"))
    print(f"BAYMAX OS {__version__} → http://{host}:{port}")
    uvicorn.run(create_app(), host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
