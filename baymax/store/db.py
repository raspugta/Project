"""Local SQLite store: health events, memory, consents, settings, audit log.

Design rules
- Raw camera frames and raw audio are never stored (the server never receives them).
- Health events / transcripts / memories are only written when the matching
  consent scope is granted; the store enforces this, not the caller.
- The audit log is append-only and hash-chained; `verify_audit()` detects edits.
  Audit entries hold ids, rule ids and decisions, never transcript text.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional

from ..contracts import HealthEvent, new_id
from ..ontology import DEFAULT_CONSENTS, ConsentScope

SCHEMA = """
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS consents (scope TEXT PRIMARY KEY, granted INTEGER NOT NULL, ts REAL NOT NULL, source TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS events (
  id TEXT PRIMARY KEY, ts REAL NOT NULL, kind TEXT NOT NULL, severity TEXT NOT NULL,
  provenance TEXT NOT NULL, source_ref TEXT, model_id TEXT, confidence REAL, lang TEXT, data TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS events_ts ON events(ts);
CREATE TABLE IF NOT EXISTS utterances (id TEXT PRIMARY KEY, ts REAL NOT NULL, text TEXT NOT NULL, lang TEXT, intent TEXT, role TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS memories (
  id TEXT PRIMARY KEY, ts REAL NOT NULL, kind TEXT NOT NULL, value TEXT NOT NULL,
  source_ref TEXT, lang TEXT, active INTEGER NOT NULL DEFAULT 1);
CREATE TABLE IF NOT EXISTS escalations (id TEXT PRIMARY KEY, ts REAL NOT NULL, reason TEXT, level TEXT, status TEXT, detail TEXT);
CREATE TABLE IF NOT EXISTS audit (
  seq INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, actor TEXT NOT NULL, action TEXT NOT NULL,
  detail TEXT NOT NULL, prev_hash TEXT NOT NULL, hash TEXT NOT NULL);
"""

GENESIS = "0" * 64


class ConsentError(PermissionError):
    pass


def _canon(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


class Store:
    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._db.executescript(SCHEMA)
            if self.path != ":memory:":
                self._db.execute("PRAGMA journal_mode=WAL")

    def _q(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._db.execute(sql, args).fetchall()

    # ---------------------------------------------------------------- audit
    def audit(self, actor: str, action: str, **detail: Any) -> str:
        with self._lock:
            row = self._db.execute("SELECT hash FROM audit ORDER BY seq DESC LIMIT 1").fetchone()
            prev = row["hash"] if row else GENESIS
            ts = time.time()
            payload = {"ts": round(ts, 6), "actor": actor, "action": action, "detail": detail}
            h = hashlib.sha256((prev + _canon(payload)).encode()).hexdigest()
            self._db.execute("INSERT INTO audit (ts, actor, action, detail, prev_hash, hash) VALUES (?,?,?,?,?,?)",
                             (round(ts, 6), actor, action, _canon(detail), prev, h))
            return h

    def audit_log(self, limit: int = 200) -> list[dict]:
        rows = self._q("SELECT * FROM audit ORDER BY seq DESC LIMIT ?", (limit,))
        return [{**dict(r), "detail": json.loads(r["detail"])} for r in rows]

    def verify_audit(self) -> dict:
        prev = GENESIS
        n = 0
        for r in self._q("SELECT * FROM audit ORDER BY seq ASC"):
            payload = {"ts": r["ts"], "actor": r["actor"], "action": r["action"], "detail": json.loads(r["detail"])}
            h = hashlib.sha256((prev + _canon(payload)).encode()).hexdigest()
            if r["prev_hash"] != prev or r["hash"] != h:
                return {"ok": False, "entries": n, "broken_at": r["seq"]}
            prev = h
            n += 1
        return {"ok": True, "entries": n, "head": prev}

    # ------------------------------------------------------------- consents
    def consents(self) -> dict[str, bool]:
        out = {s.value: v for s, v in DEFAULT_CONSENTS.items()}
        for r in self._q("SELECT scope, granted FROM consents"):
            out[r["scope"]] = bool(r["granted"])
        return out

    def has_consent(self, scope: ConsentScope) -> bool:
        r = self._q("SELECT granted FROM consents WHERE scope=?", (scope.value,))
        return bool(r[0]["granted"]) if r else DEFAULT_CONSENTS[scope]

    def set_consent(self, scope: ConsentScope, granted: bool, source: str = "ui") -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO consents (scope, granted, ts, source) VALUES (?,?,?,?)",
                             (scope.value, int(granted), time.time(), source))
        self.audit(source, "consent_changed", scope=scope.value, granted=granted)
        if not granted:
            self._enforce_revocation(scope)

    def _enforce_revocation(self, scope: ConsentScope) -> None:
        """Revoking a storage consent deletes what it covered."""
        if scope is ConsentScope.STORE_TRANSCRIPTS:
            n = self._q("SELECT COUNT(*) c FROM utterances")[0]["c"]
            self._q("DELETE FROM utterances")
            self.audit("system", "data_deleted_on_revocation", scope=scope.value, rows=n)
        elif scope is ConsentScope.STORE_HEALTH_EVENTS:
            n = self._q("SELECT COUNT(*) c FROM events")[0]["c"]
            self._q("DELETE FROM events")
            self.audit("system", "data_deleted_on_revocation", scope=scope.value, rows=n)
        elif scope is ConsentScope.LONG_TERM_MEMORY:
            n = self._q("SELECT COUNT(*) c FROM memories")[0]["c"]
            self._q("DELETE FROM memories")
            self.audit("system", "data_deleted_on_revocation", scope=scope.value, rows=n)

    # ------------------------------------------------------------- settings
    def get_setting(self, key: str, default: Any = None) -> Any:
        r = self._q("SELECT value FROM settings WHERE key=?", (key,))
        return json.loads(r[0]["value"]) if r else default

    def set_setting(self, key: str, value: Any, actor: str = "ui") -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?,?)", (key, _canon(value)))
        self.audit(actor, "setting_changed", key=key)

    # --------------------------------------------------------------- events
    def add_event(self, ev: HealthEvent, emergency: bool = False) -> bool:
        """Store a health event if consented. Emergency/escalation records are
        kept regardless (legitimate-interest safety record), but only as
        structured data without transcript text."""
        if not (self.has_consent(ConsentScope.STORE_HEALTH_EVENTS) or emergency):
            return False
        data = ev.model_dump(mode="json", exclude={"id", "ts", "kind", "severity", "provenance", "source_ref",
                                                   "model_id", "confidence", "lang"})
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO events (id, ts, kind, severity, provenance, source_ref, model_id, confidence, lang, data)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)",
                (ev.id, ev.ts, ev.kind.value, ev.severity.value, ev.provenance.value, ev.source_ref, ev.model_id,
                 ev.confidence, ev.lang, _canon(data)))
        self.audit("system", "event_stored", event_id=ev.id, kind=ev.kind.value, provenance=ev.provenance.value)
        return True

    def events(self, since: float = 0, until: Optional[float] = None, kind: Optional[str] = None, limit: int = 500) -> list[HealthEvent]:
        sql, args = "SELECT * FROM events WHERE ts >= ?", [since]
        if until is not None:
            sql += " AND ts <= ?"
            args.append(until)
        if kind:
            sql += " AND kind = ?"
            args.append(kind)
        sql += " ORDER BY ts DESC LIMIT ?"
        args.append(limit)
        out = []
        for r in self._q(sql, tuple(args)):
            d = json.loads(r["data"])
            out.append(HealthEvent(id=r["id"], ts=r["ts"], kind=r["kind"], severity=r["severity"], provenance=r["provenance"],
                                   source_ref=r["source_ref"], model_id=r["model_id"], confidence=r["confidence"], lang=r["lang"], **d))
        return out

    def get_event(self, event_id: str) -> Optional[HealthEvent]:
        for e in self.events(limit=100000):
            if e.id == event_id:
                return e
        return None

    def delete_event(self, event_id: str, actor: str = "user") -> bool:
        with self._lock:
            n = self._db.execute("DELETE FROM events WHERE id=?", (event_id,)).rowcount
        if n:
            self.audit(actor, "event_deleted", event_id=event_id)
        return bool(n)

    def delete_events_from(self, source_ref: str, actor: str = "user") -> int:
        with self._lock:
            n = self._db.execute("DELETE FROM events WHERE source_ref=?", (source_ref,)).rowcount
            n += self._db.execute("UPDATE memories SET active=0 WHERE source_ref=?", (source_ref,)).rowcount
            n += self._db.execute("DELETE FROM utterances WHERE id=?", (source_ref,)).rowcount
        self.audit(actor, "forget_utterance", source_ref=source_ref, rows=n)
        return n

    # ----------------------------------------------------------- utterances
    def add_utterance(self, utt_id: str, text: str, lang: str, intent: str, role: str = "user") -> bool:
        if not self.has_consent(ConsentScope.STORE_TRANSCRIPTS):
            return False
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO utterances (id, ts, text, lang, intent, role) VALUES (?,?,?,?,?,?)",
                             (utt_id, time.time(), text, lang, intent, role))
        return True

    def utterances(self, limit: int = 100) -> list[dict]:
        return [dict(r) for r in self._q("SELECT * FROM utterances ORDER BY ts DESC LIMIT ?", (limit,))]

    # ------------------------------------------------------------- memories
    def add_memory(self, kind: str, value: str, source_ref: str, lang: str) -> Optional[str]:
        if not self.has_consent(ConsentScope.LONG_TERM_MEMORY):
            return None
        mid = new_id("mem")
        with self._lock:
            self._db.execute("INSERT INTO memories (id, ts, kind, value, source_ref, lang, active) VALUES (?,?,?,?,?,?,1)",
                             (mid, time.time(), kind, value, source_ref, lang))
        self.audit("user", "memory_saved", memory_id=mid, kind=kind)
        return mid

    def memories(self, active_only: bool = True) -> list[dict]:
        sql = "SELECT * FROM memories" + (" WHERE active=1" if active_only else "") + " ORDER BY ts DESC"
        return [dict(r) for r in self._q(sql)]

    def forget_memory(self, memory_id: str, actor: str = "user") -> bool:
        with self._lock:
            n = self._db.execute("DELETE FROM memories WHERE id=?", (memory_id,)).rowcount
        if n:
            self.audit(actor, "memory_deleted", memory_id=memory_id)
        return bool(n)

    # ----------------------------------------------------------- escalations
    def add_escalation(self, esc_id: str, reason: str, level: str, status: str, detail: dict) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO escalations (id, ts, reason, level, status, detail) VALUES (?,?,?,?,?,?)",
                             (esc_id, time.time(), reason, level, status, _canon(detail)))

    def escalations(self, limit: int = 50) -> list[dict]:
        return [{**dict(r), "detail": json.loads(r["detail"])} for r in
                self._q("SELECT * FROM escalations ORDER BY ts DESC LIMIT ?", (limit,))]

    # ------------------------------------------------------ export / delete
    def export_all(self) -> dict:
        self.audit("user", "data_exported")
        return {
            "exported_at": time.time(),
            "consents": self.consents(),
            "settings": {r["key"]: json.loads(r["value"]) for r in self._q("SELECT * FROM settings")},
            "events": [e.model_dump(mode="json") for e in self.events(limit=1000000)],
            "utterances": self.utterances(limit=1000000),
            "memories": self.memories(active_only=False),
            "escalations": self.escalations(limit=1000000),
            "audit": self.audit_log(limit=1000000),
        }

    def delete_all(self, actor: str = "user") -> dict:
        counts = {}
        with self._lock:
            for t in ("events", "utterances", "memories", "escalations"):
                counts[t] = self._db.execute(f"DELETE FROM {t}").rowcount
        # The audit log records THAT data was deleted (not what it was).
        self.audit(actor, "all_data_deleted", **counts)
        return counts
