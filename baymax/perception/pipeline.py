"""Perception models operating on body features computed in the browser.

Input:  BodySample (pose landmarks summary or motion-blob summary) at ~5-15 Hz,
        AudioLevel (RMS dBFS) at ~10 Hz. No frames, no audio.
Output: PerceptionEvent with model_id, confidence, evidence and failure_flags.

See model cards fall_kinematic_v1, activity_v1, posture_v1, impact_audio_v1.
"""
from __future__ import annotations

import statistics
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional

from ..contracts import AudioLevel, BodySample, PerceptionEvent


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _is_lying(s: BodySample) -> Optional[bool]:
    if not s.present:
        return None
    if s.source == "pose" and s.torso_angle is not None:
        return s.torso_angle >= 55
    if s.bbox_w and s.bbox_h:
        return s.bbox_w / max(s.bbox_h, 1e-3) >= 1.15
    return None


# ---------------------------------------------------------------------------
@dataclass
class FallDetector:
    model_id: str = "fall_kinematic_v1"
    descent_window_s: float = 1.2
    post_window_s: float = 2.0
    emit_threshold: float = 0.6
    cooldown_s: float = 20.0
    buf: Deque[BodySample] = field(default_factory=lambda: deque(maxlen=200))
    _candidate: Optional[dict] = None
    _last_emit: float = -1e9

    def update(self, s: BodySample) -> list[PerceptionEvent]:
        self.buf.append(s)
        out: list[PerceptionEvent] = []
        if s.t - self._last_emit < self.cooldown_s:
            return out
        cand = self._descent(s.t)
        if cand and cand["drop_score"] >= 0.5 and cand["velocity_score"] >= 0.2:
            cur = self._candidate
            # keep extending the candidate while the body is still going down
            if cur is None or (cand["drop"] > cur["drop"] and s.t - cur["t_end"] < 1.0):
                self._candidate = cand
        if self._candidate is not None and s.t - self._candidate["t_end"] >= self.post_window_s:
            ev = self._evaluate(self._candidate, s.t)
            self._candidate = None
            if ev and ev.confidence >= self.emit_threshold:
                self._last_emit = s.t
                out.append(ev)
        return out

    def _descent(self, now: float) -> Optional[dict]:
        win = [x for x in self.buf if now - x.t <= self.descent_window_s and x.present and x.center_y is not None]
        if len(win) < 3:
            return None
        # largest top->bottom drop where the top precedes the bottom
        best = None
        lo_i = 0
        for j in range(1, len(win)):
            if win[j].center_y < win[lo_i].center_y:
                lo_i = j
            drop = win[j].center_y - win[lo_i].center_y
            if best is None or drop > best[0]:
                best = (drop, lo_i, j)
        if not best or best[0] <= 0:
            return None
        drop, i, j = best
        start, end = win[i], win[j]
        ref_h = start.bbox_h or 0.4
        drop_ratio = drop / max(ref_h, 0.2)
        vmax = 0.0
        for a, b in zip(win[i:j], win[i + 1:j + 1]):
            dt = b.t - a.t
            if dt > 0:
                vmax = max(vmax, (b.center_y - a.center_y) / dt)
        drop_score = _clip((drop_ratio - 0.2) / 0.2)       # hip centre falls by >= ~40% of body height
        velocity_score = _clip((vmax - 0.3) / 0.5)          # and does so fast (normalised units / s)
        descent = 0.5 * drop_score + 0.5 * velocity_score
        return {"t_start": start.t, "t_end": end.t, "drop": round(drop, 3), "drop_ratio": round(drop_ratio, 3),
                "peak_velocity": round(vmax, 3), "drop_score": round(drop_score, 3),
                "velocity_score": round(velocity_score, 3), "descent": round(descent, 3),
                "pre_visibility": start.visibility}

    def _evaluate(self, c: dict, now: float) -> Optional[PerceptionEvent]:
        post = [x for x in self.buf if c["t_end"] < x.t <= now]
        flags: list[str] = []
        present = [x for x in post if x.present]
        if not present:
            flags.append("left_frame_after_descent")
            lying_score, still_score, vis = 0.6, 0.5, c["pre_visibility"]
        else:
            lying_votes = [_is_lying(x) for x in present]
            known = [v for v in lying_votes if v is not None]
            if known:
                lying_score = sum(known) / len(known)
            else:  # no posture evidence: fall back to "is the body low in the frame"
                lying_score = 0.6 * sum(1 for x in present if (x.center_y or 0) >= 0.72) / len(present)
            ys = [x.center_y for x in present if x.center_y is not None]
            motion = [x.motion for x in present if x.motion is not None]
            spread = (max(ys) - min(ys)) if len(ys) > 1 else 0.0
            still_score = _clip(1 - spread / 0.15)
            if motion:
                still_score = min(still_score, _clip(1 - statistics.mean(motion) / 0.3))
            vis = statistics.mean(x.visibility for x in present)
            if any(x.persons > 1 for x in present):
                flags.append("multiple_people")
            if not known:
                flags.append("posture_unknown")
        if vis < 0.5:
            flags.append("low_visibility")
        vis_factor = _clip(vis / 0.6, 0.4, 1.0)
        conf = (0.45 * c["descent"] + 0.35 * lying_score + 0.2 * still_score) * vis_factor
        return PerceptionEvent(kind="fall", model_id=self.model_id, confidence=round(conf, 3),
                               evidence={**{k: v for k, v in c.items() if k != "pre_visibility"},
                                         "lying": round(lying_score, 3), "stillness": round(still_score, 3),
                                         "visibility": round(vis, 3)},
                               failure_flags=flags)


# ---------------------------------------------------------------------------
@dataclass
class ActivityClassifier:
    model_id: str = "activity_v1"
    inactivity_s: float = 300.0
    buf: Deque[BodySample] = field(default_factory=lambda: deque(maxlen=60))
    label: str = "absent"
    confidence: float = 0.0
    _lying_since: Optional[float] = None
    _inactive_emitted: bool = False
    _absent_since: Optional[float] = None

    def classify(self, s: BodySample) -> str:
        if not s.present:
            return "absent"
        if _is_lying(s):
            return "lying"
        if (s.motion or 0) > 0.12:
            return "moving"
        return "upright_still"

    def update(self, s: BodySample) -> list[PerceptionEvent]:
        self.buf.append(s)
        labels = [self.classify(x) for x in self.buf if s.t - x.t <= 2.0]
        top = max(set(labels), key=labels.count)
        vis = statistics.mean(x.visibility for x in self.buf if s.t - x.t <= 2.0)
        self.confidence = round(labels.count(top) / len(labels) * (vis if top != "absent" else 1.0), 3)
        out: list[PerceptionEvent] = []
        if top != self.label:
            self.label = top
            out.append(PerceptionEvent(kind="activity", model_id=self.model_id, confidence=self.confidence,
                                       evidence={"label": top}))
        lying_still = top == "lying" and (s.motion is None or s.motion < 0.05)
        if lying_still:
            self._lying_since = self._lying_since if self._lying_since is not None else s.t
            if not self._inactive_emitted and s.t - self._lying_since >= self.inactivity_s:
                self._inactive_emitted = True
                out.append(PerceptionEvent(kind="lying_inactive", model_id=self.model_id, confidence=self.confidence,
                                           evidence={"lying_seconds": round(s.t - self._lying_since, 1)}))
        else:
            self._lying_since = None
            self._inactive_emitted = False
        return out


# ---------------------------------------------------------------------------
@dataclass
class PostureAnalyzer:
    model_id: str = "posture_v1"
    neck_threshold_deg: float = 25.0
    sustain_s: float = 60.0
    cooldown_s: float = 900.0
    buf: Deque[BodySample] = field(default_factory=lambda: deque(maxlen=2000))
    _last_emit: float = -1e9

    def update(self, s: BodySample) -> list[PerceptionEvent]:
        if s.source != "pose" or s.neck_forward is None or not s.present:
            return []
        self.buf.append(s)
        win = [x for x in self.buf if s.t - x.t <= self.sustain_s]
        if s.t - win[0].t < self.sustain_s * 0.9 or s.t - self._last_emit < self.cooldown_s:
            return []
        frac = sum(1 for x in win if (x.neck_forward or 0) > self.neck_threshold_deg) / len(win)
        if frac >= 0.7:
            self._last_emit = s.t
            vis = statistics.mean(x.visibility for x in win)
            return [PerceptionEvent(kind="posture_poor", model_id=self.model_id, confidence=round(frac * vis, 3),
                                    evidence={"fraction_over_threshold": round(frac, 3),
                                              "mean_neck_forward": round(statistics.mean(x.neck_forward or 0 for x in win), 1)})]
        return []


# ---------------------------------------------------------------------------
@dataclass
class ImpactDetector:
    model_id: str = "impact_audio_v1"
    jump_db: float = 25.0
    floor_db: float = -35.0
    buf: Deque[AudioLevel] = field(default_factory=lambda: deque(maxlen=60))
    _last: float = -1e9

    def update(self, a: AudioLevel) -> list[PerceptionEvent]:
        base = statistics.median([x.rms_db for x in self.buf]) if len(self.buf) >= 5 else None
        self.buf.append(a)
        if base is None or a.t - self._last < 2.0:
            return []
        jump = a.rms_db - base
        if jump >= self.jump_db and a.rms_db >= self.floor_db:
            self._last = a.t
            return [PerceptionEvent(kind="loud_impact", model_id=self.model_id, confidence=round(_clip((jump - 20) / 20), 3),
                                    evidence={"jump_db": round(jump, 1), "level_db": round(a.rms_db, 1)},
                                    failure_flags=["not_specific_to_falls"])]
        return []


# ---------------------------------------------------------------------------
class PerceptionHub:
    """Runs all perception models and fuses fall + impact evidence."""

    def __init__(self, inactivity_s: float = 300.0) -> None:
        self.fall = FallDetector()
        self.activity = ActivityClassifier(inactivity_s=inactivity_s)
        self.posture = PostureAnalyzer()
        self.impact = ImpactDetector()
        self._impacts: Deque[float] = deque(maxlen=20)
        self.last_sample: Optional[BodySample] = None

    def ingest_body(self, samples: list[BodySample]) -> list[PerceptionEvent]:
        out: list[PerceptionEvent] = []
        for s in samples:
            self.last_sample = s
            for ev in self.fall.update(s):
                if any(abs(t - ev.evidence.get("t_end", s.t)) <= 2.5 for t in self._impacts):
                    ev.confidence = round(min(1.0, ev.confidence + 0.1), 3)
                    ev.evidence["impact_corroborated"] = True
                out.append(ev)
            out += self.activity.update(s)
            out += self.posture.update(s)
        return out

    def ingest_audio(self, levels: list[AudioLevel]) -> list[PerceptionEvent]:
        out: list[PerceptionEvent] = []
        for a in levels:
            evs = self.impact.update(a)
            for e in evs:
                self._impacts.append(a.t)
            out += evs
        return out

    def status(self) -> dict:
        s = self.last_sample
        return {"activity": self.activity.label, "activity_confidence": self.activity.confidence,
                "source": s.source if s else None, "visibility": round(s.visibility, 2) if s else None,
                "persons": s.persons if s else 0}
