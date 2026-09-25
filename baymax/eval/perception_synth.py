"""Synthetic body-trajectory scenarios for evaluating perception models.

Each scenario is a labelled BodySample sequence produced from a parametric
kinematic template with random timing, amplitude, sensor noise, visibility
and camera mode (pose landmarks vs motion-blob only). Negatives are chosen to
be the classic false-positive traps for fall detection.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

from ..contracts import BodySample

HZ = 10.0

POSITIVE = ["fall_forward", "fall_backward", "fall_sideways", "fall_collapse", "fall_out_of_frame"]
NEGATIVE = ["sit_chair", "sit_floor_controlled", "lie_down_bed", "bend_pick_up", "jump", "walk_across",
            "plop_on_sofa", "stand_still", "leave_frame_walking", "exercise_squats"]


@dataclass
class Scenario:
    id: str
    label: str
    is_fall: bool
    mode: str
    samples: list[BodySample]


def _ease(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3 - 2 * x)


def _seq(rng: random.Random, mode: str, keyframes: list[tuple[float, dict]], total_s: float,
         vis: float, present_fn=None, noise: float = 1.0) -> list[BodySample]:
    """Interpolate keyframes (time, state) into noisy samples."""
    out = []
    n = int(total_s * HZ)
    prev = None
    for i in range(n):
        t = i / HZ
        # find segment
        k = 0
        while k + 1 < len(keyframes) and keyframes[k + 1][0] <= t:
            k += 1
        t0, s0 = keyframes[k]
        if k + 1 < len(keyframes):
            t1, s1 = keyframes[k + 1]
            a = _ease((t - t0) / max(t1 - t0, 1e-6))
        else:
            s1, a = s0, 0.0
        st = {key: s0[key] + (s1.get(key, s0[key]) - s0[key]) * a for key in s0}
        cy = st["cy"] + rng.gauss(0, 0.006 * noise)
        cx = st.get("cx", 0.5) + rng.gauss(0, 0.004 * noise)
        torso = st["torso"] + rng.gauss(0, 3.0 * noise)
        bh, bw = st["bh"] + rng.gauss(0, 0.01 * noise), st["bw"] + rng.gauss(0, 0.01 * noise)
        motion = abs(cy - prev) * 8 + abs(st.get("dx", 0.0)) if prev is not None else 0.02
        motion = min(1.0, motion + abs(rng.gauss(0.02, 0.01)))
        prev = cy
        present = present_fn(t, st) if present_fn else True
        v = max(0.05, min(1.0, vis + rng.gauss(0, 0.05 * noise)))
        if noise > 1 and rng.random() < 0.03 * noise:   # dropped detections
            out.append(BodySample(t=t, present=False, source=mode, visibility=0.0, persons=0))
            continue
        if not present:
            out.append(BodySample(t=t, present=False, source=mode, visibility=0.0, persons=0))
            continue
        out.append(BodySample(
            t=t, present=True, source=mode, center_y=cy, center_x=cx, head_y=cy - bh * 0.45,
            bbox_w=max(0.05, bw), bbox_h=max(0.05, bh),
            torso_angle=(max(0.0, min(90.0, torso)) if mode == "pose" else None),
            neck_forward=(10 + rng.gauss(0, 2) if mode == "pose" else None),
            visibility=v, motion=motion, persons=1))
    return out


STAND = {"cy": 0.55, "torso": 5.0, "bh": 0.7, "bw": 0.25, "cx": 0.5}


def make(label: str, rng: random.Random, mode: str, idx: int, noise: float = 1.0) -> Scenario:
    vis = rng.choice([0.9, 0.8, 0.65] if noise <= 1 else [0.8, 0.6, 0.45])
    t0 = rng.uniform(1.5, 3.0)
    total = t0 + 8.0
    S = dict(STAND)
    present_fn = None
    if label in ("fall_forward", "fall_backward", "fall_sideways"):
        d = rng.uniform(0.35, 0.9)
        end = {"cy": rng.uniform(0.82, 0.92), "torso": rng.uniform(70, 90), "bh": rng.uniform(0.18, 0.3), "bw": rng.uniform(0.55, 0.75), "cx": 0.5 + rng.uniform(-0.1, 0.1)}
        kf = [(0, S), (t0, S), (t0 + d, end), (total, end)]
    elif label == "fall_collapse":
        d = rng.uniform(0.9, 1.3)
        mid = {"cy": 0.72, "torso": 30, "bh": 0.45, "bw": 0.35, "cx": 0.5}
        end = {"cy": rng.uniform(0.84, 0.92), "torso": rng.uniform(65, 90), "bh": 0.25, "bw": 0.6, "cx": 0.5}
        kf = [(0, S), (t0, S), (t0 + d * 0.5, mid), (t0 + d, end), (total, end)]
    elif label == "fall_out_of_frame":
        d = rng.uniform(0.35, 0.7)
        end = {"cy": 0.97, "torso": 80, "bh": 0.15, "bw": 0.5, "cx": 0.5}
        kf = [(0, S), (t0, S), (t0 + d, end), (total, end)]
        present_fn = lambda t, st, t0=t0, d=d: t < t0 + d * 0.8
    elif label == "sit_chair":
        end = {"cy": rng.uniform(0.62, 0.68), "torso": rng.uniform(5, 20), "bh": 0.55, "bw": 0.3, "cx": 0.5}
        kf = [(0, S), (t0, S), (t0 + rng.uniform(0.8, 1.5), end), (total, end)]
    elif label == "sit_floor_controlled":
        end = {"cy": rng.uniform(0.78, 0.85), "torso": rng.uniform(10, 30), "bh": 0.4, "bw": 0.35, "cx": 0.5}
        kf = [(0, S), (t0, S), (t0 + rng.uniform(2.2, 3.5), end), (total, end)]
    elif label == "lie_down_bed":
        mid = {"cy": 0.66, "torso": 15, "bh": 0.5, "bw": 0.3, "cx": 0.5}
        end = {"cy": rng.uniform(0.66, 0.74), "torso": rng.uniform(75, 90), "bh": 0.22, "bw": 0.65, "cx": 0.5}
        kf = [(0, S), (t0, S), (t0 + 1.4, mid), (t0 + rng.uniform(3.0, 4.5), end), (total, end)]
    elif label == "bend_pick_up":
        low = {"cy": 0.62, "torso": rng.uniform(50, 75), "bh": 0.45, "bw": 0.35, "cx": 0.5}
        kf = [(0, S), (t0, S), (t0 + 0.9, low), (t0 + 1.8, low), (t0 + 2.8, S), (total, S)]
    elif label == "jump":
        up = dict(S, cy=0.45)
        down = dict(S, cy=0.6)
        kf = [(0, S), (t0, S), (t0 + 0.25, up), (t0 + 0.5, down), (t0 + 0.8, S), (total, S)]
    elif label == "walk_across":
        a = dict(S, cx=0.15, dx=0.15)
        b = dict(S, cx=0.85, dx=0.15)
        kf = [(0, a), (total, b)]
    elif label == "plop_on_sofa":
        end = {"cy": rng.uniform(0.66, 0.72), "torso": rng.uniform(15, 35), "bh": 0.5, "bw": 0.35, "cx": 0.5}
        kf = [(0, S), (t0, S), (t0 + rng.uniform(0.35, 0.55), end), (total, end)]
    elif label == "stand_still":
        kf = [(0, S), (total, S)]
    elif label == "leave_frame_walking":
        a = dict(S, cx=0.5, dx=0.15)
        b = dict(S, cx=1.1, dx=0.15)
        kf = [(0, a), (t0 + 2, b), (total, b)]
        present_fn = lambda t, st: st["cx"] < 0.98
    elif label == "exercise_squats":
        low = dict(S, cy=0.68, torso=25, bh=0.5)
        kf = [(0, S)]
        t = t0
        while t < total - 1.5:
            kf += [(t, S), (t + 0.7, low), (t + 1.4, S)]
            t += 1.6
        kf.append((total, S))
    else:
        raise ValueError(label)
    samples = _seq(rng, mode, kf, total, vis, present_fn, noise)
    return Scenario(id=f"{label}-{mode}-{idx}", label=label, is_fall=label in POSITIVE, mode=mode, samples=samples)


def generate(n_per: int = 12, seed: int = 7, noise: float = 1.0) -> list[Scenario]:
    rng = random.Random(seed)
    out = []
    for label in POSITIVE + NEGATIVE:
        for i in range(n_per):
            mode = "pose" if i % 3 != 2 else "motion"
            out.append(make(label, rng, mode, i, noise))
    return out


def evaluate_fall_detector(scenarios: list[Scenario] | None = None) -> dict:
    from ..perception.pipeline import FallDetector
    scenarios = scenarios or generate()
    tp = fn = fp = tn = 0
    per_label: dict[str, dict[str, int]] = {}
    failures: list[dict] = []
    for sc in scenarios:
        det = FallDetector()
        events = []
        for s in sc.samples:
            events += det.update(s)
        hit = bool(events)
        pl = per_label.setdefault(sc.label, {"n": 0, "detected": 0})
        pl["n"] += 1
        pl["detected"] += int(hit)
        if sc.is_fall and hit:
            tp += 1
        elif sc.is_fall:
            fn += 1
            failures.append({"id": sc.id, "type": "missed_fall"})
        elif hit:
            fp += 1
            failures.append({"id": sc.id, "type": "false_alarm", "confidence": events[0].confidence})
        else:
            tn += 1
    return {
        "n": len(scenarios),
        "fall_recall": round(tp / max(tp + fn, 1), 4),
        "false_positive_rate": round(fp / max(fp + tn, 1), 4),
        "per_label": per_label,
        "failures": failures,
    }
