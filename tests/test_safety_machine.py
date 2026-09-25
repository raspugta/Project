"""Deterministic safety state machine: exhaustive and property-based checks."""
import itertools
import random
import typing

import pytest

from baymax.contracts import SafetyContext, SafetyInput, SafetyInputKind
from baymax.ontology import SAFETY_STATE_RANK, SafetyState as S, Severity
from baymax.safety.machine import SafetyMachine, SafetyPolicy, step

KINDS = list(typing.get_args(SafetyInputKind))
POLICY = SafetyPolicy()
ESCALATING_INPUTS = {"user_request_help_now", "user_confirm", "timer_expired"}


def ctx_for(state: S, now: float = 0.0, **kw) -> SafetyContext:
    timers = {S.CHECK_IN: "checkin", S.ESCALATION_COUNTDOWN: "countdown", S.ASSISTING: "assist", S.ESCALATED: "reminder"}
    name = timers.get(state)
    return SafetyContext(state=state, since=now, timer_name=name, timer_deadline=(now + 30) if name else None,
                         awaiting={S.CHECK_IN: "ok_check", S.ASSISTING: "confirm_escalation",
                                   S.ESCALATION_COUNTDOWN: "confirm_escalation"}.get(state), **kw)


def inp(kind: str, **detail) -> SafetyInput:
    timer = detail.pop("timer", None)
    d = dict(detail)
    if kind == "timer_expired":
        d["timer"] = timer
    return SafetyInput(kind=kind, source="timer" if kind == "timer_expired" else "nlu", detail=d, confidence=0.9)


def test_no_llm_input_kind_exists():
    """Structural guarantee: nothing produced by a language model can enter the machine."""
    assert not any("llm" in k for k in KINDS)
    sources = typing.get_args(SafetyInput.model_fields["source"].annotation)
    assert set(sources) == {"nlu", "perception", "ui", "timer"}


@pytest.mark.parametrize("state,kind", list(itertools.product(list(S), KINDS)))
def test_every_state_input_pair_is_well_defined(state, kind):
    c = ctx_for(state)
    t = step(c, inp(kind, timer=c.timer_name, pain_score=5), POLICY, now=1.0)
    if t is None:
        return
    assert t.before == state
    assert t.context.state == t.after
    assert t.rule.startswith("R-")
    # states that wait for the user always carry a deadline
    if t.after in (S.CHECK_IN, S.ESCALATION_COUNTDOWN, S.ESCALATED):
        assert t.context.timer_name and t.context.timer_deadline and t.context.timer_deadline > 1.0
    # escalation to ESCALATED only through explicit confirmation, the help button or an expired countdown
    if t.after == S.ESCALATED and state != S.ESCALATED:
        assert kind in ESCALATING_INPUTS
        if kind == "timer_expired":
            assert state == S.ESCALATION_COUNTDOWN
        assert any(a.kind == "notify_contacts" for a in t.actions)


@pytest.mark.parametrize("state", [s for s in S if s != S.ESCALATED])
def test_critical_input_always_starts_countdown(state):
    c = ctx_for(state)
    t = step(c, inp("critical_utterance", concepts=["chest_pain"]), POLICY, now=1.0)
    assert t is not None
    assert SAFETY_STATE_RANK[t.after] >= SAFETY_STATE_RANK[S.ESCALATION_COUNTDOWN]
    assert any(a.kind == "say" and a.phrase_key in ("call_emergency", "emergency_other", "crisis_support") for a in t.actions)


def test_stale_timer_is_ignored():
    c = ctx_for(S.ESCALATION_COUNTDOWN)
    assert step(c, inp("timer_expired", timer="checkin"), POLICY, 1.0) is None


def test_fall_detected_below_threshold_is_ignored():
    t = step(ctx_for(S.MONITORING), SafetyInput(kind="fall_detected", source="perception", confidence=0.3), POLICY, 1.0)
    assert t is None


def test_full_fall_no_response_path():
    m = SafetyMachine(POLICY)
    t = m.feed(SafetyInput(kind="fall_detected", source="perception", confidence=0.8), 0)
    assert t.after == S.CHECK_IN
    t = m.feed(m.due_timer(POLICY.checkin_timeout_s + 0.1), POLICY.checkin_timeout_s + 0.1)
    assert t.after == S.ESCALATION_COUNTDOWN
    t2 = POLICY.checkin_timeout_s + POLICY.countdown_s + 0.2
    t = m.feed(m.due_timer(t2), t2)
    assert t.after == S.ESCALATED and t.rule == "R-X-TIMEOUT"
    assert any(a.kind == "notify_contacts" for a in t.actions)


def test_informational_red_flag_question_times_out_quietly():
    m = SafetyMachine(POLICY)
    m.feed(inp("possible_emergency", red_flag_uncertain=True), 0)
    t = m.feed(m.due_timer(100), 100)
    assert t.after == S.MONITORING and t.rule == "R-C-TIMEOUT-INFO"


def test_cancelling_a_critical_countdown_still_advises_care():
    m = SafetyMachine(POLICY)
    m.feed(inp("critical_utterance", concepts=["chest_pain"]), 0)
    t = m.feed(inp("user_cancel"), 1)
    assert t.after == S.MONITORING
    assert [a.phrase_key for a in t.actions if a.kind == "say"] == ["cancelled", "cancelled_but_advise"]


def test_random_walks_preserve_invariants():
    rng = random.Random(1234)
    for _ in range(3000):
        m = SafetyMachine(POLICY)
        now = 0.0
        for _ in range(12):
            now += rng.choice([0.5, 5, 40])
            kind = rng.choice(KINDS)
            if kind == "timer_expired":
                i = m.due_timer(now) or inp("timer_expired", timer="nonexistent")
            else:
                i = inp(kind, pain_score=rng.choice([None, 2, 9]), cannot_get_up=rng.random() < 0.3)
            before = m.ctx.state
            t = m.feed(i, now)
            if t is None:
                assert m.ctx.state == before
                continue
            if t.after == S.ESCALATED and before != S.ESCALATED:
                assert i.kind in ESCALATING_INPUTS
            if m.ctx.state in (S.CHECK_IN, S.ESCALATION_COUNTDOWN, S.ESCALATED):
                assert m.ctx.timer_deadline is not None
        # from anywhere, the user can always get back to monitoring
        for k in ("user_ok", "resolve"):
            m.feed(inp(k), now + 1)
        if m.ctx.state == S.ESCALATION_COUNTDOWN:
            m.feed(inp("user_cancel"), now + 2)
        assert m.ctx.state == S.MONITORING, m.ctx
