"""Automated safety regression: every dataset case + every release gate.

Each curated/adversarial case is its own test so a regression names the exact
utterance that broke. The final test runs the full evaluation and asserts
every release gate.
"""
import pytest

from baymax.eval.harness import DATASETS, load_cases, run_case

CURATED = load_cases(DATASETS / "curated" / "baymax_curated.jsonl")
ADVERSARIAL = load_cases(DATASETS / "adversarial" / "adversarial.jsonl")
SYNTHETIC = load_cases(DATASETS / "synthetic" / "synthetic.jsonl")


def _check(case):
    r = run_case(case)
    assert r.passed, f"{case.text!r}: {r.failures + r.invariant_violations} (intent={r.intent}, state={r.state})"


@pytest.mark.parametrize("case", ADVERSARIAL, ids=lambda c: c.id)
def test_adversarial(case):
    _check(case)


@pytest.mark.parametrize("case", CURATED, ids=lambda c: c.id)
def test_curated(case):
    _check(case)


@pytest.mark.parametrize("case", SYNTHETIC, ids=lambda c: c.id)
def test_synthetic(case):
    _check(case)


def test_all_release_gates():
    from baymax.eval.__main__ import run
    report = run(heldout_seed=9090, verbose=False)
    failing = [g for g in report["gates"] if not g["passed"]]
    assert not failing, failing
