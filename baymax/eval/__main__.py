"""Run the full BAYMAX evaluation and write reports/eval_latest.{json,md}.

    python -m baymax.eval                 # everything, exit code 1 if any gate fails
    python -m baymax.eval --heldout-seed 1234
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .criteria import evaluate_gates
from .harness import DATASETS, ROOT, load_cases, load_guard_cases, run_case, run_guard_eval, run_llm_e2e, summarize
from .perception_synth import evaluate_fall_detector, generate as gen_scenarios
from .synth import generate as gen_synthetic
from .schema import Case


def run(heldout_seed: int = 7777, verbose: bool = True) -> dict:
    t0 = time.time()
    sets: dict[str, list[Case]] = {
        "curated": load_cases(DATASETS / "curated" / "baymax_curated.jsonl"),
        "adversarial": load_cases(DATASETS / "adversarial" / "adversarial.jsonl"),
        "synthetic": load_cases(DATASETS / "synthetic" / "synthetic.jsonl"),
        "synthetic_heldout": [Case(**c) for c in gen_synthetic(seed=heldout_seed)],
    }
    for c in sets["synthetic_heldout"]:
        c.id = c.id.replace("syn-", "heldout-")
    text: dict[str, dict] = {}
    all_results = []
    for name, cases in sets.items():
        if verbose:
            print(f"  text/{name}: {len(cases)} cases", flush=True)
        res = [run_case(c) for c in cases]
        all_results += res
        text[name] = summarize(res)
    text["all"] = summarize(all_results)
    guard_cases = load_guard_cases(DATASETS / "adversarial" / "llm_attack_outputs.jsonl")
    if verbose:
        print(f"  llm guard: {len(guard_cases)} outputs", flush=True)
    guard = run_guard_eval(guard_cases)
    e2e = run_llm_e2e(guard_cases)
    if verbose:
        print("  perception: dev / held-out / stress", flush=True)
    perception = {
        "dev": evaluate_fall_detector(gen_scenarios(12, seed=7)),
        "heldout": evaluate_fall_detector(gen_scenarios(20, seed=2026)),
        "stress": evaluate_fall_detector(gen_scenarios(20, seed=99, noise=2.0)),
    }
    report = {
        "generated_at": time.time(),
        "duration_s": round(time.time() - t0, 1),
        "heldout_seed": heldout_seed,
        "text": text,
        "llm_guard": guard,
        "llm_e2e": e2e,
        "perception": perception,
    }
    report["gates"] = evaluate_gates(report)
    report["passed"] = all(g["passed"] for g in report["gates"])
    report["summary"] = {"cases": sum(len(v) for v in sets.values()) + len(guard_cases),
                         "text_cases": len(all_results), "guard_outputs": len(guard_cases),
                         "perception_scenarios": sum(p["n"] for p in perception.values())}
    return report


def to_markdown(r: dict) -> str:
    L = [f"# BAYMAX evaluation report", "",
         f"Generated {time.strftime('%Y-%m-%d %H:%M', time.localtime(r['generated_at']))} in {r['duration_s']} s — "
         f"**{'ALL GATES PASS' if r['passed'] else 'GATES FAILING'}**", "",
         "## Release gates", "", "| Gate | Value | Threshold | Result |", "|---|---|---|---|"]
    for g in r["gates"]:
        tag = " (safety)" if g["safety"] else ""
        L.append(f"| {g['name']}{tag} | {g['value']} | {g['op']} {g['threshold']} | {'PASS' if g['passed'] else '**FAIL**'} |")
    L += ["", "## Text pipeline", "", "| Set | Cases | Pass | Emergency recall | False escalation | Intent acc. | Language acc. |",
          "|---|---|---|---|---|---|---|"]
    for name, s in r["text"].items():
        L.append(f"| {name} | {s['cases']} | {s['pass_rate']:.1%} | {s['emergency_recall']:.1%} ({s['emergency_cases']}) | "
                 f"{s['false_escalation_rate']:.1%} ({s['no_escalation_cases']}) | {s['intent']['accuracy']:.1%} | {s['language_accuracy']:.1%} |")
    L += ["", "### Adversarial categories", "", "| Category | Cases | Passed |", "|---|---|---|"]
    for cat, v in r["text"]["adversarial"]["by_category"].items():
        L.append(f"| {cat} | {v['n']} | {v['passed']} |")
    g, e = r["llm_guard"], r["llm_e2e"]
    L += ["", "## LLM output guard", "",
          f"- attack outputs: {g['attack_cases']} — block rate **{g['block_rate']:.1%}**",
          f"- benign outputs: {g['benign_cases']} — false-block rate {g['false_block_rate']:.1%}",
          f"- end-to-end with a malicious LLM in the engine: {e['attack_outputs_reaching_user']} attack outputs reached the user; "
          f"{e['llm_safety_influence']} cases where the LLM changed safety behaviour", ""]
    L += ["## Perception (synthetic body trajectories)", "", "| Set | Scenarios | Fall recall | False-positive rate |", "|---|---|---|---|"]
    for name, p in r["perception"].items():
        L.append(f"| {name} | {p['n']} | {p['fall_recall']:.1%} | {p['false_positive_rate']:.1%} |")
    fails = [f for s in ("curated", "adversarial", "synthetic", "synthetic_heldout") for f in r["text"][s]["failures"]]
    if fails:
        L += ["", "## Failing text cases", ""]
        for f in fails[:80]:
            L.append(f"- `{f['id']}` “{f['text']}” → {'; '.join(f['failures'])}")
    L += ["", "_Synthetic perception scenarios are not real-world footage; see docs/EVALUATION.md for limitations._", ""]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="BAYMAX evaluation harness")
    ap.add_argument("--heldout-seed", type=int, default=7777)
    ap.add_argument("--out", type=Path, default=ROOT / "reports")
    args = ap.parse_args(argv)
    print("BAYMAX evaluation")
    r = run(args.heldout_seed)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "eval_latest.json").write_text(json.dumps(r, ensure_ascii=False, indent=1, default=str))
    (args.out / "eval_latest.md").write_text(to_markdown(r))
    width = max(len(g["name"]) for g in r["gates"])
    for g in r["gates"]:
        print(f"  {'PASS' if g['passed'] else 'FAIL'}  {g['name']:<{width}}  {g['value']} {g['op']} {g['threshold']}")
    print(f"\n{'ALL GATES PASS' if r['passed'] else 'GATES FAILING'} — report: {args.out / 'eval_latest.md'}")
    return 0 if r["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
