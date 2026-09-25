"""Generate docs/ONTOLOGY.md and docs/MODEL_CARDS.md from the code (single source of truth).

    python scripts/gen_docs.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baymax.knowledge.kb import default_kb  # noqa: E402
from baymax.language import lexicon as lx  # noqa: E402
from baymax.language import phrases as ph  # noqa: E402
from baymax.language.text import split_phrases  # noqa: E402
from baymax.models.registry import REGISTRY  # noqa: E402
from baymax.ontology import (CONSENT_DESCRIPTIONS, MEASUREMENT_PLAUSIBLE, MEASUREMENT_UNITS, RED_FLAG_CONCEPTS,  # noqa: E402
                             SUPPORTED_LANGS, BodyRegion, Concept, EventKind, Intent, Provenance, SafetyState, Severity)


def ontology() -> str:
    kb = default_kb()
    L = ["# BAYMAX ontology", "", "_Generated from `baymax/ontology.py`, the lexicons and the knowledge base by "
         "`scripts/gen_docs.py`. Do not edit by hand._", "",
         "## Languages", "", "Baymax understands and answers only in: " + ", ".join(f"`{l}`" for l in SUPPORTED_LANGS) +
         ". Every phrase, knowledge entry and lexicon exists in all of them (enforced by tests).", "",
         "## Provenance", "", "| Value | Meaning |", "|---|---|"]
    meaning = {Provenance.USER: "the user said/typed/entered it", Provenance.KNOWLEDGE: "curated Baymax knowledge or safety protocol text",
               Provenance.MODEL: "output of an identified model, with model id and confidence",
               Provenance.LLM: "free text from the optional LLM (never a health fact; guarded)",
               Provenance.SYSTEM: "non-health dialogue text (greetings, prompts) or Baymax's own records"}
    L += [f"| `{p.value}` | {meaning[p]} |" for p in Provenance]
    L += ["", "## Intents", "", ", ".join(f"`{i.value}`" for i in Intent), "",
          "## Concepts", "", "Red-flag concepts (affirmed, present tense) always drive the safety machine to an alert countdown.", "",
          "| Concept | Red flag | Knowledge entries | Example phrases (en) |", "|---|---|---|---|"]
    for c in Concept:
        ex = ", ".join(split_phrases(lx.CONCEPTS[c]["en"])[:4])
        L.append(f"| `{c.value}` | {'**yes**' if c in RED_FLAG_CONCEPTS else ''} | {', '.join(e.id for e in kb.by_concept(c.value))} | {ex} |")
    L += ["", "Pain in a region derives a specific concept: " +
          ", ".join(f"pain + `{r.value}` → `{c.value}`" for r, c in lx.PAIN_REGION_DERIVED.items()), "",
          "## Body regions", "", ", ".join(f"`{b.value}`" for b in BodyRegion), "",
          "## Measurements", "", "| Type | Canonical unit | Plausible range (outside = rejected, never corrected) |", "|---|---|---|"]
    for m, (lo, hi) in MEASUREMENT_PLAUSIBLE.items():
        L.append(f"| `{m.value}` | {MEASUREMENT_UNITS[m]} | {lo} – {hi} |")
    L += ["", "## Health-event taxonomy", "", ", ".join(f"`{e.value}`" for e in EventKind), "",
          "Severity: " + " < ".join(f"`{s.value}`" for s in Severity), "",
          "## Safety states", "", " → ".join(f"`{s.value}`" for s in SafetyState) + " (see docs/SAFETY.md)", "",
          "## Consent scopes (all default OFF)", "", "| Scope | What it allows |", "|---|---|"]
    L += [f"| `{s.value}` | {d} |" for s, d in CONSENT_DESCRIPTIONS.items()]
    L += ["", "## Deterministic phrases", "", f"{len(ph.P)} phrase keys × {len(SUPPORTED_LANGS)} languages. "
          "Health-guidance phrases (provenance `baymax_knowledge`): " + ", ".join(f"`{k}`" for k in sorted(ph.HEALTH_PHRASES)) + ".", ""]
    return "\n".join(L)


def model_cards() -> str:
    L = ["# Model cards", "", "_Generated from `baymax/models/registry.py` by `scripts/gen_docs.py`._", "",
         "Every component that infers something the user did not directly state is registered here. "
         "Only components marked **may trigger safety** can produce `SafetyInput`s; the LLM cannot.", ""]
    for c in REGISTRY.values():
        L += [f"## {c.name} — `{c.model_id}`", "",
              f"- **Kind:** {c.kind}", f"- **Runs:** {c.runs}", f"- **Input:** {c.input}", f"- **Output:** {c.output}",
              f"- **Confidence:** {c.confidence}", f"- **Safety-critical:** {'yes' if c.safety_critical else 'no'} · "
              f"**may trigger safety:** {'yes' if c.may_trigger_safety else 'no'}", "", "**Failure modes**", ""]
        L += [f"- {f}" for f in c.failure_modes]
        L += ["", "**Mitigations**", ""] + [f"- {m}" for m in c.mitigations] + [""]
    return "\n".join(L)


if __name__ == "__main__":
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "ONTOLOGY.md").write_text(ontology(), encoding="utf-8")
    (ROOT / "docs" / "MODEL_CARDS.md").write_text(model_cards(), encoding="utf-8")
    print("wrote docs/ONTOLOGY.md, docs/MODEL_CARDS.md")
