"""Static integrity of the ontology, lexicons, phrase table, knowledge base and model registry."""
import re
from pathlib import Path

import pytest

from baymax.knowledge.kb import default_kb
from baymax.language import lexicon as lx
from baymax.language import phrases as ph
from baymax.language.text import normalize, split_phrases
from baymax.models.registry import REGISTRY
from baymax.ontology import SUPPORTED_LANGS, BodyRegion, Concept, ConsentScope, CONSENT_DESCRIPTIONS

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("concept", list(Concept))
def test_every_concept_has_every_language(concept):
    assert set(lx.CONCEPTS[concept]) == set(SUPPORTED_LANGS)
    for lang in SUPPORTED_LANGS:
        assert split_phrases(lx.CONCEPTS[concept][lang]), f"{concept} has no {lang} phrases"


def test_every_body_region_has_every_language():
    for region in BodyRegion:
        assert set(lx.BODY[region]) == set(SUPPORTED_LANGS), region


TABLES = {name: getattr(lx, name) for name in (
    "EMERGENCY_STRONG", "EMERGENCY_EXACT", "HELP_OPEN", "POSSIBLE_EMERGENCY", "PAIN_EXCLAMATIONS", "USER_OK",
    "CONFIRM", "CANCEL", "GREETING", "GOODBYE", "QUESTION", "NEGATORS", "HYPOTHETICAL", "PAST", "OTHER_SUBJECT",
    "INJECTION", "FILLERS", "RECALL", "MEDICATION_TAKEN", "DIAGNOSIS_REQUEST", "DOSAGE_REQUEST", "PRIVATE_REQUEST")}


@pytest.mark.parametrize("name", sorted(TABLES))
def test_intent_tables_cover_all_languages(name):
    assert set(TABLES[name]) == set(SUPPORTED_LANGS), name


def test_lexicon_phrases_are_already_normalised():
    """Phrases are matched against normalised text, so an accented phrase would silently never match."""
    bad = []
    for table in [*lx.CONCEPTS.values(), *lx.BODY.values(), *TABLES.values(), *lx.PRIVACY.values()]:
        for lang, phrases in table.items():
            for p in split_phrases(phrases):
                if normalize(p.rstrip("*")) != p.rstrip("*"):
                    bad.append((lang, p))
    assert not bad, bad[:10]


def test_phrase_table_complete_and_consistent():
    assert ph.validate() == []


def test_every_concept_has_a_localised_name():
    for c in Concept:
        assert f"c:{c.value}" in ph.NAMES


def test_knowledge_base_complete():
    kb = default_kb()
    assert kb.validate() == []
    for c in ("chest_pain", "breathing_difficulty", "stroke_signs", "severe_bleeding", "seizure", "self_harm",
              "anaphylaxis_signs", "poisoning", "unconscious"):
        assert kb.by_concept(c), f"red flag {c} has no knowledge entry"


def test_knowledge_base_contains_no_doses():
    kb = default_kb()
    for e in kb.entries.values():
        for lang, text in e.text.items():
            assert not re.search(r"\d+\s*(mg|ml|mcg|tablets?|pills?)\b", normalize(text)), (e.id, lang)


def test_consent_scopes_documented():
    assert set(CONSENT_DESCRIPTIONS) == set(ConsentScope)


def test_every_referenced_model_id_is_registered():
    """Any model_id literal in the code base must have a model card."""
    ids = set()
    for py in (ROOT / "baymax").rglob("*.py"):
        ids |= set(re.findall(r'model_id(?:: str)?\s*=\s*"([a-z0-9_]+)"', py.read_text()))
        ids |= set(re.findall(r'MODEL_ID\s*=\s*"([a-z0-9_]+)"', py.read_text()))
    assert ids, "no model ids found"
    missing = {i for i in ids if i not in REGISTRY}
    assert not missing, missing


def test_model_cards_are_complete():
    for card in REGISTRY.values():
        assert card.input and card.output and card.confidence
        assert card.failure_modes, card.model_id
        if card.may_trigger_safety:
            assert card.mitigations, card.model_id
    assert REGISTRY["llm"].may_trigger_safety is False
