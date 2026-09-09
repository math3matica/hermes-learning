from __future__ import annotations

import pytest

from rex_learning.provider_boundary import normalize_provider_abstraction
from rex_learning import DeepStudyEngine, LearningStore


def answer(**overrides):
    value = {
        "source_purpose": "teach a rule",
        "learner_purpose": "build a capability",
        "objective_relevance": "supports the goal",
        "relevance_strength": "core",
        "expected_learning_contribution": "correct reasoning",
        "retained_abstraction": "retain the rule",
        "generalizable_invariants": ["the invariant"],
        "exact_details_to_preserve": ["the exact condition"],
        "contingent_details": ["the example context"],
        "applicability": ["when the condition holds"],
        "non_applicability": ["when it does not hold"],
        "prerequisite_or_dependency_role": "supports later work",
        "optionality": "required",
        "currentness": "enduring",
        "currentness_confidence": "high",
        "external_verification_need": "none",
        "learning_action": "normal-study",
        "abstraction_level": "concept",
        "instructional_value": "mastery_useful",
        "mastery_required": True,
        "required_evidence": ["explain"],
        "source_quality": "adequate",
        "confidence": 0.8,
        "reasoning_trace": "The source supports the judgment.",
        "uncertainty": "none",
        "trace": {"provider": "qwen"},
    }
    value.update(overrides)
    return value


def test_numeric_confidence_is_preserved_as_v2_confidence_object():
    result = normalize_provider_abstraction(answer())
    assert result.canonical["confidence"] == {"label": None, "score": 0.8, "raw": 0.8}
    assert result.provenance["normalizer_version"]


def test_qualitative_confidence_is_preserved_without_fake_precision():
    result = normalize_provider_abstraction(answer(confidence="high"))
    assert result.canonical["confidence"] == {"label": "high", "score": None, "raw": "high"}
    assert "qualitative-confidence-preserved" in result.provenance["operations"]


def test_numeric_string_confidence_is_normalized_but_raw_is_retained():
    result = normalize_provider_abstraction(answer(confidence="0.8"))
    assert result.canonical["confidence"] == {"label": None, "score": 0.8, "raw": "0.8"}


@pytest.mark.parametrize("value", ["unknown-confidence", "1.2", [], {"score": 0.8}])
def test_malformed_or_unknown_confidence_is_rejected(value):
    with pytest.raises(ValueError):
        normalize_provider_abstraction(answer(confidence=value))


def test_scalar_lists_and_boolean_are_representation_only_normalizations():
    raw = answer(
        generalizable_invariants="one invariant",
        contingent_details="one contingent detail",
        exact_details_to_preserve="one exact detail",
        applicability="when applicable",
        non_applicability="when not applicable",
        required_evidence="explain",
        mastery_required="true",
    )
    result = normalize_provider_abstraction(raw)
    assert result.normalized["generalizable_invariants"] == ["one invariant"]
    assert result.canonical["required_evidence"] == ["explain"]
    assert result.canonical["mastery_required"] is True
    assert result.raw == raw


def test_enum_capitalization_is_normalized_but_unknown_semantics_are_preserved():
    result = normalize_provider_abstraction(answer(relevance_strength="CORE", learning_action="Review the SDK documentation."))
    assert result.canonical["relevance_strength"] == "core"
    assert result.canonical["learning_action"] == "Review the SDK documentation."
    assert result.canonical["normalization_metadata"]["unknown_enum_values"]["learning_action"] == "Review the SDK documentation."


def test_uncertain_mastery_judgment_is_preserved_not_coerced():
    result = normalize_provider_abstraction(answer(mastery_required="familiarity"))
    assert result.canonical["mastery_required"] == "familiarity"
    assert "uncertain-boolean-preserved:mastery_required" in result.provenance["operations"]


def test_missing_substantive_field_is_rejected_without_scaffold_completion():
    raw = answer()
    del raw["retained_abstraction"]
    with pytest.raises(ValueError, match="retained_abstraction"):
        normalize_provider_abstraction(raw)


def test_raw_answer_is_not_mutated_and_unknown_action_is_not_repaired():
    raw = answer(learning_action="Implement a simple activity and test it.")
    before = dict(raw)
    result = normalize_provider_abstraction(raw)
    assert raw == before
    assert result.canonical["learning_action"] == before["learning_action"]
    assert result.canonical["learning_action"] != "practice"


def test_trace_and_complete_semantic_fields_survive_normalization():
    result = normalize_provider_abstraction(answer())
    for field in ("applicability", "non_applicability", "retained_abstraction", "contingent_details", "required_evidence", "reasoning_trace", "uncertainty"):
        assert field in result.canonical


def test_v2_provider_layers_persist_immutably(tmp_path):
    result = normalize_provider_abstraction(answer(confidence="high"))
    engine = DeepStudyEngine(LearningStore(tmp_path / "learning"))
    provenance = {"operation_id": "op-1", **result.provenance}
    record = engine.record_provider_instructional_abstraction(
        intent_id="intent-v2", source_id="source", unit_id="unit", source_hash="source-hash",
        raw_provider_answer=result.raw, normalized_provider_answer=result.normalized,
        canonical_abstraction=result.canonical, provenance=provenance,
    )
    assert record["schema"] == "rex-learning-instructional-abstraction-v2-record"
    assert record["raw_provider_answer"]["confidence"] == "high"
    assert record["canonical_abstraction"]["confidence"]["label"] == "high"
    assert engine.store.list("instructional_abstractions")[0]["provenance"]["semantic_repair_performed"] is False