from __future__ import annotations

import pytest

from rex_learning.semantic_evaluator import (
    JUDGMENT_LABELS,
    build_feedback,
    evaluate_promotion,
    validate_judgment,
)


def _judgment(**overrides):
    value = {
        "label": "correct",
        "dimensions": {
            "semantic_correctness": "pass",
            "procedure_execution": "pass",
            "applicability_judgment": "pass",
            "uncertainty_calibration": "pass",
        },
        "diagnoses": [],
        "missing_propositions": [],
        "unsupported_claims": [],
        "feedback": "The response is supported and appropriately qualified.",
        "confidence": 0.9,
    }
    value.update(overrides)
    return value


def test_semantic_judgment_accepts_valid_paraphrase_and_exposes_actionable_feedback() -> None:
    judgment = validate_judgment(_judgment(label="valid_alternative_formulation", diagnoses=["valid_alternative_formulation"]))

    assert judgment["label"] in JUDGMENT_LABELS
    assert build_feedback(judgment)["action"] == "advance"


def test_semantic_judgment_rejects_unknown_label_and_bad_confidence() -> None:
    with pytest.raises(ValueError, match="label"):
        validate_judgment(_judgment(label="fluent"))
    with pytest.raises(ValueError, match="confidence"):
        validate_judgment(_judgment(confidence=2))
    with pytest.raises(ValueError, match="diagnosis"):
        validate_judgment(_judgment(diagnoses=["The answer misses the boundary condition."]))


def test_essential_omission_routes_to_targeted_remediation() -> None:
    judgment = validate_judgment(_judgment(
        label="correct_but_incomplete",
        diagnoses=["essential_omission"],
        missing_propositions=["spacing is used across separated sessions"],
    ))

    feedback = build_feedback(judgment)
    assert feedback["action"] == "targeted_retrieval"
    assert feedback["missing_propositions"]


def test_promotion_requires_behavioral_independent_evidence() -> None:
    evidence = {
        "semantic_pass": True,
        "source_free": True,
        "novel_transfer": True,
        "negative_case": True,
        "reproducible": True,
        "contamination_status": "clean",
        "baseline_status": "absent",
        "evaluator_independence": "independent",
        "learner_behavior_evidence": True,
    }

    decision = evaluate_promotion(evidence)

    assert decision["promoted"] is False
    assert decision["reason"] == "baseline_missing"