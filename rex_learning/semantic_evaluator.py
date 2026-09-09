from __future__ import annotations

from typing import Any

JUDGMENT_LABELS = frozenset(
    {
        "correct",
        "correct_but_incomplete",
        "materially_incorrect",
        "misconception",
        "essential_omission",
        "valid_alternative_formulation",
        "unsupported_embellishment",
        "overgeneralization",
        "underqualification",
        "inappropriate_applicability",
        "uncertain_ambiguous",
        "evaluator_cannot_determine",
        "legitimate_not_applicable",
    }
)

_DIMENSION_VALUES = frozenset({"pass", "fail", "uncertain", "not_applicable"})
_DIAGNOSES = JUDGMENT_LABELS - {"correct", "materially_incorrect", "uncertain_ambiguous"}


def validate_judgment(judgment: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize one frozen semantic evaluator result.

    This deliberately validates the evaluator's shape, not its truth. Truth is
    established by source-grounded evidence and provider provenance recorded by
    the qualification runner.
    """
    if not isinstance(judgment, dict):
        raise ValueError("judgment must be an object")
    label = judgment.get("label")
    if label not in JUDGMENT_LABELS:
        raise ValueError("judgment label is invalid")
    dimensions = judgment.get("dimensions")
    if not isinstance(dimensions, dict) or not dimensions:
        raise ValueError("judgment dimensions are required")
    if any(value not in _DIMENSION_VALUES for value in dimensions.values()):
        raise ValueError("judgment dimension value is invalid")
    confidence = judgment.get("confidence")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not 0 <= confidence <= 1:
        raise ValueError("judgment confidence must be between 0 and 1")
    for field in ("diagnoses", "missing_propositions", "unsupported_claims"):
        if not isinstance(judgment.get(field, []), list) or not all(isinstance(item, str) and item.strip() for item in judgment.get(field, [])):
            raise ValueError(f"judgment {field} must be a list of nonempty strings")
    diagnoses = list(dict.fromkeys(judgment.get("diagnoses", [])))
    if any(item not in _DIAGNOSES for item in diagnoses):
        raise ValueError("judgment diagnosis is invalid")
    feedback = judgment.get("feedback")
    if not isinstance(feedback, str) or not feedback.strip():
        raise ValueError("judgment feedback is required")
    normalized = dict(judgment)
    normalized["label"] = label
    normalized["diagnoses"] = diagnoses
    normalized["confidence"] = float(confidence)
    normalized["missing_propositions"] = list(judgment.get("missing_propositions", []))
    normalized["unsupported_claims"] = list(judgment.get("unsupported_claims", []))
    return normalized


def build_feedback(judgment: dict[str, Any]) -> dict[str, Any]:
    """Turn a validated judgment into a deterministic learning action."""
    value = validate_judgment(judgment)
    diagnoses = set(value["diagnoses"])
    label = value["label"]
    if label in {"correct", "valid_alternative_formulation", "legitimate_not_applicable"}:
        action = "advance"
    elif "essential_omission" in diagnoses or label == "correct_but_incomplete":
        action = "targeted_retrieval"
    elif "misconception" in diagnoses or label in {"materially_incorrect", "inappropriate_applicability"}:
        action = "contrastive_remediation"
    elif "overgeneralization" in diagnoses or "underqualification" in diagnoses:
        action = "retrieve_boundary_or_exception"
    elif label in {"uncertain_ambiguous", "evaluator_cannot_determine"}:
        action = "human_or_second_evaluator_review"
    else:
        action = "targeted_retrieval"
    return {
        "schema": "rex-learning-semantic-feedback-v1",
        "action": action,
        "diagnoses": value["diagnoses"],
        "missing_propositions": value["missing_propositions"],
        "unsupported_claims": value["unsupported_claims"],
        "feedback": value["feedback"],
    }


def evaluate_promotion(evidence: dict[str, Any]) -> dict[str, Any]:
    """Apply the frozen conservative capability-promotion boundary."""
    required = (
        ("semantic_pass", "semantic_failure"),
        ("source_free", "source_exposure"),
        ("novel_transfer", "transfer_missing"),
        ("negative_case", "negative_case_missing"),
        ("reproducible", "reproducibility_missing"),
        ("learner_behavior_evidence", "learner_evidence_missing"),
        ("baseline_status", "baseline_missing"),
        ("delayed_retention", "delayed_retention_missing"),
    )
    for key, reason in required:
        if evidence.get(key) is not True:
            return {"schema": "rex-learning-capability-promotion-v1", "promoted": False, "reason": reason}
    if evidence.get("contamination_status") != "clean":
        return {"schema": "rex-learning-capability-promotion-v1", "promoted": False, "reason": "contamination"}
    if evidence.get("evaluator_independence") != "independent":
        return {"schema": "rex-learning-capability-promotion-v1", "promoted": False, "reason": "evaluator_not_independent"}
    return {"schema": "rex-learning-capability-promotion-v1", "promoted": True, "reason": "frozen_criteria_satisfied"}


def provenance_record(*, learner: dict[str, Any], evaluator: dict[str, Any], judgment: dict[str, Any], contamination: dict[str, Any], remediation: dict[str, Any] | None = None) -> dict[str, Any]:
    """Create an inspectable record without mixing evaluator claims with learner evidence."""
    return {
        "schema": "rex-learning-semantic-evaluation-v1",
        "learner": dict(learner),
        "evaluator": dict(evaluator),
        "judgment": validate_judgment(judgment),
        "feedback": build_feedback(judgment),
        "contamination": dict(contamination),
        "remediation": dict(remediation or {}),
        "capability_evidence_boundary": "learner_behavior_only",
    }
