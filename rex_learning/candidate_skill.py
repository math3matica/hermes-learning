"""Untrusted candidate-skill package and promotion-boundary primitives.

Qwen may propose learning. It may not silently turn a proposal into durable
skill state. This module is provider-neutral and keeps candidate evidence,
provenance, practice, review, and promotion decision separate.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from .engine import LearningError

CANDIDATE_STATES = frozenset({"EXTRACTED", "PRACTICING", "AWAITING_REVIEW", "REVISE", "REJECTED", "APPROVED_WITH_LIMITS", "APPROVED"})
REVIEW_DECISIONS = frozenset({"PROMOTE", "PROMOTE_WITH_NARROWER_SCOPE", "REVISE_AND_RETEST", "NEEDS_SOURCE_VERIFICATION", "REJECT_UNSUPPORTED", "DUPLICATE_EXISTING_SKILL", "DEFER"})


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def _nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LearningError(f"{name} is required")
    return value.strip()


def build_candidate_skill(*, case: Mapping[str, Any], qwen_surface: Mapping[str, Any], provenance: Mapping[str, Any], practice: list[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Build an inspectable, non-promoted candidate package from a learner surface."""
    case_id = _nonempty(case.get("id"), "case.id")
    if not isinstance(qwen_surface, Mapping):
        raise LearningError("qwen candidate surface must be an object")
    if not isinstance(provenance, Mapping) or not _nonempty(provenance.get("provider"), "provenance.provider") or not _nonempty(provenance.get("session_id"), "provenance.session_id"):
        raise LearningError("candidate provenance requires provider and session_id")
    source = case.get("source")
    intent = case.get("intent")
    if not isinstance(source, Mapping) or not isinstance(intent, Mapping):
        raise LearningError("candidate requires source and curriculum intent")
    evidence = [
        {"kind": "source", "refs": list(source.get("source_refs", [])), "title": source.get("title"), "text": source.get("text", "")},
        {"kind": "curriculum_intent", "key": intent.get("key"), "target_capability": intent.get("target_capability")},
        {"kind": "qwen_surface", "surface_schema": qwen_surface.get("surface_schema", "unknown"), "surface": dict(qwen_surface)},
    ]
    package = {
        "schema": "rex-learning-candidate-skill-package-v1",
        "candidate_id": f"candidate-{case_id}-{_digest({'case': case_id, 'surface': qwen_surface})[:16]}",
        "state": "EXTRACTED",
        "trusted": False,
        "promoted": False,
        "case_id": case_id,
        "source": dict(source),
        "curriculum_intent": dict(intent),
        "proposed_skill": {
            "kind": qwen_surface.get("kind", "procedure"),
            "claim": qwen_surface.get("transferable_abstractions") or qwen_surface.get("source_invariant") or qwen_surface.get("capability") or intent.get("target_capability"),
            "scope": qwen_surface.get("applicability"),
            "operational_procedure": qwen_surface.get("operational_procedure") or qwen_surface.get("procedure"),
            "preconditions": qwen_surface.get("preconditions"),
            "limitations": qwen_surface.get("limitations") or qwen_surface.get("non_applicability"),
            "learning_action": qwen_surface.get("learning_action"),
            "evidence_path": qwen_surface.get("evidence_path"),
            "uncertainties": qwen_surface.get("uncertainties"),
        },
        "qwen_interpretation": dict(qwen_surface),
        "provenance": dict(provenance),
        "evidence": evidence,
        "practice": [dict(item) for item in (practice or [])],
        "review": None,
        "promotion": None,
    }
    return package


def validate_review_decision(decision: Mapping[str, Any], *, candidate_id: str) -> dict[str, Any]:
    if not isinstance(decision, Mapping):
        raise LearningError("review decision must be an object")
    value = dict(decision)
    if value.get("decision") not in REVIEW_DECISIONS:
        raise LearningError("review decision is invalid")
    if value.get("candidate_id") != candidate_id:
        raise LearningError("review decision candidate identity mismatch")
    if not isinstance(value.get("rationale"), str) or not value["rationale"].strip():
        raise LearningError("review rationale is required")
    for field in ("accepted_claims", "rejected_claims", "required_changes", "source_refs", "scope_limits"):
        if not isinstance(value.get(field, []), list) or not all(isinstance(item, str) and item.strip() for item in value.get(field, [])):
            raise LearningError(f"review {field} must be a list of strings")
    if "reviewed_surface" in value and not isinstance(value["reviewed_surface"], Mapping):
        raise LearningError("reviewed_surface must be an object")
    if value["decision"] in {"PROMOTE", "PROMOTE_WITH_NARROWER_SCOPE"}:
        surface = value.get("reviewed_surface")
        if not isinstance(surface, Mapping):
            raise LearningError("promotable review requires reviewed_surface")
        claim = surface.get("claim") or surface.get("source_invariant") or surface.get("transferable_abstractions")
        procedure = surface.get("operational_procedure") or surface.get("procedure")
        if not claim or not procedure:
            raise LearningError("reviewed_surface requires claim and operational_procedure")
    return value


def attach_review(package: Mapping[str, Any], decision: Mapping[str, Any], *, reviewer: Mapping[str, Any]) -> dict[str, Any]:
    candidate_id = _nonempty(package.get("candidate_id"), "candidate_id")
    checked = validate_review_decision(decision, candidate_id=candidate_id)
    if not isinstance(reviewer, Mapping) or not _nonempty(reviewer.get("provider"), "reviewer.provider") or not _nonempty(reviewer.get("session_id"), "reviewer.session_id"):
        raise LearningError("reviewer provenance requires provider and session_id")
    updated = dict(package)
    updated["review"] = {"decision": checked, "reviewer": dict(reviewer), "independence": "independent", "contamination": {"status": "clean"}}
    if isinstance(checked.get("reviewed_surface"), Mapping):
        updated["reviewed_surface"] = dict(checked["reviewed_surface"])
    updated["state"] = {"PROMOTE": "APPROVED", "PROMOTE_WITH_NARROWER_SCOPE": "APPROVED_WITH_LIMITS", "REVISE_AND_RETEST": "REVISE", "REJECT_UNSUPPORTED": "REJECTED", "DUPLICATE_EXISTING_SKILL": "REJECTED", "NEEDS_SOURCE_VERIFICATION": "AWAITING_REVIEW", "DEFER": "AWAITING_REVIEW"}[checked["decision"]]
    updated["trusted"] = checked["decision"] in {"PROMOTE", "PROMOTE_WITH_NARROWER_SCOPE"}
    updated["promoted"] = False
    return updated


def promotion_boundary(package: Mapping[str, Any], *, behavioral_evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Promotion remains impossible without independent behavioral evidence."""
    review = package.get("review")
    if not isinstance(review, Mapping) or review.get("decision", {}).get("decision") not in {"PROMOTE", "PROMOTE_WITH_NARROWER_SCOPE"}:
        return {"schema": "rex-learning-candidate-promotion-v1", "promoted": False, "reason": "review_not_promotable"}
    required = ("source_free", "novel_transfer", "negative_case", "reproducible", "delayed_retention")
    missing = next((key for key in required if behavioral_evidence.get(key) is not True), None)
    if missing:
        return {"schema": "rex-learning-candidate-promotion-v1", "promoted": False, "reason": f"{missing}_missing"}
    if behavioral_evidence.get("contamination_status") != "clean" or behavioral_evidence.get("evaluator_independence") != "independent":
        return {"schema": "rex-learning-candidate-promotion-v1", "promoted": False, "reason": "behavioral_evidence_not_independent_clean"}
    return {"schema": "rex-learning-candidate-promotion-v1", "promoted": True, "reason": "review_and_behavioral_evidence_satisfied"}
