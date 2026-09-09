from __future__ import annotations

"""Lightweight integrated instructional abstraction boundary.

The provider performs one coherent semantic judgment. This module only normalizes
representation, validates provenance/integrity, detects structural risk, and
applies a narrowly scoped provider critique. It does not decide educational
meaning by deterministic taxonomy.
"""

import copy
from dataclasses import dataclass
from typing import Any, Mapping

from .instructional_abstraction import CurriculumIntent, SourceContext


INTEGRATED_OPERATION = "instructional_abstraction_integrated"
CRITIC_OPERATION = "critique_instructional_claims"
STATUSES = {"ADMISSIBLE", "QUARANTINED", "UNKNOWN", "EXTERNAL_VERIFICATION_REQUIRED", "REJECTED"}
SUPPORT_TYPES = {"source_established", "intent_established", "bounded_inference", "unknown", "external_verification_required"}


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _sequence(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple)):
        return list(value)
    return [copy.deepcopy(value)] if value is not None else []


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _refs(value: Any) -> list[str]:
    return [item.strip() for item in _list(value) if isinstance(item, str) and item.strip()]


def _object(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


_ALIASES = {
    "source_invariant": ("source_invariant", "invariants", "invariant_core", "invariant_content", "source_invariant"),
    "source_facts": ("source_facts", "source_summary"),
    "transferable_abstractions": ("transferable_abstractions", "transferable_abstraction", "transferable_mechanisms"),
    "contingent_or_example_specific_details": ("contingent_or_example_specific_details", "contingent_details"),
    "goal_relevance": ("goal_relevance", "relevance"),
    "dependency_or_optionality": ("dependency_or_optionality", "dependency", "optionality"),
    "learning_action": ("learning_action",),
    "evidence_path": ("evidence_path",),
}


def _adapt_object_field(raw: Mapping[str, Any], field: str, aliases: tuple[str, ...], issues: list[dict[str, Any]]) -> tuple[Any, str | None]:
    present = next(((name, raw[name]) for name in aliases if name in raw), (None, None))
    name, value = present
    if name is None:
        return {}, None
    if isinstance(value, Mapping):
        adapted = copy.deepcopy(dict(value))
        if name != field:
            issues.append({"field": field, "code": "ALIAS_MAPPED", "from": name})
        return adapted, name
    if isinstance(value, list):
        if len(value) == 1 and isinstance(value[0], Mapping):
            issues.append({"field": field, "code": "LIST_SINGLETON_UNWRAPPED", "original_shape": "list"})
            return copy.deepcopy(dict(value[0])), name
        if field == "evidence_path" and all(isinstance(item, (str, int, float)) for item in value):
            evidence = copy.deepcopy(value)
            issues.append({"field": field, "code": "EVIDENCE_LIST_WRAPPED", "original_shape": "list"})
            return {
                "evidence": evidence,
                "support_refs": [item for item in evidence if isinstance(item, str)],
                "raw_value": evidence,
                "representation_status": "LIST_WRAPPED",
            }, name
        if all(isinstance(item, (str, int, float, bool)) for item in value):
            issues.append({"field": field, "code": "LIST_PRESERVED", "original_shape": "list"})
            return copy.deepcopy(value), name
        if not all(isinstance(item, Mapping) for item in value):
            issues.append({"field": field, "code": "SCHEMA_INCOMPATIBLE", "original_shape": "list", "raw_value": copy.deepcopy(value)})
            return {"raw_value": copy.deepcopy(value), "representation_status": "SCHEMA_INCOMPATIBLE"}, name
        issues.append({"field": field, "code": "LIST_PRESERVED", "original_shape": "list"})
        return copy.deepcopy(value), name
    if isinstance(value, (str, int, float, bool)):
        issues.append({"field": field, "code": "SCALAR_WRAPPED", "original_shape": type(value).__name__})
        return {"value": copy.deepcopy(value), "raw_value": copy.deepcopy(value), "representation_status": "SCALAR_WRAPPED"}, name
    issues.append({"field": field, "code": "SCHEMA_INCOMPATIBLE", "original_shape": type(value).__name__, "raw_value": copy.deepcopy(value)})
    return {"raw_value": copy.deepcopy(value), "representation_status": "SCHEMA_INCOMPATIBLE"}, name


def _faithful_surface(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Return a representation-only copy for provider semantic evaluation."""
    return copy.deepcopy(dict(raw))


def normalize_integrated_answer(answer: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt known provider shapes without silently discarding semantic content."""
    raw = copy.deepcopy(dict(answer))
    result = copy.deepcopy(raw)
    representation_issues: list[dict[str, Any]] = []
    field_sources: dict[str, str] = {}
    result["source_facts"] = _sequence(raw.get("source_facts", raw.get("source_summary")))
    result["exact_details"] = _sequence(raw.get("exact_details", raw.get("exact_details_to_preserve")))
    result["transferable_abstractions"] = _sequence(raw.get("transferable_abstractions", raw.get("transferable_abstraction", raw.get("transferable_mechanisms"))))
    result["contingent_or_example_specific_details"] = _sequence(raw.get("contingent_or_example_specific_details", raw.get("contingent_details")))
    result["uncertainties"] = _sequence(raw.get("uncertainties", raw.get("uncertainty")))
    result["bounded_inferences"] = _list(raw.get("bounded_inferences"))
    result["source_purpose"] = raw.get("source_purpose")
    for field, aliases in _ALIASES.items():
        value, source_field = _adapt_object_field(raw, field, aliases, representation_issues)
        result[field] = value
        if source_field:
            field_sources[field] = source_field
    result["claims"] = _normalize_claims(answer.get("claims"))
    if "currentness_applicable" not in result:
        currentness = _object(answer.get("currentness"))
        result["currentness_applicable"] = currentness.get("applicable")
        result["currentness_status"] = currentness.get("status")
    result["currentness_status"] = answer.get("currentness_status", result.get("currentness_status"))
    result["representation_fidelity"] = {
        "status": "SCHEMA_INCOMPATIBLE" if any(item["code"] == "SCHEMA_INCOMPATIBLE" for item in representation_issues) else ("PRESERVED_WITH_ADAPTER" if representation_issues else "PRESERVED"),
        "adapter": "rex-learning-integrated-semantic-adapter-v1",
        "raw_fields": raw,
        "faithful_surface": _faithful_surface(raw),
        "field_sources": field_sources,
        "incompatibilities": [item for item in representation_issues if item["code"] == "SCHEMA_INCOMPATIBLE"],
        "operations": representation_issues,
    }
    return result


def _normalize_claims(value: Any) -> list[dict[str, Any]]:
    claims = []
    for index, raw in enumerate(_list(value), 1):
        if not isinstance(raw, Mapping):
            continue
        claim = dict(raw)
        claim.setdefault("id", f"claim-{index}")
        claim["support_refs"] = _refs(claim.get("support_refs"))
        claim.setdefault("support_type", "unknown")
        claim.setdefault("status", "PROPOSED")
        claims.append(claim)
    return claims


@dataclass(frozen=True)
class IntegrityResult:
    canonical: dict[str, Any]
    issues: list[dict[str, Any]]
    quarantined_claim_ids: list[str]


def validate_integrated_answer(answer: Mapping[str, Any], source: SourceContext, intent: CurriculumIntent) -> IntegrityResult:
    """Validate references and internal representation without semantic replacement."""
    canonical = normalize_integrated_answer(answer)
    source_refs = set(source.source_refs)
    intent_refs = {intent.key, f"intent:{intent.key}"}
    allowed_refs = source_refs | intent_refs
    issues: list[dict[str, Any]] = []
    quarantined: list[str] = []
    seen: set[str] = set()
    for claim in canonical["claims"]:
        claim_id = str(claim.get("id"))
        if claim_id in seen:
            issues.append({"code": "DUPLICATE_CLAIM_ID", "claim_id": claim_id})
            claim["status"] = "QUARANTINED"
            quarantined.append(claim_id)
            continue
        seen.add(claim_id)
        support_type = claim.get("support_type")
        refs = set(claim.get("support_refs", []))
        bad_refs = sorted(refs - allowed_refs)
        missing_support = not refs and support_type not in {"unknown", "external_verification_required"}
        wrong_kind = (support_type == "source_established" and not refs.intersection(source_refs)) or (support_type == "intent_established" and not refs.intersection(intent_refs))
        if bad_refs:
            issues.append({"code": "UNKNOWN_SUPPORT_REF", "claim_id": claim_id, "refs": bad_refs})
        if missing_support:
            issues.append({"code": "MISSING_SUPPORT", "claim_id": claim_id})
        if wrong_kind:
            issues.append({"code": "SUPPORT_KIND_MISMATCH", "claim_id": claim_id, "support_type": support_type})
        if bad_refs or missing_support or wrong_kind:
            claim["status"] = "QUARANTINED"
            quarantined.append(claim_id)
        elif claim.get("status") in {"PROPOSED", ""}:
            claim["status"] = "ADMISSIBLE"
    _validate_ref_object(canonical.get("source_invariant", {}), "source_invariant", source_refs, issues)
    _validate_ref_object(canonical.get("goal_relevance", {}), "goal_relevance", intent_refs, issues, allow_mixed=True)
    _validate_ref_object(canonical.get("learning_action", {}), "learning_action", allowed_refs, issues, allow_mixed=True)
    _validate_ref_object(canonical.get("evidence_path", {}), "evidence_path", allowed_refs, issues, allow_mixed=True)
    applicable = canonical.get("currentness_applicable")
    status = canonical.get("currentness_status")
    if applicable is False and status not in {None, "NOT_APPLICABLE"}:
        issues.append({"code": "CURRENTNESS_INCOHERENT", "reason": "non-applicable currentness cannot carry a substantive status"})
    if applicable is True and status in {None, ""}:
        issues.append({"code": "CURRENTNESS_INCOHERENT", "reason": "applicable currentness requires a status"})
    canonical["integrity"] = {"issues": issues, "quarantined_claim_ids": quarantined, "validator": "rex-learning-integrated-integrity-v1"}
    return IntegrityResult(canonical, issues, quarantined)


def _validate_ref_object(value: Mapping[str, Any], field: str, allowed: set[str], issues: list[dict[str, Any]], *, allow_mixed: bool = False) -> None:
    refs = set(_refs(value.get("support_refs", value.get("basis_refs"))))
    if refs - allowed:
        issues.append({"code": "UNKNOWN_SUPPORT_REF", "field": field, "refs": sorted(refs - allowed)})
    if not refs and value:
        issues.append({"code": "MISSING_SUPPORT", "field": field})


def assess_risks(canonical: Mapping[str, Any], source: SourceContext, intent: CurriculumIntent) -> list[dict[str, Any]]:
    """Detect structural reasons to spend one additional cognitive call."""
    risks: list[dict[str, Any]] = []
    issues = canonical.get("integrity", {}).get("issues", [])
    for issue in issues:
        if issue.get("code") in {"UNKNOWN_SUPPORT_REF", "SUPPORT_KIND_MISMATCH", "MISSING_SUPPORT"}:
            risks.append({"code": "ACTION_SUPPORT_WEAK" if issue.get("field") == "learning_action" else "CONFLICTING_CLAIM_SUPPORT", "basis": issue})
    applicable = canonical.get("currentness_applicable")
    status = str(canonical.get("currentness_status") or "").upper()
    if applicable is True and status in {"UNKNOWN", "UNKNOWN_RELEVANT", ""}:
        risks.append({"code": "CURRENTNESS_RELEVANT_BUT_UNRESOLVED", "basis": "provider marked currentness relevant without an established status"})
    if _refs(canonical.get("learning_action", {}).get("basis_refs")) == []:
        risks.append({"code": "ACTION_SUPPORT_WEAK", "basis": "learning action has no basis refs"})
    if not _list(canonical.get("evidence_path", {}).get("evidence")):
        risks.append({"code": "EVIDENCE_PATH_SUPPORT_WEAK", "basis": "evidence path is empty"})
    text = source.text.casefold()
    exact_markers = ("must", "before", "after", "exactly", "two-byte", "network byte order", "threshold", "exception", "type")
    if any(marker in text for marker in exact_markers) and not _list(canonical.get("exact_details")):
        risks.append({"code": "EXACT_DETAIL_AT_RISK", "basis": "source contains explicit precision markers but answer has no exact details"})
    claims_text = " ".join(str(item.get("claim", "")) for item in _list(canonical.get("claims"))).casefold()
    if any(word in claims_text for word in ("always", "never", "all systems", "guarantees", "universal")):
        risks.append({"code": "UNSUPPORTED_BROAD_GENERALIZATION", "basis": "claim contains broadening language requiring targeted review"})
    if source.source_quality in {"poor", "damaged", "incomplete", "unknown"}:
        risks.append({"code": "SOURCE_EXTRACTION_DEFECT", "basis": source.source_quality})
    return _dedupe_risks(risks)


def _dedupe_risks(risks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    seen = set()
    for risk in risks:
        key = risk["code"]
        if key not in seen:
            seen.add(key)
            out.append(risk)
    return out


def apply_targeted_critic(answer: Mapping[str, Any], critic: Mapping[str, Any]) -> dict[str, Any]:
    """Apply only named critic revisions; unrelated integrated claims survive."""
    result = normalize_integrated_answer(answer)
    by_id = {claim["id"]: claim for claim in result["claims"]}
    for revision in _list(critic.get("target_claims")):
        if not isinstance(revision, Mapping):
            continue
        claim = by_id.get(revision.get("id"))
        if claim is None:
            continue
        decision = str(revision.get("decision", "")).casefold()
        if decision == "keep":
            claim["status"] = "ADMISSIBLE"
        elif decision == "narrow":
            if revision.get("revised_claim"):
                claim["claim"] = revision["revised_claim"]
            claim["status"] = "BOUNDED_INFERENCE"
        elif decision == "reject":
            claim["status"] = "REJECTED"
        elif decision == "mark_unknown":
            claim["status"] = "UNKNOWN"
        elif decision == "external_verification":
            claim["status"] = "EXTERNAL_VERIFICATION_REQUIRED"
        claim["critic_reasoning_basis"] = revision.get("reasoning_basis", "")
        claim["critic_support_refs"] = _refs(revision.get("support_refs"))
    if isinstance(critic.get("affected_learning_action"), Mapping):
        result["learning_action"] = dict(critic["affected_learning_action"])
    if isinstance(critic.get("affected_evidence_path"), Mapping):
        result["evidence_path"] = dict(critic["affected_evidence_path"])
    result["critic_applied"] = True
    return result
