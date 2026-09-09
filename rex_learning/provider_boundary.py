from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

from .engine import LearningError
from .instructional_abstraction import ACTIONS, CURRENTNESS, LEVELS, OPTIONALITY, RELEVANCE, VALUES


NORMALIZER_VERSION = "provider-boundary-normalizer-v1"
CANONICAL_SCHEMA = "rex-learning-instructional-abstraction-v2"

_REQUIRED_FIELDS = (
    "source_purpose",
    "learner_purpose",
    "objective_relevance",
    "relevance_strength",
    "expected_learning_contribution",
    "retained_abstraction",
    "generalizable_invariants",
    "contingent_details",
    "applicability",
    "non_applicability",
    "prerequisite_or_dependency_role",
    "optionality",
    "currentness",
    "external_verification_need",
    "learning_action",
    "abstraction_level",
    "instructional_value",
    "mastery_required",
    "required_evidence",
    "source_quality",
    "confidence",
)
_LIST_FIELDS = (
    "generalizable_invariants",
    "exact_details_to_preserve",
    "contingent_details",
    "applicability",
    "non_applicability",
    "required_evidence",
)
_ENUM_FIELDS = {
    "relevance_strength": RELEVANCE,
    "optionality": OPTIONALITY,
    "currentness": CURRENTNESS,
    "learning_action": ACTIONS,
    "abstraction_level": LEVELS,
    "instructional_value": VALUES,
}


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def _nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LearningError(f"{field} is required as a nonempty string")
    return value.strip()


def _normalize_list(value: Any, field: str) -> tuple[list[str], str | None]:
    if isinstance(value, str):
        if not value.strip():
            raise LearningError(f"{field} cannot be empty")
        return [value.strip()], f"scalar-to-singleton-list:{field}"
    if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
        raise LearningError(f"{field} must be a string or list of nonempty strings")
    return [item.strip() for item in value], None


def _normalize_bool(value: Any, field: str) -> tuple[Any, str | None]:
    if isinstance(value, bool):
        return value, None
    if isinstance(value, str) and value.strip().casefold() in {"true", "false"}:
        return value.strip().casefold() == "true", f"boolean-string-normalized:{field}"
    if isinstance(value, str) and value.strip():
        return value, f"uncertain-boolean-preserved:{field}"
    raise LearningError(f"{field} must be a boolean or nonempty provider judgment string")


def _normalize_confidence(value: Any) -> tuple[dict[str, Any], str]:
    if isinstance(value, bool):
        raise LearningError("confidence must not be boolean")
    if isinstance(value, (int, float)):
        score = float(value)
        if not 0 <= score <= 1:
            raise LearningError("numeric confidence must be between 0 and 1")
        return {"label": None, "score": score, "raw": value}, "numeric-confidence-preserved"
    if isinstance(value, str):
        stripped = value.strip()
        try:
            score = float(stripped)
        except ValueError:
            if stripped.casefold() not in {"low", "medium", "high"}:
                raise LearningError("unknown qualitative confidence label")
            return {"label": stripped, "score": None, "raw": value}, "qualitative-confidence-preserved"
        if not 0 <= score <= 1:
            raise LearningError("numeric confidence must be between 0 and 1")
        return {"label": None, "score": score, "raw": value}, "numeric-string-confidence-normalized"
    raise LearningError("confidence must be numeric, numeric string, or known qualitative label")


@dataclass(frozen=True)
class ProviderAbstractionNormalization:
    raw: dict[str, Any]
    normalized: dict[str, Any]
    canonical: dict[str, Any]
    provenance: dict[str, Any]


def normalize_provider_abstraction(raw: Mapping[str, Any]) -> ProviderAbstractionNormalization:
    """Normalize provider representation without supplying or repairing semantic judgments."""
    if not isinstance(raw, Mapping):
        raise LearningError("provider abstraction must be an object")
    original = copy.deepcopy(dict(raw))
    missing = [field for field in _REQUIRED_FIELDS if field not in original]
    if missing:
        raise LearningError(f"provider abstraction is missing substantive fields: {', '.join(missing)}")

    normalized = copy.deepcopy(original)
    operations: list[str] = []
    changed_fields: list[str] = []
    unknown_enum_values: dict[str, str] = {}

    for field in _LIST_FIELDS:
        value, operation = _normalize_list(original[field], field)
        normalized[field] = value
        if operation:
            operations.append(operation)
            changed_fields.append(field)

    normalized["mastery_required"], operation = _normalize_bool(original["mastery_required"], "mastery_required")
    if operation:
        operations.append(operation)
        changed_fields.append("mastery_required")

    normalized["confidence"], confidence_operation = _normalize_confidence(original["confidence"])
    operations.append(confidence_operation)
    if normalized["confidence"] != original["confidence"]:
        changed_fields.append("confidence")

    for field, allowed in _ENUM_FIELDS.items():
        value = _nonempty_string(original[field], field)
        folded = value.casefold()
        if folded in allowed and value != folded:
            normalized[field] = folded
            operations.append(f"enum-capitalization-normalized:{field}")
            changed_fields.append(field)
        elif folded not in allowed:
            normalized[field] = value
            unknown_enum_values[field] = value
            operations.append(f"unknown-enum-preserved:{field}")

    for field in _REQUIRED_FIELDS:
        if field in _LIST_FIELDS or field in {"mastery_required", "confidence"} or field in _ENUM_FIELDS:
            continue
        normalized[field] = _nonempty_string(original[field], field)

    if "exact_details_to_preserve" not in normalized:
        normalized["exact_details_to_preserve"] = []

    trace = normalized.get("trace", {})
    if not isinstance(trace, dict):
        trace = {"provider_trace": copy.deepcopy(trace)}
        operations.append("trace-object-wrapped")
        changed_fields.append("trace")
    for field in ("exact_details_to_preserve", "currentness_confidence", "reasoning_trace", "uncertainty"):
        if field in normalized:
            trace[field] = copy.deepcopy(normalized[field])
    normalized["trace"] = trace

    metadata = {
        "normalizer_version": NORMALIZER_VERSION,
        "unknown_enum_values": unknown_enum_values,
        "changed_fields": sorted(set(changed_fields)),
        "operations": operations,
    }
    normalized["normalization_metadata"] = metadata
    canonical = {"schema": CANONICAL_SCHEMA, **copy.deepcopy(normalized)}
    provenance = {
        "normalizer_version": NORMALIZER_VERSION,
        "operations": operations,
        "changed_fields": sorted(set(changed_fields)),
        "unknown_enum_values": unknown_enum_values,
        "raw_hash": _digest(original),
        "normalized_hash": _digest(normalized),
        "canonical_hash": _digest(canonical),
        "semantic_repair_performed": False,
        "deterministic_scaffold_used": False,
    }
    return ProviderAbstractionNormalization(raw=original, normalized=normalized, canonical=canonical, provenance=provenance)
