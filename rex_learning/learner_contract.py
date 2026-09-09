from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


class LearnerContractError(ValueError):
    """A learner response violated the provider-neutral wire contract."""

    def __init__(self, message: str, *, code: str, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = dict(details or {})

    def diagnostic(self) -> dict[str, Any]:
        return {"code": self.code, **self.details}


def validate_learner_response(
    parsed: Any,
    *,
    expected_case_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validate and normalize a decoded learner response.

    This function performs only wire-contract validation. It does not judge the
    answer content and it never supplies missing answers or case IDs.
    """
    if not isinstance(parsed, Mapping):
        raise LearnerContractError(
            "learner response must be a JSON object",
            code="response_not_object",
            details={"actual_type": type(parsed).__name__},
        )
    responses = parsed.get("responses")
    if not isinstance(responses, list):
        raise LearnerContractError(
            "learner response must contain a responses list",
            code="missing_responses_list",
            details={"actual_type": type(responses).__name__},
        )

    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(responses):
        if not isinstance(item, Mapping):
            raise LearnerContractError(
                "learner responses must be objects",
                code="response_item_not_object",
                details={"index": index, "actual_type": type(item).__name__},
            )
        case = item.get("case", item.get("case_id"))
        if not isinstance(case, str) or not case.strip():
            raise LearnerContractError(
                "learner response case is required",
                code="response_case_missing",
                details={"index": index},
            )
        case = case.strip()
        if case in seen:
            raise LearnerContractError(
                "learner responses contain duplicate case IDs",
                code="duplicate_case_id",
                details={"case": case, "index": index},
            )
        seen.add(case)
        payload_keys = (
            "answer", "artifact", "candidate_artifact", "candidate_artifact_or_answer",
            "candidate_answer", "judgment", "not_applicable", "response",
        )
        present = [key for key in payload_keys if key in item]
        if len(present) > 1:
            raise LearnerContractError(
                "learner response must contain one answer or artifact payload",
                code="multiple_response_payloads",
                details={"case": case, "keys": present},
            )
        payload = item.get(present[0]) if present else None
        normalized.append({"case": case, "answer": payload})

    if expected_case_ids is not None:
        expected = [case.strip() for case in expected_case_ids if isinstance(case, str) and case.strip()]
        expected_set = set(expected)
        if len(expected) != len(expected_set):
            raise LearnerContractError(
                "expected learner case IDs must be unique and non-empty",
                code="expected_case_ids_invalid",
            )
        actual_set = {item["case"] for item in normalized}
        unknown = sorted(actual_set - expected_set)
        missing = sorted(expected_set - actual_set)
        if unknown or missing or len(normalized) != len(expected):
            raise LearnerContractError(
                "learner responses must match expected case IDs exactly once",
                code="case_cardinality_mismatch",
                details={"expected": expected, "actual": [item["case"] for item in normalized], "missing": missing, "unknown": unknown},
            )

    result = {**dict(parsed), "responses": normalized}
    return result
