from __future__ import annotations

import pytest

from rex_learning import LearnerContractError, validate_learner_response


def test_validate_learner_response_accepts_case_alias_and_exact_cardinality() -> None:
    result = validate_learner_response(
        {"responses": [{"case": "a", "answer": "one"}, {"case_id": "b", "answer": "two"}]},
        expected_case_ids=("a", "b"),
    )
    assert result["responses"] == [{"case": "a", "answer": "one"}, {"case": "b", "answer": "two"}]


@pytest.mark.parametrize(
    ("payload", "expected_code"),
    [
        ({"answer": "prose"}, "missing_responses_list"),
        ({"responses": ["not an object"]}, "response_item_not_object"),
        ({"responses": [{"answer": "missing case"}]}, "response_case_missing"),
        ({"responses": [{"case": "a"}, {"case": "a"}]}, "duplicate_case_id"),
        ({"responses": [{"case": "unexpected"}]}, "case_cardinality_mismatch"),
    ],
)
def test_validate_learner_response_classifies_contract_failures(payload: dict, expected_code: str) -> None:
    with pytest.raises(LearnerContractError) as error:
        validate_learner_response(payload, expected_case_ids=("a", "b"))
    assert error.value.code == expected_code
    assert error.value.diagnostic()["code"] == expected_code


def test_validate_learner_response_never_invents_missing_answers() -> None:
    result = validate_learner_response({"responses": [{"case": "a"}]})
    assert result["responses"] == [{"case": "a", "answer": None}]


@pytest.mark.parametrize("key", ("artifact", "candidate_artifact", "candidate_artifact_or_answer", "candidate_answer", "judgment", "not_applicable", "response"))
def test_validate_learner_response_normalizes_equivalent_artifact_payload_keys(key: str) -> None:
    result = validate_learner_response({"responses": [{"case": "a", key: "compiled artifact"}]})
    assert result["responses"] == [{"case": "a", "answer": "compiled artifact"}]


def test_validate_learner_response_rejects_conflicting_payload_keys() -> None:
    with pytest.raises(LearnerContractError, match="one answer or artifact payload") as error:
        validate_learner_response({"responses": [{"case": "a", "answer": "one", "artifact": "two"}]})
    assert error.value.code == "multiple_response_payloads"
