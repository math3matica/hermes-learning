"""JSON-stdin adapter for Hermes-owned executable contract evaluation."""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from typing import Any

from .evaluator import ContractArtifactEvaluator


def evaluate_envelope(envelope: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate an executable-validator envelope using the production contract evaluator."""
    test = envelope["test"]
    attempt = envelope["attempt"]
    bound_test = dict(test)
    bound_test["evaluator_id"] = "hermes-contract-validator"
    bound_test["evaluator_type"] = "contract_artifact_evaluator"
    bound_test["artifact_schema"] = test.get("artifact_schema") or test.get("evaluation_contract")
    result = ContractArtifactEvaluator(
        evaluator_id="hermes-contract-validator",
        provider="hermes",
        session_id="contract-validator-process",
    ).evaluate(bound_test, attempt)
    return {"case_results": result["case_results"]}


def main() -> None:
    try:
        envelope = json.load(sys.stdin)
        if not isinstance(envelope, Mapping):
            raise ValueError("validator envelope must be an object")
        json.dump(evaluate_envelope(envelope), sys.stdout)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        json.dump({"case_results": []}, sys.stdout)


if __name__ == "__main__":
    main()