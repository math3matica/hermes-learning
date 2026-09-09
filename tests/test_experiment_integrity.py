from __future__ import annotations

import json
from pathlib import Path

import pytest

from rex_learning.experiment_integrity import (
    CardinalityError,
    freeze_hidden_artifact,
    require_exact_cardinality,
    verify_hidden_artifact,
)


def test_hidden_artifact_is_durable_hashed_and_learner_invisible(tmp_path: Path) -> None:
    path = tmp_path / "hidden-reference-v2.json"
    record = freeze_hidden_artifact(
        path,
        payload={"cases": [{"id": "case-1", "expected": "boundary"}]},
        provider="openai-codex",
        designer_model="gpt-5.6-luna",
        session_id="designer-session",
        source_hash="source-hash",
    )

    assert path.exists()
    assert record["artifact_sha256"]
    assert record["metadata"]["source_visibility"] == "evaluator_only"
    assert record["metadata"]["learner_visibility"] is False
    assert verify_hidden_artifact(path) == record


def test_hidden_artifact_verification_rejects_tampering(tmp_path: Path) -> None:
    path = tmp_path / "hidden-reference-v2.json"
    freeze_hidden_artifact(
        path,
        payload={"cases": [{"id": "case-1", "expected": "boundary"}]},
        provider="openai-codex",
        designer_model="gpt-5.6-luna",
        session_id="designer-session",
        source_hash="source-hash",
    )
    value = json.loads(path.read_text())
    value["payload"]["cases"][0]["expected"] = "tampered"
    path.write_text(json.dumps(value, indent=2) + "\n")

    with pytest.raises(ValueError, match="hash"):
        verify_hidden_artifact(path)


def test_cardinality_requires_exact_case_judgments() -> None:
    assert require_exact_cardinality("transfer", ["a", "b"], 2) == ["a", "b"]
    with pytest.raises(CardinalityError, match="transfer"):
        require_exact_cardinality("transfer", ["a"], 2)
    with pytest.raises(CardinalityError, match="transfer"):
        require_exact_cardinality("transfer", ["a", "a"], 2)
