from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Sequence


class CardinalityError(ValueError):
    """Raised when an evaluator arm does not produce its frozen cardinality."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def freeze_hidden_artifact(
    path: Path,
    *,
    payload: Any,
    provider: str,
    designer_model: str,
    session_id: str,
    source_hash: str,
) -> dict[str, Any]:
    """Persist and verify an evaluator-only artifact before learner exposure.

    The digest covers the hidden payload and its metadata, excluding the digest
    field itself. The returned record is the exact durable envelope written to
    disk; callers should record its digest in the exposure event.
    """
    if path.exists():
        raise FileExistsError(f"hidden artifact already exists: {path}")
    if not all(isinstance(value, str) and value.strip() for value in (provider, designer_model, session_id, source_hash)):
        raise ValueError("hidden artifact provenance is incomplete")
    metadata = {
        "created_at_unix": time.time(),
        "provider": provider,
        "designer_model": designer_model,
        "session_id": session_id,
        "source_hash": source_hash,
        "source_visibility": "evaluator_only",
        "learner_visibility": False,
    }
    digest = hashlib.sha256(_canonical({"payload": payload, "metadata": metadata})).hexdigest()
    record = {"schema": "rex-learning-hidden-evaluator-artifact-v2", "metadata": metadata, "payload": payload, "artifact_sha256": digest}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return verify_hidden_artifact(path)


def verify_hidden_artifact(path: Path) -> dict[str, Any]:
    """Fail closed unless a durable hidden artifact is complete and untampered."""
    if not path.is_file():
        raise ValueError(f"hidden artifact is missing: {path}")
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"hidden artifact cannot be read: {path}") from exc
    if not isinstance(record, dict) or record.get("schema") != "rex-learning-hidden-evaluator-artifact-v2":
        raise ValueError("hidden artifact schema is invalid")
    metadata = record.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("source_visibility") != "evaluator_only" or metadata.get("learner_visibility") is not False:
        raise ValueError("hidden artifact visibility metadata is invalid")
    stored = record.get("artifact_sha256")
    actual = hashlib.sha256(_canonical({"payload": record.get("payload"), "metadata": metadata})).hexdigest()
    if not isinstance(stored, str) or stored != actual:
        raise ValueError("hidden artifact hash mismatch")
    return record


def require_exact_cardinality(label: str, values: Sequence[Any], expected: int) -> list[Any]:
    """Require exactly ``expected`` unique result identifiers/records."""
    if not isinstance(expected, int) or expected < 0:
        raise ValueError("expected cardinality must be a non-negative integer")
    actual = list(values)
    if len(actual) != expected:
        raise CardinalityError(f"{label}: expected {expected} results, received {len(actual)}")
    try:
        unique = len(set(actual))
    except TypeError:
        unique = len({json.dumps(item, ensure_ascii=False, sort_keys=True, default=str) for item in actual})
    if unique != expected:
        raise CardinalityError(f"{label}: duplicate results violate exact cardinality {expected}")
    return actual
