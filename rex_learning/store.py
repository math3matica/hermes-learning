from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping


def resolve_learning_root(host: Any | None = None, *, environ: Mapping[str, str] | None = None) -> Path:
    """Resolve profile-local state without coupling Rex to a personal path.

    A host-owned ``state_dir`` (or ``learning_state_dir``) wins, followed by
    the explicit Rex override and finally Hermes' standard home environment.
    The returned directory is dedicated to Rex Learning state.
    """
    env = environ if environ is not None else os.environ
    if host is not None:
        for name in ("learning_state_dir", "state_dir"):
            value = getattr(host, name, None)
            if value:
                root = Path(value).expanduser().resolve()
                return root if root.name == "rex-learning" else root / "rex-learning"
    explicit = env.get("HERMES_REX_LEARNING_HOME", "").strip()
    if explicit:
        return Path(explicit).expanduser().resolve()
    hermes_home = env.get("HERMES_HOME", "").strip()
    return (Path(hermes_home).expanduser().resolve() if hermes_home else Path.home() / ".hermes") / "rex-learning"


class LearningStore:
    """Atomic, inspectable filesystem store; JSON is the source of truth."""

    COLLECTIONS = ("curricula", "curriculum_intents", "sources", "notes", "knowledge", "capability_proposals", "capability_discovery_attempts", "candidate_skills", "skill_integration_plans", "skill_conflict_reviews", "skills", "skill_versions", "tests", "attempts", "evaluations", "baseline_preflights", "applications", "process_runs", "process_attempts", "process_evaluations", "composition_runs", "composition_attempts", "composition_evaluations", "events", "cognitive_operations", "study_runs", "study_units", "study_exposures", "study_retrievals", "study_rereads", "study_relations", "study_conflicts", "study_practice_tasks", "study_practice_results", "study_tests", "study_methods", "study_semantic_evaluations", "instructional_abstractions", "staged_abstraction_runs", "acquisition_runs", "learning_jobs")

    def __init__(self, root: Path):
        self.root = Path(root).expanduser().resolve()
        for collection in self.COLLECTIONS:
            (self.root / collection).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe(identifier: str) -> str:
        if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", identifier):
            raise ValueError("invalid learning identifier")
        return identifier

    def path(self, collection: str, identifier: str) -> Path:
        if collection not in self.COLLECTIONS:
            raise ValueError("unknown learning collection")
        return self.root / collection / f"{self._safe(identifier)}.json"

    def write(self, collection: str, identifier: str, value: dict[str, Any]) -> dict[str, Any]:
        path = self.path(collection, identifier)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            Path(temporary_name).replace(path)
        finally:
            temporary = Path(temporary_name)
            if temporary.exists():
                temporary.unlink()
        return value

    def read(self, collection: str, identifier: str) -> dict[str, Any]:
        path = self.path(collection, identifier)
        if not path.exists():
            raise KeyError(f"unknown learning record: {collection}/{identifier}")
        return json.loads(path.read_text(encoding="utf-8"))

    def list(self, collection: str) -> list[dict[str, Any]]:
        if collection not in self.COLLECTIONS:
            raise ValueError("unknown learning collection")
        records = []
        for path in sorted((self.root / collection).glob("*.json")):
            records.append(json.loads(path.read_text(encoding="utf-8")))
        return records

    def append_event(self, event: dict[str, Any]) -> None:
        identifier = f"event-{event['id']}"
        self.write("events", identifier, event)
