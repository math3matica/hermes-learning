from __future__ import annotations

from pathlib import Path
from importlib import import_module

import pytest

from rex_learning.entrypoint import LearningRuntime, run_learning
from rex_learning.engine import LearningEngine, LearningError
from rex_learning.store import LearningStore


class _Learner:
    provider = "test"
    session_id = "session"

    def answer(self, *, task: str, material: str = "", revision: str = "") -> dict:
        raise AssertionError("the entrypoint contract test must not run a provider")


def test_run_learning_adapts_generic_runtime_to_autonomous_study(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "guide.md"
    source.write_text("# Guide\n\nInstructional material.\n", encoding="utf-8")
    runtime = LearningRuntime(
        learner=_Learner(),
        design_provider=lambda request: {},
        runtime_policy={"max_attempts": 1},
    )
    captured: dict[str, object] = {}

    def fake_study(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"schema": "rex-learning-autonomous-study-v1"}

    monkeypatch.setattr("rex_learning.entrypoint.autonomous_study", fake_study)

    result = run_learning(
        engine=object(),
        source_path=source,
        objective="Learn the transferable procedure",
        runtime=runtime,
    )

    assert result == {
        "schema": "rex-learning-run-v1",
        "runtime_policy": {"max_attempts": 1},
        "study": {"schema": "rex-learning-autonomous-study-v1"},
    }
    assert captured == {
        "engine": captured["engine"],
        "source_path": source,
        "learner": runtime.learner,
        "educational_objective": "Learn the transferable procedure",
        "design_provider": runtime.design_provider,
        "qualification_dependencies": None,
        "limit": 2,
        "baseline_scores": None,
        "successful_acquisition_limit": 2,
        "discovery_packet_chars": 12000,
        "runtime_policy": {"max_attempts": 1},
        "resume_run_id": None,
        }


def test_run_learning_requires_explicit_runtime_dependencies(tmp_path: Path) -> None:
    with pytest.raises(LearningError, match="runtime must be a LearningRuntime"):
        run_learning(
            engine=object(),
            source_path=tmp_path / "guide.md",
            objective="Learn a procedure",
            runtime=object(),  # type: ignore[arg-type]
        )


def test_run_learning_reloads_completed_study_and_rejects_policy_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "guide.md"
    source.write_text("# Guide\n\nInstructional material.\n", encoding="utf-8")
    store = LearningStore(tmp_path / "learning")
    engine = LearningEngine(store)
    ingestion = {
        "source": {"id": "source-1", "content_hash": "hash-1"},
        "chunks": [],
        "capability_discovery": {"schema": "rex-learning-capability-discovery-v1", "proposals": []},
    }
    autonomous_module = import_module("rex_learning.autonomous_study")
    monkeypatch.setattr(autonomous_module, "ingest_source", lambda *args, **kwargs: ingestion)
    runtime = LearningRuntime(learner=_Learner(), design_provider=lambda request: {})

    first = run_learning(engine=engine, source_path=source, objective="Learn a procedure", runtime=runtime, limit=None)
    run_id = store.list("study_runs")[0]["id"]
    calls = 0

    def unexpected_provider(request: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        raise AssertionError("completed study must be loaded from its persisted artifact")

    resumed = run_learning(
        engine=engine,
        source_path=source,
        objective="Learn a procedure",
        runtime=LearningRuntime(learner=_Learner(), design_provider=unexpected_provider),
        limit=0,
        resume_run_id=run_id,
    )
    assert resumed == {"schema": "rex-learning-run-v1", "runtime_policy": {}, "study": first["study"]}
    assert calls == 0
    with pytest.raises(LearningError, match="policy does not match"):
        run_learning(
            engine=engine,
            source_path=source,
            objective="Learn a procedure",
            runtime=LearningRuntime(learner=_Learner(), design_provider=unexpected_provider, runtime_policy={"changed": True}),
            limit=0,
            resume_run_id=run_id,
        )
