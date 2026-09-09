from __future__ import annotations

import json
from pathlib import Path

from rex_learning.learner import HostLLMLearner
from rex_learning.store import LearningStore, resolve_learning_root


def test_learning_root_prefers_host_owned_state_directory(tmp_path: Path) -> None:
    host = type("Context", (), {"state_dir": tmp_path / "host-state"})()
    assert resolve_learning_root(host) == (tmp_path / "host-state" / "rex-learning").resolve()


def test_learning_root_uses_explicit_environment_override(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HERMES_REX_LEARNING_HOME", str(tmp_path / "configured"))
    assert resolve_learning_root() == (tmp_path / "configured").resolve()


def test_host_llm_learner_adapts_host_response_without_provider_name(tmp_path: Path) -> None:
    calls = []

    def llm(**kwargs):
        calls.append(kwargs)
        return {"responses": [{"case": "task", "answer": {"ok": True}}]}

    learner = HostLLMLearner(llm, session_id="s")
    result = learner.answer(task="case=task\nReturn JSON")
    assert result["responses"][0]["case"] == "task"
    assert calls and "model" not in calls[0] and "provider" not in calls[0]
    assert learner.provider == "host"


def test_host_llm_learner_parses_json_text(tmp_path: Path) -> None:
    learner = HostLLMLearner(lambda prompt: json.dumps({"responses": [{"case": "task", "answer": "ok"}]}))
    assert learner.answer(task="case=task")["responses"][0]["answer"] == "ok"
