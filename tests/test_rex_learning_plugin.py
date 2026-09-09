from __future__ import annotations

import importlib.util
import json
from pathlib import Path

PLUGIN = Path(__file__).parents[1] / ".hermes/plugins/rex-learning/__init__.py"
spec = importlib.util.spec_from_file_location("rex_learning_plugin_test", PLUGIN)
assert spec and spec.loader
plugin = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plugin)


def test_rex_plugin_registers_command_cli_and_skill() -> None:
    class Context:
        state_dir = Path("/tmp/host-state")
        llm = None
        def __init__(self): self.commands=[]; self.cli=[]; self.skills=[]
        def register_command(self, *args, **kwargs): self.commands.append((args, kwargs))
        def register_cli_command(self, *args, **kwargs): self.cli.append((args, kwargs))
        def register_skill(self, *args, **kwargs): self.skills.append((args, kwargs))
    ctx = Context()
    plugin.register(ctx)
    assert ctx.commands[0][0][0] == "rex-learning"
    assert ctx.cli[0][0][0] == "rex-learning"
    assert ctx.skills[0][0][0] == "rex-learning"


def test_status_reports_context_owned_state(tmp_path: Path) -> None:
    ctx = type("Context", (), {"state_dir": tmp_path / "state"})()
    result = json.loads(plugin._status(ctx))
    assert result["state_root"] == str((tmp_path / "state/rex-learning").resolve())
    assert result["study_runs"] == 0


def test_run_fails_closed_without_host_llm(tmp_path: Path) -> None:
    ctx = type("Context", (), {"state_dir": tmp_path / "state"})()
    result = json.loads(plugin._run(ctx, str(tmp_path / "missing.epub"), "learn"))
    assert result == {"status": "blocked", "reason": "host_llm_unavailable"}
