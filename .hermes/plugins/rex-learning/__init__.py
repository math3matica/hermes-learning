"""Provider-neutral Rex Learning plugin facade.

The plugin owns orchestration and evidence boundaries while Hermes owns model
routing and profile state. It deliberately does not name a provider or model.
"""
from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any, Mapping

from rex_learning.engine import LearningEngine
from rex_learning.entrypoint import LearningRuntime, run_learning
from rex_learning.jobs import LearningJobStore, run_learning_job
from rex_learning.learner import HostLLMLearner
from rex_learning.store import LearningStore, resolve_learning_root


def _llm(ctx: Any) -> Any:
    value = getattr(ctx, "llm", None)
    return value if callable(value) else None


def _json_call(llm: Any, prompt: str) -> Mapping[str, Any]:
    try:
        raw = llm(prompt=prompt)
    except TypeError:
        raw = llm(prompt)
    if isinstance(raw, Mapping):
        raw = raw.get("output", raw.get("content", raw))
        if isinstance(raw, Mapping) and "choices" in raw:
            raw = raw["choices"][0]["message"].get("content", "")
    if isinstance(raw, str):
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        raw = json.loads(text)
    if not isinstance(raw, Mapping):
        raise ValueError("ctx.llm design response must be a JSON object")
    return raw


def _state_root(ctx: Any) -> Path:
    return resolve_learning_root(ctx)


def _status(ctx: Any) -> str:
    store = LearningStore(_state_root(ctx))
    return json.dumps({
        "schema": "rex-learning-plugin-status-v1",
        "state_root": str(store.root),
        "curricula": len(store.list("curricula")),
        "study_runs": len(store.list("study_runs")),
        "candidate_skills": len(store.list("candidate_skills")),
    }, sort_keys=True)


def _run(ctx: Any, source: str, objective: str) -> str:
    llm = _llm(ctx)
    if llm is None:
        return json.dumps({"status": "blocked", "reason": "host_llm_unavailable"}, sort_keys=True)
    store = LearningStore(_state_root(ctx))
    job = LearningJobStore(store).enqueue(source=source, objective=objective)
    return json.dumps({"status": "queued", "job": job}, sort_keys=True)


def _worker(ctx: Any, job_id: str) -> str:
    llm = _llm(ctx)
    if llm is None:
        return json.dumps({"status": "blocked", "reason": "host_llm_unavailable"}, sort_keys=True)
    store = LearningStore(_state_root(ctx))
    jobs = LearningJobStore(store)
    job = jobs.get(job_id)
    learner = HostLLMLearner(llm)

    def design_provider(request: Mapping[str, Any]) -> Mapping[str, Any]:
        prompt = (
            "Return one JSON object containing independent instructional designs. "
            "Do not include answer keys, learner solutions, provider/model choices, "
            "or executable commands. Request:\n" + json.dumps(dict(request), sort_keys=True, default=str)
        )
        return _json_call(llm, prompt)

    def execute(item: Mapping[str, Any]) -> Mapping[str, Any]:
        return run_learning(
            engine=LearningEngine(store), source_path=Path(item["source"]), objective=item["objective"],
            runtime=LearningRuntime(learner=learner, design_provider=design_provider,
                                    runtime_policy={"model_routing": "host-owned-ctx.llm"}),
        )
    lifecycle = getattr(ctx, "subagent_lifecycle", None)
    lifecycle = lifecycle if callable(lifecycle) else None
    result = run_learning_job(jobs, job_id, execute, lifecycle=lifecycle)
    return json.dumps({"status": result["state"], "job": result}, sort_keys=True, default=str)


def _command(ctx: Any, raw_args: str) -> str:
    try:
        args = shlex.split(raw_args)
    except ValueError as exc:
        return json.dumps({"status": "error", "error": str(exc)})
    if not args or args[0] == "status":
        return _status(ctx)
    if args[0] == "run" and len(args) >= 3:
        return _run(ctx, args[1], " ".join(args[2:]))
    if args[0] == "worker" and len(args) == 2:
        return _worker(ctx, args[1])
    return "Usage: /rex-learning status | run <source> <objective> | worker <job-id>"


def _cli_setup(parser: Any) -> None:
    parser.add_argument("operation", choices=("status", "run", "worker"))
    parser.add_argument("source", nargs="?")
    parser.add_argument("objective", nargs="*")


def register(ctx: Any) -> None:
    ctx.register_command("rex-learning", lambda raw: _command(ctx, raw), description="Run provider-neutral Rex Learning jobs.", args_hint="status | run <source> <objective>")
    if hasattr(ctx, "register_cli_command"):
        def handler(args: Any) -> None:
            raw = args.operation
            if args.source:
                raw += " " + shlex.quote(args.source)
            if args.objective:
                raw += " " + shlex.quote(" ".join(args.objective))
            print(_command(ctx, raw))
        ctx.register_cli_command("rex-learning", help="Run provider-neutral Rex Learning jobs.", setup_fn=_cli_setup, handler_fn=handler, description="Run source-to-study jobs using the active host model.")
    ctx.register_skill("rex-learning", Path(__file__).with_name("SKILL.md"), description="Evidence-gated, provider-neutral learning orchestration.")
