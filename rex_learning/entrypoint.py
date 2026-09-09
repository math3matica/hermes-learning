from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .autonomous_study import IndependentDesignProvider, autonomous_study
from .engine import LearningEngine, LearningError
from .learner import Learner


@dataclass(frozen=True)
class LearningRuntime:
    """Provider-neutral runtime bindings for the public learning entrypoint.

    The runtime supplies execution boundaries, not instructional answers:
    discovery is performed by the learner and independent designs are supplied
    by the configured design boundary. Trusted qualification dependencies are
    optional, but when present they are passed unchanged to the existing
    fail-closed acquisition executor.
    """

    learner: Learner
    design_provider: IndependentDesignProvider
    qualification_dependencies: Mapping[str, Mapping[str, Any]] | None = None
    runtime_policy: Mapping[str, Any] = field(default_factory=dict)


def run_learning(
    *,
    engine: LearningEngine,
    source_path: Path,
    objective: str,
    runtime: LearningRuntime,
    limit: int | None = 2,
    baseline_scores: Mapping[str, Any] | None = None,
    successful_acquisition_limit: int | None = 2,
    discovery_packet_chars: int = 12000,
    resume_run_id: str | None = None,
) -> dict[str, Any]:
    """Run the generic Rex Learning V1 source-to-study entrypoint.

    Callers provide only a source, broad objective, engine, and provider
    bindings. Capability names, tests, answer keys, and practice plans remain
    outputs of the existing discovery/design boundaries rather than public
    inputs. This function deliberately does not construct a learner, grader,
    reviewer, or other trusted evaluator from provider configuration.
    """
    if not isinstance(runtime, LearningRuntime):
        raise LearningError("runtime must be a LearningRuntime")
    if not isinstance(runtime.runtime_policy, Mapping):
        raise LearningError("runtime policy must be a mapping")

    study = autonomous_study(
        engine=engine,
        source_path=Path(source_path),
        learner=runtime.learner,
        educational_objective=objective,
        design_provider=runtime.design_provider,
        qualification_dependencies=runtime.qualification_dependencies,
        limit=limit,
        baseline_scores=baseline_scores,
        successful_acquisition_limit=successful_acquisition_limit,
        discovery_packet_chars=discovery_packet_chars,
        runtime_policy=runtime.runtime_policy,
        resume_run_id=resume_run_id,
    )
    return {
        "schema": "rex-learning-run-v1",
        "runtime_policy": dict(runtime.runtime_policy),
        "study": study,
    }
