from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .capability_discovery import execute_learning_treatments
from .engine import LearningEngine, LearningError
from .deep_study import DeepStudyEngine
from .ingestion import ingest_source
from .learner import Learner
from .preflight import prepare_independent_designs


IndependentDesignProvider = Callable[[Mapping[str, Any]], Mapping[str, Any]]
def autonomous_study(
    *,
    engine: LearningEngine,
    source_path: Path,
    learner: Learner,
    educational_objective: str,
    design_provider: IndependentDesignProvider,
    qualification_dependencies: Mapping[str, Mapping[str, Any]] | None = None,
    limit: int | None = 2,
    baseline_scores: Mapping[str, Any] | None = None,
    successful_acquisition_limit: int | None = None,
    discovery_packet_chars: int = 12000,
    runtime_policy: Mapping[str, Any] | None = None,
    resume_run_id: str | None = None,
) -> dict[str, Any]:
    """Route a supported instructional source through discovery and design binding.

    The caller supplies a source, broad objective, learner, and independent
    design provider; it does not supply capability names, test cases, answer
    keys, or practice plans. Qualification execution remains a separate step
    until trusted evaluator and learner dependencies are bound.
    """
    if not callable(design_provider):
        raise LearningError("design_provider must be callable")
    if not isinstance(educational_objective, str) or not educational_objective.strip():
        raise LearningError("educational_objective is required")
    source_path = Path(source_path)
    curriculum = engine.create_curriculum(source_path.name, educational_objective)
    ingestion = ingest_source(
        engine,
        curriculum["id"],
        source_path,
        learner=learner,
        educational_objective=educational_objective,
        discovery_packet_chars=discovery_packet_chars,
    )
    source = ingestion.get("source")
    if not isinstance(source, Mapping):
        raise LearningError("source ingestion did not produce source identity")
    study_engine = DeepStudyEngine(engine.store)
    policy = dict(runtime_policy or {})
    study_run = study_engine.create_run(
        title=f"Rex Learning: {source_path.name} — {educational_objective.strip()}",
        source_id=str(source["id"]),
        source_hash=str(source["content_hash"]),
        method_version="rex-learning.autonomous-study-v1",
        provider_policy=policy,
    )
    if study_run.get("provider_policy") != policy:
        raise LearningError("resume run policy does not match the original run")
    if resume_run_id is not None and resume_run_id != study_run["id"]:
        raise LearningError("resume run does not match source identity or method version")
    if study_run.get("checkpoint", {}).get("last_phase") == "completed":
        artifacts = study_run.get("artifacts")
        if isinstance(artifacts, Mapping) and isinstance(artifacts.get("study"), Mapping):
            return dict(artifacts["study"])
    study_engine.checkpoint(study_run["id"], phase="ingested", artifacts={"ingestion": ingestion})
    discovery = ingestion.get("capability_discovery")
    if not isinstance(discovery, Mapping):
        raise LearningError("EPUB ingestion did not produce capability discovery")
    selection_limit = None if successful_acquisition_limit is not None else limit
    preflight = prepare_independent_designs(
        discovery=discovery,
        design_provider=design_provider,
        limit=selection_limit,
        baseline_scores=baseline_scores,
    )
    treatments = preflight["treatments"]
    requests = preflight["requests"]
    bound_designs = preflight["bound_designs"]
    study_engine.checkpoint(study_run["id"], phase="discovered", artifacts={"ingestion": ingestion, "treatments": treatments})
    study_engine.checkpoint(study_run["id"], phase="design_bound", artifacts={"ingestion": ingestion, "preflight": preflight})
    execution: dict[str, Any] | None = None
    if qualification_dependencies is not None:
        source_text = "\n\n".join(
            f"## {chunk['heading']}\n{chunk['text']}"
            for chunk in ingestion.get("chunks", [])
            if isinstance(chunk, Mapping)
            and isinstance(chunk.get("heading"), str)
            and isinstance(chunk.get("text"), str)
        )
        if not source_text.strip():
            raise LearningError("EPUB ingestion produced no instructional text for qualification")
        chunks_by_ref = {
            str(chunk["source_ref"]): str(chunk["text"])
            for chunk in ingestion.get("chunks", [])
            if isinstance(chunk, Mapping)
            and isinstance(chunk.get("source_ref"), str)
            and isinstance(chunk.get("text"), str)
        }
        source_text_by_intent = {
            str(treatment["intent_key"]): "\n\n".join(
                chunks_by_ref[ref]
                for ref in treatment.get("source_refs", [])
                if isinstance(ref, str) and ref in chunks_by_ref
            )
            for treatment in treatments
        }
        execution = execute_learning_treatments(
            engine=engine,
            discovery=discovery,
            treatments=treatments,
            dependencies=qualification_dependencies,
            learner=learner,
            independent_designs={
                str(design["intent_key"]): design
                for design in bound_designs
                if isinstance(design.get("intent_key"), str)
            },
            source_text=source_text,
            source_text_by_intent=source_text_by_intent,
            successful_acquisition_limit=successful_acquisition_limit,
            continue_on_ceiling=successful_acquisition_limit is not None,
        )
    result = {
        "schema": "rex-learning-autonomous-study-v1",
        "curriculum": curriculum,
        "ingestion": ingestion,
        "discovery": discovery,
        "treatments": treatments,
        "requests": requests,
        "bound_designs": bound_designs,
        "preflight": preflight,
        "execution": execution,
    }
    study_engine.checkpoint(study_run["id"], phase="completed", artifacts={"study": result})
    return result
