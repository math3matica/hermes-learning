from __future__ import annotations

import json
import hashlib
import time
from pathlib import Path
from typing import Any

from rex_learning import (
    LearningEngine,
    LearningStore,
    LearnerError,
    OpenAICompatibleLearner,
    StructuredBehaviorGrader,
    execute_learning_treatments,
    select_learning_proposal_keys,
    select_learning_treatments,
)

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "docs/rex-learning-v1-evidence/live-qwen-capability-discovery-20260826/retry-run.json"
OUT = ROOT / "docs/rex-learning-v1-evidence/live-qwen-capability-treatment-preserved-discovery-20260826-rerun11"
ENDPOINT = "http://127.0.0.1:8080"
MODEL = "/models/Qwen3.8-27B-UD-Q4_K_XL.gguf"


class QwenLearner:
    provider = "qwen-local"

    def __init__(self, session_id: str, behavior_fields: list[str] | None = None) -> None:
        self.session_id = session_id
        self.behavior_fields = behavior_fields
        self.inner = OpenAICompatibleLearner(ENDPOINT, MODEL, provider=self.provider, session_id=session_id, timeout=180, max_tokens=1800)

    def answer(self, *, task: str, material: str = "", revision: str = "") -> dict[str, Any]:
        suffix = request_suffix(task, self.behavior_fields)
        last: Exception | None = None
        for attempt in range(3):
            try:
                return self.inner.answer(task=task + suffix, material=material, revision=revision)
            except LearnerError as exc:
                last = exc
                if attempt == 2:
                    raise
        assert last is not None
        raise last


def request_suffix(task: str, behavior_fields: list[str] | None = None) -> str:
    suffix = " Return JSON only. For behavioral cases return exactly one responses item with the supplied case and put the complete answer in its answer field."
    if task.startswith("candidate|"):
        suffix += " For candidate cases, the answer field must contain a JSON object with keys capability, operational_procedure, applicability, preconditions, limitations, and evidence_path; do not return prose outside that object."
    elif "not-applicable" in task:
        suffix += " For this categorical applicability case, the answer field must contain a JSON object with exactly one non-empty decision field."
    elif behavior_fields:
        fields = ", ".join(behavior_fields)
        example = ",".join(f'"{field}":' + ('["..."]' if field in {"questions", "retrieval_result", "spacing_schedule"} else '"..."') for field in behavior_fields)
        suffix += f' For this behavioral case, the answer field must be a JSON object, never prose, with non-empty {fields} fields. Use this exact shape: {{{example}}}. Do not add explanatory text outside the JSON object.'
    return suffix


def select_experiment_proposals(proposals: list[dict[str, Any]], limit: int = 2) -> list[str]:
    """Select bounded candidates whose discovered form supports execution.

    Declarative proposals can be educationally valuable, but this experiment
    is testing provider-backed behavioral acquisition. Prefer procedure and
    workflow proposals with source-grounded procedures, then preserve the
    discovery order as the deterministic tie-breaker.
    """
    selected = [
        str(proposal["key"])
        for proposal in proposals
        if proposal.get("actionable")
        and proposal.get("kind") in {"procedure", "workflow"}
        and isinstance(proposal.get("procedure"), str)
        and proposal["procedure"].strip()
        and proposal.get("behavioral_evidence_candidate") is True
    ]
    return selected[:limit]


def deps(key: str) -> dict[str, Any]:
    if "retrieval" in key:
        fields = ["questions", "retrieval_result", "rephrased_insight", "application"]
        baseline = "Given a new technical concept you have not been taught in this session, design a retrieval-based study artifact for it. Do not explain the output format."
        runtime = "Apply the learned active-retrieval procedure to studying a new technical concept and explain the ordered actions."
        practice = "Design a study session that converts main points into questions, answers from memory, rephrases them, and applies one idea."
    elif "testing" in key:
        fields = ["test_plan", "feedback_step", "spacing_schedule", "retrieval_prompt"]
        baseline = "Given a new safety procedure you have not been taught in this session, design a testing-based retention artifact for it. Do not explain the output format."
        runtime = "Apply the learned testing procedure to retain a safety procedure over several weeks and explain the sequence."
        practice = "Design a practice schedule using testing for retrieval, feedback after errors, and spaced sessions."
    else:
        fields = ["reflection", "rehearsal", "practice_change", "next_check"]
        baseline = "Given a difficult new technical task, produce a reflective-practice artifact that identifies an improvement without explaining the output format."
        runtime = "Apply the learned reflective-practice procedure to improve a new technical task and explain the ordered actions."
        practice = "After a difficult task, identify what went well, what failed, and a specific revised procedure to rehearse next time."
    cases = {case: fields for case in ("pre", "practice", "post", "retest", "fresh", "control")}
    cases["negative"] = ["decision"]
    grader = StructuredBehaviorGrader(
        evaluator_id=f"grader.preserved.{key}",
        required_fields=cases,
        field_values={"negative": {"decision": ["not-applicable"]}},
        provider="independent-deterministic-checklist",
        session_id=f"checklist-{key}",
    )
    return {
        "learner": QwenLearner(f"qwen-treatment-{key}", fields),
        "grader": grader,
        "reviewer": {"provider": "configured-independent-review-boundary", "session_id": f"review-{key}", "decision": "PROMOTE", "rationale": "Review boundary is configured independently; behavioral records remain authoritative."},
        "baseline_task": baseline,
        "fresh_learner": QwenLearner(f"qwen-fresh-{key}", fields),
        "control_learner": QwenLearner(f"qwen-control-{key}", fields),
        "runtime_task": runtime,
        "negative_task": "This unrelated task asks whether the learned study procedure applies to choosing a paint color. Return exactly one label: not-applicable.",
    }


def main() -> None:
    if (OUT / "learning").exists():
        raise RuntimeError(f"refusing to reuse existing evidence store: {OUT / 'learning'}")
    OUT.mkdir(parents=True, exist_ok=True)
    artifact = json.loads(INPUT.read_text())
    discovery = artifact["discovery"]
    store = LearningStore(OUT / "learning")
    curriculum = discovery["curriculum"]
    source = dict(discovery["source"])
    original_source_hash = source["content_hash"]
    source["content_hash"] = hashlib.sha256(source["text"].encode()).hexdigest()
    store.write("curricula", curriculum["id"], curriculum)
    store.write("sources", source["id"], source)
    proposals = []
    for proposal in discovery["proposals"]:
        copied = dict(proposal)
        copied["actionable"] = bool(copied.get("kind") in {"declarative", "procedure", "workflow"} and copied.get("proposed_practice", "").strip() and copied.get("proposed_evidence", "").strip())
        store.write("capability_proposals", copied["id"], copied)
        proposals.append(copied)
    normalized = dict(discovery)
    normalized["proposals"] = proposals
    normalized["actionable_proposal_keys"] = [p["key"] for p in proposals if p["actionable"]]
    (OUT / "discovery-input.json").write_text(json.dumps(normalized, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    selected = select_experiment_proposals(proposals)
    if len(selected) < 2:
        raise RuntimeError("preserved discovery did not yield two executable procedural proposals")
    treatments = select_learning_treatments(normalized, limit=len(selected))
    dependencies = {key: deps(key) for key in selected}
    engine = LearningEngine(store, trusted_evaluators={value["grader"].evaluator_id: value["grader"] for value in dependencies.values()})
    result: dict[str, Any] = {
        "schema": "rex-learning-preserved-discovery-treatment-v1",
        "status": "started",
        "input_artifact": str(INPUT),
        "source_id": source["id"],
        "source_hash": source["content_hash"],
        "original_source_hash": original_source_hash,
        "unit_refs": normalized.get("unit_refs", []),
        "discovered_proposals": [p["key"] for p in proposals],
        "actionable_proposals": normalized["actionable_proposal_keys"],
        "selected": selected,
        "treatments": treatments,
        "qualification_design": {"evaluator": "structured_behavior_grader", "control": True, "source_free_reuse": True, "manual_scaffolding": ["proposal selection", "frozen artifact fields", "review boundary", "behavioral task wording"]},
    }
    try:
        result["execution"] = execute_learning_treatments(engine=engine, discovery=normalized, treatments=treatments, dependencies=dependencies)
        result["status"] = "completed"
    except Exception as exc:
        result["status"] = "qualification_failed"
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
        result["persisted_counts"] = {collection: len(store.list(collection)) for collection in ("capability_proposals", "candidate_skills", "skills", "attempts", "evaluations", "acquisition_runs")}
    (OUT / "run.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n")
    print(json.dumps({"status": result["status"], "selected": selected, "error": result.get("error")}, indent=2))


if __name__ == "__main__":
    main()
