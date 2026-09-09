from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from rex_learning import (
    ExecutableGrader,
    LearningEngine,
    LearningStore,
    TrustedEvaluator,
    learn_multiple_from_source,
)


class CapabilityLearner:
    provider = "test-learner"

    def __init__(self, mode: str, session_id: str) -> None:
        self.mode = mode
        self.session_id = session_id

    def answer(self, *, task: str, material: str = "", revision: str = "") -> dict:
        if task.startswith("candidate"):
            return {
                "responses": [],
                "capability": f"Apply {self.mode} rule",
                "applicability": [f"{self.mode} tasks"],
                "procedure": f"Apply the {self.mode} rule and reject other modes.",
                "preconditions": [f"Input is a {self.mode} task."],
                "limitations": ["Not applicable to other modes."],
            }
        case = task.split("|case=", 1)[1].split("|", 1)[0]
        if case == "negative":
            answer = "not-applicable"
        elif case == "pre":
            answer = "wrong"
        else:
            answer = f"{self.mode}-ok"
        return {"responses": [{"case": case, "answer": answer}]}


class WorkflowLearner(CapabilityLearner):
    def answer(self, *, task: str, material: str = "", revision: str = "") -> dict:
        result = super().answer(task=task, material=material, revision=revision)
        if task.startswith("candidate"):
            result["procedure"] = "1. inspect the input; 2. verify the prerequisite; 3. record the result."
            result["preconditions"] = ["Input is available before verification."]
        elif task.split("|case=", 1)[1].split("|", 1)[0] not in {"pre", "negative"}:
            result["responses"][0]["answer"] = "workflow-pass"
        return result


def explicit_reviewer(candidate: dict, evidence: dict) -> dict:
    claim = str(candidate.get("proposed_skill", {}).get("claim") or candidate.get("candidate_id", "candidate"))
    return {
        "candidate_id": candidate["candidate_id"],
        "decision": "PROMOTE",
        "rationale": "The candidate surface is operational and bounded.",
        "accepted_claims": [claim],
        "rejected_claims": [],
        "required_changes": [],
        "source_refs": [],
        "scope_limits": [],
        "reviewed_surface": {
            "claim": claim,
            "operational_procedure": str(candidate.get("proposed_skill", {}).get("operational_procedure") or "Apply the reviewed procedure."),
        },
    }


def test_one_source_can_qualify_independent_skills_without_pipeline_branches(tmp_path: Path) -> None:
    learners = {
        mode: CapabilityLearner(mode, f"{mode}-session") for mode in ("alpha", "beta")
    }
    graders = {
        mode: ExecutableGrader(
            evaluator_id=f"grader.{mode}",
            answer_key={case: ("not-applicable" if case == "negative" else f"{mode}-ok") for case in ("pre", "practice", "post", "retest", "negative", "fresh", "control")},
            provider="independent-grader",
            session_id=f"grader-{mode}",
        )
        for mode in learners
    }
    engine = LearningEngine(
        LearningStore(tmp_path / "learning"),
        trusted_evaluators=cast(dict[str, TrustedEvaluator], {grader.evaluator_id: grader for grader in graders.values()}),
    )
    result = learn_multiple_from_source(
        engine=engine,
        learner=learners["alpha"],
        source_title="Two-rule field guide",
        source_text="Alpha and beta are independent procedures with distinct applicability conditions.",
        curriculum_title="Independent field procedures",
        capabilities=[
            {"intent_key": "alpha-rule", "target_capability": "Apply alpha", "grader": graders["alpha"], "reviewer": explicit_reviewer, "learner": learners["alpha"], "fresh_learner": CapabilityLearner("alpha", "alpha-fresh")},
            {"intent_key": "beta-rule", "target_capability": "Apply beta", "skill_kind": "workflow", "grader": graders["beta"], "reviewer": explicit_reviewer, "learner": learners["beta"], "fresh_learner": CapabilityLearner("beta", "beta-fresh")},
        ],
    )
    assert result["schema"] == "rex-learning-multi-skill-acquisition-v1"
    assert len(result["runs"]) == 2
    assert len({run["source"]["id"] for run in result["runs"]}) == 1
    assert {run["promotion"]["key"] for run in result["runs"]} == {"alpha-rule", "beta-rule"}
    assert {run["promotion"]["kind"] for run in result["runs"]} == {"procedure", "workflow"}
    assert all(run["fresh_context"]["application"]["outcome"]["passed"] for run in result["runs"])
    assert all(run["negative_applicability"] is False for run in result["runs"])
    bundle = engine.compose_applicable_skills({"requirements": ["alpha tasks", "beta tasks"]})
    assert bundle["schema"] == "rex-learning-composed-context-v1"
    assert {item["key"] for item in bundle["skills"]} == {"alpha-rule", "beta-rule"}
    assert len(bundle["operational_contexts"]) == 2


def test_independently_qualified_skills_execute_in_order_and_are_qualified_as_composition(tmp_path: Path) -> None:
    learners = {mode: CapabilityLearner(mode, f"{mode}-compose-session") for mode in ("alpha", "beta")}
    graders = {mode: ExecutableGrader(evaluator_id=f"grader.compose.{mode}", answer_key={case: ("not-applicable" if case == "negative" else f"{mode}-ok") for case in ("pre", "practice", "post", "retest", "negative", "fresh", "control")}, provider="independent-grader", session_id=f"grader-compose-{mode}") for mode in learners}
    composition_grader = ExecutableGrader(evaluator_id="grader.composition", answer_key={"composed": "alpha-then-beta"}, provider="independent-composition-grader", session_id="composition-grader")
    engine = LearningEngine(LearningStore(tmp_path / "learning"), trusted_evaluators=cast(dict[str, TrustedEvaluator], {grader.evaluator_id: grader for grader in graders.values()} | {composition_grader.evaluator_id: composition_grader}))
    acquired = learn_multiple_from_source(engine=engine, learner=learners["alpha"], source_title="Composable field guide", source_text="Alpha prepares the input; beta validates the prepared result.", curriculum_title="Composable procedures", capabilities=[{"intent_key": "alpha-rule", "target_capability": "alpha tasks", "grader": graders["alpha"], "reviewer": explicit_reviewer, "learner": learners["alpha"], "fresh_learner": CapabilityLearner("alpha", "alpha-fresh")}, {"intent_key": "beta-rule", "target_capability": "beta tasks", "grader": graders["beta"], "reviewer": explicit_reviewer, "learner": learners["beta"], "fresh_learner": CapabilityLearner("beta", "beta-fresh")}])
    assert all(run["promotion"]["state"] == "demonstrated" for run in acquired["runs"])
    observed: list[str] = []
    def executor(*, task_context: dict, composition: dict, operational_contexts: list[dict]) -> dict:
        observed.extend(item["skill_id"] for item in operational_contexts)
        answer = "alpha-then-beta" if len(operational_contexts) == 2 else "missing"
        return {"responses": [{"case": "composed", "answer": answer}], "provider_response": {"used_skill_ids": list(observed)}}
    def control_executor(*, task_context: dict, composition: dict, operational_contexts: list[dict]) -> dict:
        return {"responses": [{"case": "composed", "answer": "missing"}], "provider_response": {"used_skill_ids": []}}
    result = engine.qualify_composed_task({"requirements": ["alpha tasks", "beta tasks"], "execution_order": ["alpha-rule", "beta-rule"]}, task_id="novel-alpha-beta", executor=executor, control_executor=control_executor, evaluator=composition_grader)
    assert result["schema"] == "rex-learning-composition-run-v1"
    assert result["composition"] == {"mode": "sequential", "basis": "task-declared dependency order", "order": ["alpha-rule", "beta-rule"]}
    assert result["evaluation"]["passed"] is True
    assert result["control"]["evaluation"]["passed"] is False
    assert observed == result["discovered_skill_ids"]
    assert len(result["applications"]) == 2
    assert len(engine.store.list("composition_runs")) == 1

    def failing_executor(*, task_context: dict, composition: dict, operational_contexts: list[dict]) -> dict:
        return {"responses": [{"case": "composed", "answer": "component-only"}], "provider_response": {"used_skill_ids": [item["skill_id"] for item in operational_contexts]}}

    failed = engine.qualify_composed_task({"requirements": ["alpha tasks", "beta tasks"], "execution_order": ["alpha-rule", "beta-rule"]}, task_id="novel-alpha-beta-failed", executor=failing_executor, evaluator=composition_grader)
    assert failed["evaluation"]["passed"] is False
    assert failed["schema"] == "rex-learning-composition-run-v1"
    assert len(engine.store.list("composition_runs")) == 2
    assert all(engine.skill(skill_id)["state"] in {"demonstrated", "robust"} for skill_id in result["discovered_skill_ids"])


def test_composition_does_not_silently_drop_ordered_skills_when_context_is_too_small(tmp_path: Path) -> None:
    learner = CapabilityLearner("alpha", "bounded-compose-session")
    graders = {
        mode: ExecutableGrader(
            evaluator_id=f"grader.bound.{mode}",
            answer_key={case: ("not-applicable" if case == "negative" else f"{mode}-ok") for case in ("pre", "practice", "post", "retest", "negative", "fresh", "control")},
            provider="independent-grader",
            session_id=f"grader-bound-{mode}",
        )
        for mode in ("alpha", "beta")
    }
    engine = LearningEngine(
        LearningStore(tmp_path / "learning"),
        trusted_evaluators=cast(dict[str, TrustedEvaluator], {grader.evaluator_id: grader for grader in graders.values()}),
    )
    learn_multiple_from_source(
        engine=engine,
        learner=learner,
        source_title="Bounded composition guide",
        source_text="Alpha prepares the input; beta validates it.",
        curriculum_title="Bounded composition",
        capabilities=[
            {"intent_key": "alpha-rule", "target_capability": "Apply alpha", "grader": graders["alpha"], "reviewer": explicit_reviewer, "learner": CapabilityLearner("alpha", "alpha-bound"), "fresh_learner": CapabilityLearner("alpha", "alpha-bound-fresh")},
            {"intent_key": "beta-rule", "target_capability": "Apply beta", "grader": graders["beta"], "reviewer": explicit_reviewer, "learner": CapabilityLearner("beta", "beta-bound"), "fresh_learner": CapabilityLearner("beta", "beta-bound-fresh")},
        ],
    )
    with pytest.raises(ValueError, match="composition context exceeds max_chars"):
        engine.compose_applicable_skills({"requirements": ["alpha tasks", "beta tasks"], "execution_order": ["alpha-rule", "beta-rule"]}, max_chars=100)


def test_workflow_process_execution_is_independently_qualified_and_persisted(tmp_path: Path) -> None:
    learner = WorkflowLearner("workflow", "workflow-study")
    grader = ExecutableGrader(evaluator_id="grader.workflow-process", answer_key={case: ("not-applicable" if case == "negative" else "workflow-pass") for case in ("pre", "practice", "post", "retest", "negative", "fresh", "control", "process")}, provider="independent-process-grader", session_id="workflow-process-grader")
    engine = LearningEngine(LearningStore(tmp_path / "learning"), trusted_evaluators={grader.evaluator_id: grader})
    acquired = learn_multiple_from_source(
        engine=engine,
        learner=learner,
        source_title="Workflow guide",
        source_text="A reliable verification workflow is ordered: inspect, verify prerequisites, then record the result.",
        curriculum_title="Workflow qualification",
        capabilities=[{"intent_key": "verification-workflow", "target_capability": "Use a verification workflow", "skill_kind": "workflow", "grader": grader, "reviewer": explicit_reviewer, "learner": learner, "fresh_learner": WorkflowLearner("workflow", "workflow-fresh")}],
    )
    skill = acquired["runs"][0]["promotion"]
    observed: list[str] = []

    def executor(*, task_context: dict, operational_context: dict | None) -> dict:
        if operational_context:
            observed.extend(["inspect", "verify", "record"])
            assert operational_context["procedure"].startswith("1. inspect")
            answer = "workflow-pass"
        else:
            answer = "missing-prerequisite"
        return {"responses": [{"case": "process", "answer": answer}], "provider_response": {"observed_steps": list(observed)}}

    result = engine.qualify_process_task({"requirements": ["workflow tasks"]}, skill_key="verification-workflow", task_id="fresh-workflow-task", executor=executor, control_executor=executor, evaluator=grader)
    assert result["schema"] == "rex-learning-process-run-v1"
    assert result["evaluation"]["passed"] is True
    assert result["control"]["evaluation"]["passed"] is False
    assert result["application"]["execution_ref"] == result["attempt"]["id"]
    assert result["application"]["evaluator_ref"] == result["evaluation"]["id"]
    assert result["skill_id"] == skill["id"]
    assert observed == ["inspect", "verify", "record"]
    assert len(engine.store.list("process_runs")) == 1
    assert len(engine.store.list("process_attempts")) == 2
    assert len(engine.store.list("process_evaluations")) == 2
