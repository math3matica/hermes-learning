from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .engine import LearningEngine
from .evaluator import ExecutableGrader
from .store import LearningStore


MATERIAL = """# Boundary classification procedure

Classify a boundary by the independent reason for change. A change to domain behavior is domain; a change only to deployment topology is deployment. For ambiguous wording, inspect the independent reason before choosing.
"""

REQUIREMENT = "reason for change classification"


def _answer_without_skill(task: dict[str, Any]) -> str:
    return _answer_with_skill(task, "")


def _answer_with_skill(task: dict[str, Any], procedure: str) -> str:
    facts = task.get("facts", {})
    reason = str(facts.get("reason", "")).casefold()
    instructions = procedure.casefold()
    if "independent reason" not in instructions:
        return "deployment"
    if "domain behavior" in reason and "domain" in instructions:
        return "domain"
    if "deployment topology" in reason and "deployment" in instructions:
        return "deployment"
    if "ambiguous" in reason and "do not inspect" not in instructions and "inspect ambiguous" in instructions:
        return "domain"
    return "deployment"


def _responses(cases: list[dict[str, Any]], answerer) -> dict[str, Any]:
    return {"responses": [{"case": case["id"], "answer": answerer(case)} for case in cases]}


def _test(engine: LearningEngine, curriculum: dict[str, Any], skill: dict[str, Any], grader: ExecutableGrader, phase: str, cases: list[dict[str, Any]]) -> dict[str, Any]:
    return engine.define_test(curriculum["id"], skill["id"], {"phase": phase, "cases": [case["id"] for case in cases], "success_threshold": 1.0, "evaluator_type": "executable_grader", "evaluator_id": grader.evaluator_id, "evaluator_independence": "independent", "contamination_status": "not_applicable"})


def _evaluate(engine: LearningEngine, test: dict[str, Any], responses: dict[str, Any], *, task_id: str = "") -> tuple[dict[str, Any], dict[str, Any]]:
    attempt = engine.record_attempt(test["id"], responses, task_id=task_id)
    return attempt, engine.evaluate_trusted_attempt(attempt["id"])


def _identify_requirements(description: str) -> list[str]:
    text = description.casefold()
    if any(marker in text for marker in ("independent reason", "behavioral contract", "api contract", "what changes rather than where deployed")):
        return [REQUIREMENT]
    if "deployment topology" in text or "place the service" in text:
        return ["deployment topology planning"]
    return []


def run_behavioral_reuse_experiment(root: Path) -> dict[str, Any]:
    """Run a fresh-context behavioral-reuse qualification.

    This is an objective, deterministic integration experiment. It proves that
    persisted demonstrated state can alter a later task through selective
    discovery and operational context. It does not prove provider cognition.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    material_path = root / "instructional-material.md"
    material_path.write_text(MATERIAL, encoding="utf-8")
    grader = ExecutableGrader(
        evaluator_id="grader.behavioral-reuse.v1",
        answer_key={
            "pre-1": "domain",
            "post-1": "deployment", "post-2": "domain",
            "retest-1": "deployment", "retest-2": "domain", "retest-3": "domain",
            "later-positive": "domain", "later-transfer": "domain",
        },
        provider="local-executable-grader",
        session_id="behavioral-reuse-grader-1",
    )

    acquisition = LearningEngine(LearningStore(root / "learning"), trusted_evaluators={grader.evaluator_id: grader})
    curriculum = acquisition.create_curriculum("Boundary classification behavioral reuse", "Acquire and later apply reason-for-change boundary classification", method_version="behavioral-reuse-v1")
    source = acquisition.add_source(curriculum["id"], "Boundary classification procedure", str(material_path), kind="markdown", content_hash=hashlib.sha256(MATERIAL.encode()).hexdigest())
    skill = acquisition.create_skill_hypothesis(curriculum["id"], {"key": "boundary.classification.behavioral", "kind": "procedure", "claim": "Classify a boundary by the independent reason for change.", "applicability": [REQUIREMENT], "operational_procedure": "Inspect the independent reason for change. Classify domain behavior as domain and deployment-only topology as deployment.", "limitations": ["Requires an observable reason for change"]})

    pre_cases = [{"id": "pre-1", "facts": {"reason": "domain behavior changed", "location": "unchanged"}}]
    pre = _test(acquisition, curriculum, skill, grader, "pretest", pre_cases)
    baseline_attempt, baseline_eval = _evaluate(acquisition, pre, _responses(pre_cases, lambda case: "deployment"))

    note = acquisition.record_study_note(curriculum["id"], source["id"], {"kind": "procedure", "claim": MATERIAL, "conditions": ["reason for change is observable"]})
    acquisition.record_knowledge(curriculum["id"], {"kind": "procedure", "claim": "Classify by independent reason for change, not deployment location.", "source_id": source["id"]})
    post_cases = [{"id": "post-1", "facts": {"reason": "deployment topology changed", "location": "new region"}}, {"id": "post-2", "facts": {"reason": "ambiguous reason", "location": "new region"}}]
    post = _test(acquisition, curriculum, skill, grader, "posttest", post_cases)
    post_attempt, post_eval = _evaluate(acquisition, post, _responses(post_cases, lambda case: _answer_with_skill(case, "Classify by reason, but do not inspect ambiguous cases.")))
    revision_note = acquisition.record_study_note(curriculum["id"], source["id"], {"kind": "revision", "claim": "For ambiguous wording, inspect the independent reason before choosing.", "conditions": ["post-study attempt failed on an ambiguous case"]})
    revision = acquisition.revise_skill(skill["id"], {"operational_procedure": "Inspect the independent reason for change. Classify domain behavior as domain, deployment-only topology as deployment, and inspect ambiguous cases rather than inferring from location."}, reason="Post-study evidence exposed an ambiguous-case weakness.", evidence_ids=[post_eval["id"], revision_note["id"]])
    retest_cases = [{"id": "retest-1", "facts": {"reason": "deployment topology changed", "location": "new region"}}, {"id": "retest-2", "facts": {"reason": "domain behavior changed", "location": "unchanged"}}, {"id": "retest-3", "facts": {"reason": "ambiguous reason", "location": "new region"}}]
    retest = _test(acquisition, curriculum, skill, grader, "retest", retest_cases)
    retest_attempt, retest_eval = _evaluate(acquisition, retest, _responses(retest_cases, lambda case: _answer_with_skill(case, revision["operational_procedure"])))
    demonstrated = acquisition.promote_demonstrated(skill["id"], pretest_attempt_id=baseline_attempt["id"], pretest_evaluation_id=baseline_eval["id"], posttest_evaluation_id=retest_eval["id"])

    # End acquisition context. The later engine receives only the persisted store.
    later_engine = LearningEngine(LearningStore(root / "learning"), trusted_evaluators={grader.evaluator_id: grader})
    positive_description = "A service's behavioral contract changes independently of where it is deployed. Classify the boundary by the independent reason."
    positive_task = {"task_id": "later-positive-task", "description": positive_description, "facts": {"reason": "domain behavior changed", "location": "unchanged"}, "requirements": _identify_requirements(positive_description)}
    selected = later_engine.discover_applicable_skills(positive_task, max_candidates=2, max_chars=2400)
    selected_skill = selected[0]
    later_case = [{"id": "later-positive", "facts": positive_task["facts"]}]
    later_test = _test(later_engine, curriculum, demonstrated, grader, "edge", later_case)
    learned_answer = _answer_with_skill(later_case[0], selected_skill["operational_context"]["procedure"])
    later_attempt, later_eval = _evaluate(later_engine, later_test, _responses(later_case, lambda case: learned_answer), task_id=positive_task["task_id"])
    application = later_engine.apply_skill(selected_skill["key"], positive_task["task_id"], selected_skill["operational_context"]["procedure"], outcome={"answer": learned_answer, "result": "later task classified boundary"}, selection_reason=selected_skill["why_selected"], operational_context=selected_skill["operational_context"])
    application = later_engine.attach_application_evidence(application["id"], execution_ref=later_attempt["id"], evaluation_id=later_eval["id"], outcome={"answer": learned_answer, "result": "later task classified boundary"})

    control_task = {"task_id": "later-control-task", "description": positive_description, "facts": positive_task["facts"], "requirements": positive_task["requirements"]}
    control_answer = _answer_without_skill(control_task)
    control_case = [{"id": "later-positive", "facts": control_task["facts"]}]
    control_engine = LearningEngine(LearningStore(root / "control-learning"), trusted_evaluators={grader.evaluator_id: grader})
    control_curriculum = control_engine.create_curriculum("Control", "Comparable later task")
    control_skill = control_engine.create_skill_hypothesis(control_curriculum["id"], {"key": "control", "claim": "No demonstrated procedure."})
    control_test = _test(control_engine, control_curriculum, control_skill, grader, "edge", control_case)
    _, control_eval = _evaluate(control_engine, control_test, _responses(control_case, lambda case: control_answer))

    negative_description = "Choose where to place the service based only on deployment topology."
    negative = later_engine.discover_applicable_skills({"task_id": "later-negative-task", "description": negative_description, "requirements": _identify_requirements(negative_description)})
    transfer_description = "The API contract changes even though the runtime location stays the same; decide what kind of boundary changed."
    transfer = later_engine.discover_applicable_skills({"task_id": "later-transfer-task", "description": transfer_description, "requirements": _identify_requirements(transfer_description)})

    result = {"schema": "rex-learning-behavioral-reuse-v1", "qualification_scope": "persisted_skill_selection_and_behavioral_influence_not_provider_cognition", "curriculum_id": curriculum["id"], "source_id": source["id"], "material": {"path": str(material_path), "study_note_id": note["id"], "revision_note_id": revision_note["id"]}, "baseline": {"attempt_id": baseline_attempt["id"], "evaluation_id": baseline_eval["id"], "score": baseline_eval["score"]}, "post_study": {"attempt_id": post_attempt["id"], "evaluation_id": post_eval["id"], "score": post_eval["score"]}, "revision": {"version": revision["version"], "evaluation_id": post_eval["id"]}, "held_out_retest": {"attempt_id": retest_attempt["id"], "evaluation_id": retest_eval["id"], "score": retest_eval["score"]}, "skill": {"id": demonstrated["id"], "version": demonstrated["version"], "state": demonstrated["state"]}, "persistence": {"fresh_context": True, "store_path": str(root / "learning")}, "later_task": {"task_id": positive_task["task_id"], "requirements": positive_task["requirements"], "selected_skill_id": selected_skill["skill_id"], "selected_skill_version": selected_skill["skill_version"], "application": application, "evaluation": later_eval}, "ablation": {"without_skill": {"task_id": control_task["task_id"], "score": control_eval["score"], "evaluation_id": control_eval["id"]}, "with_skill": {"task_id": positive_task["task_id"], "score": later_eval["score"], "evaluation_id": later_eval["id"]}}, "applicability": {"negative_selected": bool(negative), "transfer_selected": bool(transfer), "transfer_skill_id": transfer[0]["skill_id"] if transfer else None}, "evaluator": {"id": grader.evaluator_id, "provider": grader.provider, "session_id": grader.session_id}}
    (root / "behavioral-reuse-result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result
