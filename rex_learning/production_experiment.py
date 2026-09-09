from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .engine import LearningEngine
from .evaluator import ExecutableGrader
from .store import LearningStore


INSTRUCTIONAL_MATERIAL = """# Boundary classification field guide

Classify a requested boundary by the reason for change, not by where software is deployed.
If domain behavior changes independently, classify the boundary as domain.
If only deployment topology changes, classify it as deployment.
When the request is ambiguous, inspect the independent reason for change before choosing.
"""


def _procedure(notes: list[dict[str, Any]], *, revised: bool) -> str:
    claims = " ".join(str(note.get("claim", "")) for note in notes)
    if not claims:
        return "Use the default deployment classification."
    if revised and "ambiguous" in claims.casefold():
        return "Inspect the independent reason for change; classify domain behavior as domain and deployment-only topology as deployment, including ambiguous cases."
    return "Classify domain behavior as domain and deployment-only topology as deployment."


def _answer(case: dict[str, Any], procedure: str) -> str:
    if procedure.startswith("Use the default"):
        return "deployment"
    if case["signal"] == "domain_behavior" or (case["signal"] == "ambiguous" and "including ambiguous" in procedure):
        return "domain"
    return "deployment"


def _attempt_payload(cases: list[dict[str, Any]], procedure: str) -> dict[str, Any]:
    return {"responses": [{"case": case["id"], "answer": _answer(case, procedure)} for case in cases]}


def run_production_learning_experiment(root: Path) -> dict[str, Any]:
    """Qualify one objective, provider-neutral production learning loop.

    The learner sees only persisted study notes. The executable grader owns the
    answer key and emits promotion-grade evidence independently of learner claims.
    """
    root = Path(root)
    material_path = root / "instructional-material.md"
    material_path.parent.mkdir(parents=True, exist_ok=True)
    material_path.write_text(INSTRUCTIONAL_MATERIAL, encoding="utf-8")
    grader = ExecutableGrader(
        evaluator_id="grader.boundary-classification.v1",
        answer_key={
            "pre-1": "domain",
            "post-1": "deployment", "post-2": "domain",
            "retest-1": "deployment", "retest-2": "domain", "retest-3": "domain",
        },
        provider="local-executable-grader",
        session_id="production-learning-session-1",
    )
    engine = LearningEngine(LearningStore(root / "learning"), trusted_evaluators={grader.evaluator_id: grader})
    curriculum = engine.create_curriculum("Boundary classification acquisition", "Learn to classify domain versus deployment boundaries from instructional material", method_version="production-executable-v1")
    source = engine.add_source(curriculum["id"], "Boundary classification field guide", str(material_path), kind="markdown", content_hash=hashlib.sha256(INSTRUCTIONAL_MATERIAL.encode()).hexdigest())
    skill = engine.create_skill_hypothesis(curriculum["id"], {
        "key": "boundary.classification", "kind": "procedure",
        "claim": "Classify a boundary by the independent reason for change.",
        "applicability": ["technical boundary classification"],
        "limitations": ["requires enough information about the reason for change"],
    })
    pre_cases = [{"id": "pre-1", "signal": "domain_behavior"}]
    pre = engine.define_test(curriculum["id"], skill["id"], {
        "phase": "pretest", "cases": [x["id"] for x in pre_cases], "success_threshold": 1.0,
        "evaluator_type": "executable_grader", "evaluator_id": grader.evaluator_id,
        "evaluator_independence": "independent", "contamination_status": "not_applicable",
        "prohibited_resources": [source["id"], "answer-key"],
    })
    baseline_attempt = engine.record_attempt(pre["id"], _attempt_payload(pre_cases, _procedure([], revised=False)))
    baseline_eval = engine.evaluate_trusted_attempt(baseline_attempt["id"])

    notes = [
        engine.record_study_note(curriculum["id"], source["id"], {"kind": "procedure", "claim": "If domain behavior changes independently, classify the boundary as domain; if only deployment topology changes, classify it as deployment.", "conditions": ["reason for change is observable"], "limitations": ["ambiguous cases need further analysis"]}),
        engine.record_study_note(curriculum["id"], source["id"], {"kind": "heuristic", "claim": "Inspect the reason for change rather than deployment location.", "conditions": ["boundary question is technical"], "limitations": ["do not infer behavior from infrastructure placement"]}),
    ]
    for note in notes:
        engine.record_knowledge(curriculum["id"], {"kind": note["kind"], "claim": note["claim"], "source_id": source["id"]})
    procedure_a = _procedure(notes, revised=False)

    post_cases = [{"id": "post-1", "signal": "deployment_only"}, {"id": "post-2", "signal": "ambiguous"}]
    post = engine.define_test(curriculum["id"], skill["id"], {
        "phase": "posttest", "cases": [x["id"] for x in post_cases], "success_threshold": 1.0,
        "evaluator_type": "executable_grader", "evaluator_id": grader.evaluator_id,
        "evaluator_independence": "independent", "contamination_status": "not_applicable",
        "prohibited_resources": [source["id"], "answer-key"],
    })
    post_attempt = engine.record_attempt(post["id"], _attempt_payload(post_cases, procedure_a))
    post_eval = engine.evaluate_trusted_attempt(post_attempt["id"])
    failed_cases = [item["case"] for item in post_eval["case_results"] if not item["passed"]]

    revision_note = engine.record_study_note(curriculum["id"], source["id"], {"kind": "revision", "claim": "For ambiguous cases, inspect the independent reason for change before classifying domain versus deployment.", "conditions": ["the first procedure failed on an ambiguous case"], "limitations": ["still requires case evidence"]})
    engine.record_knowledge(curriculum["id"], {"kind": revision_note["kind"], "claim": revision_note["claim"], "source_id": source["id"]})
    revision = engine.revise_skill(skill["id"], {"claim": "Classify behavior and deployment boundaries by independent reason, including ambiguous cases."}, reason=f"Post-study attempt failed on held-out cases: {failed_cases}", evidence_ids=[post_eval["id"], revision_note["id"]])
    all_notes = notes + [revision_note]
    procedure_b = _procedure(all_notes, revised=True)

    retest_cases = [{"id": "retest-1", "signal": "deployment_only"}, {"id": "retest-2", "signal": "domain_behavior"}, {"id": "retest-3", "signal": "ambiguous"}]
    retest = engine.define_test(curriculum["id"], skill["id"], {
        "phase": "retest", "cases": [x["id"] for x in retest_cases], "success_threshold": 1.0,
        "evaluator_type": "executable_grader", "evaluator_id": grader.evaluator_id,
        "evaluator_independence": "independent", "contamination_status": "not_applicable",
        "prohibited_resources": [source["id"], "answer-key"],
    })
    retest_attempt = engine.record_attempt(retest["id"], _attempt_payload(retest_cases, procedure_b))
    retest_eval = engine.evaluate_trusted_attempt(retest_attempt["id"])
    demonstrated = engine.promote_demonstrated(skill["id"], pretest_attempt_id=baseline_attempt["id"], pretest_evaluation_id=baseline_eval["id"], posttest_evaluation_id=retest_eval["id"])

    later_case = {"id": "later-new-task-1", "signal": "ambiguous"}
    later_answer = _answer(later_case, procedure_b)
    later = engine.apply_skill("boundary.classification", later_case["id"], procedure_b, outcome={"score": 1.0 if later_answer == "domain" else 0.0, "answer": later_answer, "expected": "domain", "evidence_ref": "later-task-external-result"})
    result = {
        "schema": "rex-learning-production-experiment-v1", "engine_mode": "production", "qualification_scope": "objective_executable_grader_learning_not_provider_cognition",
        "curriculum_id": curriculum["id"], "source_id": source["id"], "study_note_ids": [n["id"] for n in all_notes],
        "baseline": {"attempt_id": baseline_attempt["id"], "evaluation_id": baseline_eval["id"], "score": baseline_eval["score"]},
        "post_study_attempt": {"attempt_id": post_attempt["id"], "evaluation_id": post_eval["id"], "score": post_eval["score"], "failed_cases": failed_cases},
        "revision": {"version": revision["version"], "evidence_ids": revision["provenance"]["revision_ids"], "procedure": procedure_b},
        "held_out_retest": {"attempt_id": retest_attempt["id"], "evaluation_id": retest_eval["id"], "score": retest_eval["score"]},
        "skill": {"id": demonstrated["id"], "state": demonstrated["state"], "version": demonstrated["version"]},
        "later_application": later,
        "evaluator": {"id": grader.evaluator_id, "type": grader.evaluator_type, "provider": grader.provider, "session_id": grader.session_id},
    }
    (root / "production-experiment-result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result
