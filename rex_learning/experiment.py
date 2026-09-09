from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .engine import LearningEngine
from .store import LearningStore


def _score(method: str, cases: list[dict[str, Any]]) -> tuple[float, list[dict[str, Any]]]:
    results = []
    for case in cases:
        if method == "A":
            passed = case["independent_changes"] == 1 and case["deployment_only"] is False
        else:
            passed = (case["independent_changes"] == 1 and case["domain_boundary"] is True and case["deployment_only"] is False) or (case["deployment_only"] is True and case["domain_boundary"] is False)
        results.append({"case": case["id"], "passed": passed})
    return sum(item["passed"] for item in results) / len(results), results


def run_bootstrap_experiment(root: Path) -> dict[str, Any]:
    """Run a small deterministic qualification experiment, not a claim about model cognition.

    The fixture deliberately separates source, baseline, practice, held-out tests, and
    later application. It qualifies the evidence machinery and the learning procedure
    contract; a provider-backed curriculum remains a future physical qualification.
    """
    engine = LearningEngine(LearningStore(root), allow_fixture=True)
    curriculum = engine.create_curriculum("Learning procedure bootstrap", "Convert instructional material into transferable operational competence", method_version="B")
    source_a = engine.add_source(curriculum["id"], "Boundary reasoning primer", "fixture://boundary-primer", kind="markdown")
    source_b = engine.add_source(curriculum["id"], "Negative cases primer", "fixture://boundary-negative-cases", kind="markdown")
    for source, note in ((source_a, {"kind": "procedure", "claim": "Inspect independent reasons to change before placing a boundary.", "limitations": ["not a deployment-boundary rule"]}), (source_b, {"kind": "counterexample", "claim": "A deployment split is not necessarily a domain boundary.", "conditions": ["deployment_only case"]})):
        engine.record_study_note(curriculum["id"], source["id"], note)
        engine.record_knowledge(curriculum["id"], {"kind": note["kind"], "claim": note["claim"], "source_id": source["id"]})
    engine.record_knowledge(curriculum["id"], {"kind": "heuristic", "claim": "avoid premature abstraction", "source_id": source_a["id"]})
    engine.record_knowledge(curriculum["id"], {"kind": "heuristic", "claim": "establish explicit boundaries early", "source_id": source_b["id"]})
    skill = engine.create_skill_hypothesis(curriculum["id"], {"key": "architecture.boundary_reasoning", "kind": "procedure", "claim": "Diagnose independent change axes and reject deployment-only boundaries.", "applicability": ["novel decomposition cases"], "failure_criteria": ["blindly apply deployment split"], "success_criteria": ["identify independent change axes", "reject deployment-only boundary"]})
    cases = [
        {"id": "pre-1", "independent_changes": 1, "domain_boundary": True, "deployment_only": False},
        {"id": "pre-2", "independent_changes": 0, "domain_boundary": False, "deployment_only": True},
        {"id": "pre-3", "independent_changes": 1, "domain_boundary": False, "deployment_only": True},
    ]
    score_a, results_a = _score("A", cases)
    expected_cases = [{"case": x["id"], "passed": True} for x in cases]
    pre = engine.define_test(curriculum["id"], skill["id"], {"phase": "pretest", "cases": [x["id"] for x in cases], "expected_case_results": expected_cases, "success_threshold": 0.66, "evaluator_type": "deterministic_test", "evaluator_independence": "independent", "contamination_status": "clean", "prohibited_resources": [source_a["id"], source_b["id"]]})
    pre_attempt = engine.record_attempt(pre["id"], {"score": score_a, "case_results": results_a})
    pre_eval = engine.evaluate_attempt(pre_attempt["id"], {"score": score_a, "passed": score_a >= 0.66, "validation_status": "verified", "evaluator_type": "deterministic_test", "independence": "independent", "contamination": {"status": "clean"}, "evidence": [{"kind": "fixture_assertion", "ref": item["case"]} for item in results_a]})
    practice_score, practice_results = _score("B", cases)
    practice = engine.define_test(curriculum["id"], skill["id"], {"phase": "practice", "cases": [x["id"] for x in cases], "expected_case_results": expected_cases, "success_threshold": 1.0, "evaluator_type": "deterministic_test", "evaluator_independence": "independent"})
    practice_attempt = engine.record_attempt(practice["id"], {"score": practice_score, "case_results": practice_results})
    practice_eval = engine.evaluate_attempt(practice_attempt["id"], {"score": practice_score, "passed": practice_score >= 1.0, "validation_status": "verified", "evaluator_type": "deterministic_test", "independence": "independent", "contamination": {"status": "clean"}, "evidence": [{"kind": "fixture_assertion", "ref": item["case"]} for item in practice_results]})
    engine.revise_skill(skill["id"], {"claim": "Diagnose change axes, test boundary type, and reject deployment-only boundaries."}, reason="Practice exposed confusion between deployment and domain boundaries.", evidence_ids=[practice_eval["id"]])
    held_out = [{"id": "post-1", "independent_changes": 1, "domain_boundary": True, "deployment_only": False}, {"id": "post-2", "independent_changes": 0, "domain_boundary": False, "deployment_only": True}, {"id": "post-3", "independent_changes": 1, "domain_boundary": True, "deployment_only": False}]
    score_post, results_post = _score("B", held_out)
    expected_held_out = [{"case": x["id"], "passed": True} for x in held_out]
    post = engine.define_test(curriculum["id"], skill["id"], {"phase": "retest", "cases": [x["id"] for x in held_out], "expected_case_results": expected_held_out, "success_threshold": 1.0, "evaluator_type": "deterministic_test", "evaluator_independence": "independent", "contamination_status": "clean", "prohibited_resources": [source_a["id"], source_b["id"]]})
    post_attempt = engine.record_attempt(post["id"], {"score": score_post, "case_results": results_post})
    post_eval = engine.evaluate_attempt(post_attempt["id"], {"score": score_post, "passed": score_post >= 1.0, "validation_status": "verified", "evaluator_type": "deterministic_test", "independence": "independent", "contamination": {"status": "clean"}, "evidence": [{"kind": "fixture_assertion", "ref": item["case"]} for item in results_post]})
    demonstrated = engine.promote_demonstrated(skill["id"], pretest_attempt_id=pre_attempt["id"], posttest_evaluation_id=post_eval["id"])
    application = engine.apply_skill("architecture.boundary_reasoning", "later-novel-task-1", "Check change axes, boundary type, and negative applicability before recommending decomposition.", outcome={"score": 1.0, "evidence_ref": "fixture://later-novel-task-1"})
    engine.set_curriculum_status(curriculum["id"], "completed", reason="Controlled fixture lifecycle completed")
    result = {"schema": "rex-learning-bootstrap-experiment-v1", "qualification_scope": "deterministic_framework_fixture_not_provider_cognition", "curriculum_id": curriculum["id"], "method_A": {"score": score_a, "pretest": pre_eval["id"]}, "practice": {"score": practice_score, "evaluation": practice_eval["id"]}, "revision": {"reason": "deployment/domain confusion", "evidence": practice_eval["id"]}, "held_out_retest": {"score": score_post, "evaluation": post_eval["id"]}, "skill": {"id": demonstrated["id"], "key": demonstrated["key"], "version": demonstrated["version"], "state": demonstrated["state"]}, "later_application": {"id": application["id"], "skill_version": application["skill_version"], "outcome": application["outcome"]}, "before_after_delta": score_post - score_a, "limitations": ["deterministic fixture does not prove Qwen/Gemma cognition", "provider-backed source ingestion and real-world outcome evaluation remain for physical qualification"]}
    (root / "experiment-result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result
