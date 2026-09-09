from pathlib import Path

import pytest

from rex_learning import LearningError
from rex_learning import ExecutableGrader, LearningEngine, LearningStore
from rex_learning.candidate_skill import (
    attach_review,
    build_candidate_skill,
    promotion_boundary,
    validate_review_decision,
)


def _case():
    return {
        "id": "v5-01",
        "source": {"title": "Contrapositive", "text": "If P implies Q, not Q implies not P.", "source_refs": ["v5://v5-01"]},
        "intent": {"key": "cap-01", "target_capability": "Reason about logical implications"},
    }


def _surface():
    return {
        "surface_schema": "rex-learning-integrated-final-surface-v3",
        "source_invariant": {"claim": "Preserve the contrapositive rule"},
        "transferable_abstractions": ["Construct contrapositives within logical implications."],
        "applicability": ["logical implications"],
        "learning_action": ["practice constructing a contrapositive"],
        "evidence_path": ["independent construction exercise"],
    }


def _provenance():
    return {"provider": "qwen-local", "session_id": "qwen-test-1", "response_digest": "abc", "role": "candidate_generator"}


def test_candidate_is_untrusted_and_retains_source_interpretation():
    package = build_candidate_skill(case=_case(), qwen_surface=_surface(), provenance=_provenance())
    assert package["state"] == "EXTRACTED"
    assert package["trusted"] is False
    assert package["promoted"] is False
    assert package["evidence"][0]["refs"] == ["v5://v5-01"]
    assert package["qwen_interpretation"]["surface_schema"].endswith("v3")


def test_procedure_only_surface_uses_immutable_target_as_claim():
    package = build_candidate_skill(
        case=_case(),
        qwen_surface={"procedure": "1. Inspect the input. 2. Apply the rule."},
        provenance=_provenance(),
    )
    assert package["proposed_skill"]["claim"] == "Reason about logical implications"
    assert package["proposed_skill"]["operational_procedure"] == "1. Inspect the input. 2. Apply the rule."
    assert package["trusted"] is False
    assert package["promoted"] is False


def test_review_identity_and_provenance_are_required():
    package = build_candidate_skill(case=_case(), qwen_surface=_surface(), provenance=_provenance())
    decision = {"candidate_id": package["candidate_id"], "decision": "PROMOTE_WITH_NARROWER_SCOPE", "rationale": "Core rule is supported; scope is narrowed.", "accepted_claims": ["Preserve the contrapositive rule"], "rejected_claims": [], "required_changes": [], "source_refs": ["v5://v5-01"], "scope_limits": ["logical implications"], "reviewed_surface": {"surface_schema": "rex-learning-integrated-final-surface-v3", "source_invariant": "Preserve the contrapositive rule", "operational_procedure": "Construct the contrapositive and verify it."}}
    reviewed = attach_review(package, decision, reviewer={"provider": "openai-codex", "session_id": "review-1", "model": "gpt-5.6-luna"})
    assert reviewed["state"] == "APPROVED_WITH_LIMITS"
    assert reviewed["trusted"] is True
    assert reviewed["promoted"] is False
    assert reviewed["reviewed_surface"]["source_invariant"] == "Preserve the contrapositive rule"

    with pytest.raises(LearningError, match="identity"):
        validate_review_decision({**decision, "candidate_id": "other"}, candidate_id=package["candidate_id"])


def test_review_does_not_bypass_behavioral_promotion_boundary():
    package = build_candidate_skill(case=_case(), qwen_surface=_surface(), provenance=_provenance())
    decision = {"candidate_id": package["candidate_id"], "decision": "PROMOTE", "rationale": "Supported candidate.", "accepted_claims": ["rule"], "rejected_claims": [], "required_changes": [], "source_refs": ["v5://v5-01"], "scope_limits": [], "reviewed_surface": {"claim": "rule", "operational_procedure": "Apply the rule."}}
    reviewed = attach_review(package, decision, reviewer={"provider": "openai-codex", "session_id": "review-1"})
    result = promotion_boundary(reviewed, behavioral_evidence={"source_free": True})
    assert result["promoted"] is False
    assert result["reason"] == "novel_transfer_missing"


def test_promotable_review_must_bind_the_durable_surface():
    package = build_candidate_skill(case=_case(), qwen_surface={**_surface(), "operational_procedure": "Unreviewed provider procedure."}, provenance=_provenance())
    decision = {"candidate_id": package["candidate_id"], "decision": "PROMOTE", "rationale": "Supported candidate.", "accepted_claims": ["rule"], "rejected_claims": [], "required_changes": [], "source_refs": ["v5://v5-01"], "scope_limits": []}
    with pytest.raises(LearningError, match="reviewed_surface"):
        attach_review(package, decision, reviewer={"provider": "reviewer", "session_id": "review-1"})


def test_invalid_reviewer_decision_is_rejected():
    with pytest.raises(LearningError, match="invalid"):
        validate_review_decision({"candidate_id": "c", "decision": "APPROVE", "rationale": "x"}, candidate_id="c")


def test_reviewed_candidate_can_be_promoted_into_durable_runtime_skill(tmp_path):
    grader = ExecutableGrader(
        evaluator_id="grader.candidate.v1",
        answer_key={"pre": "wrong", "post": "right"},
        provider="local-grader",
        session_id="grader-session",
    )
    engine = LearningEngine(LearningStore(tmp_path / "learning"), trusted_evaluators={grader.evaluator_id: grader})
    curriculum = engine.create_curriculum("Candidate promotion", "Promote a reviewed procedure")
    case = {**_case(), "intent": {**_case()["intent"], "key": "candidate-procedure"}}
    surface = {**_surface(), "operational_procedure": "Apply the rule, then verify the result."}
    package = build_candidate_skill(case=case, qwen_surface=surface, provenance=_provenance())
    decision = {"candidate_id": package["candidate_id"], "decision": "PROMOTE", "rationale": "Supported and behaviorally demonstrated.", "accepted_claims": ["rule"], "rejected_claims": [], "required_changes": [], "source_refs": ["v5://v5-01"], "scope_limits": [], "reviewed_surface": {"claim": "rule", "operational_procedure": surface["operational_procedure"]}}
    reviewed = attach_review(package, decision, reviewer={"provider": "reviewer", "session_id": "review-1"})
    skill = engine.create_skill_hypothesis(curriculum["id"], {"key": "candidate-procedure", "claim": "rule", "applicability": ["logical implications"], "operational_procedure": surface["operational_procedure"]})
    pre = engine.define_test(curriculum["id"], skill["id"], {"phase": "pretest", "cases": ["pre"], "evaluator_type": "executable_grader", "evaluator_id": grader.evaluator_id, "success_threshold": 1.0, "evaluator_independence": "independent", "contamination_status": "not_applicable"})
    pre_attempt = engine.record_attempt(pre["id"], {"responses": [{"case": "pre", "answer": "right"}]})
    pre_eval = engine.evaluate_trusted_attempt(pre_attempt["id"])
    post = engine.define_test(curriculum["id"], skill["id"], {"phase": "posttest", "cases": ["post"], "evaluator_type": "executable_grader", "evaluator_id": grader.evaluator_id, "success_threshold": 1.0, "evaluator_independence": "independent", "contamination_status": "not_applicable"})
    post_attempt = engine.record_attempt(post["id"], {"responses": [{"case": "post", "answer": "right"}]})
    post_eval = engine.evaluate_trusted_attempt(post_attempt["id"])
    promoted = engine.promote_candidate_skill(reviewed, curriculum_id=curriculum["id"], pretest_attempt_id=pre_attempt["id"], pretest_evaluation_id=pre_eval["id"], posttest_evaluation_id=post_eval["id"], behavioral_evidence={"source_free": True, "novel_transfer": True, "negative_case": True, "reproducible": True, "delayed_retention": True, "contamination_status": "clean", "evaluator_independence": "independent"})
    assert promoted["state"] == "demonstrated"
    fresh = LearningEngine(LearningStore(tmp_path / "learning"), trusted_evaluators={grader.evaluator_id: grader})
    selected = fresh.discover_applicable_skills({"requirements": ["logical implications"]})
    assert selected[0]["skill_id"] == promoted["id"]
    application = fresh.apply_skill(selected[0]["key"], "fresh-task-1", selected[0]["operational_context"]["procedure"], outcome={"passed": True}, curriculum_id=curriculum["id"], selection_reason=selected[0]["why_selected"], operational_context=selected[0]["operational_context"])
    assert application["skill_id"] == promoted["id"]


def test_candidate_promotion_evidence_is_derived_from_recorded_tests(tmp_path: Path) -> None:
    grader = ExecutableGrader(
        evaluator_id="grader.derived-evidence.v1",
        answer_key={"pre": "learned", "post": "right", "retest": "right", "negative": "not-applicable"},
        provider="local-grader",
        session_id="derived-evidence-session",
    )
    engine = LearningEngine(LearningStore(tmp_path / "learning"), trusted_evaluators={grader.evaluator_id: grader})
    curriculum = engine.create_curriculum("Derived evidence", "Derive promotion evidence from records")
    skill = engine.create_skill_hypothesis(curriculum["id"], {"key": "derived", "claim": "Use the procedure", "operational_procedure": "Use the procedure."})
    common = {"evaluator_type": "executable_grader", "evaluator_id": grader.evaluator_id, "success_threshold": 1.0, "evaluator_independence": "independent", "contamination_status": "clean", "source_free": True, "novel_transfer": True}

    def evaluate(phase: str, case: str, answer: str) -> dict:
        test = engine.define_test(curriculum["id"], skill["id"], {"phase": phase, "cases": [case], **common})
        attempt = engine.record_attempt(test["id"], {"responses": [{"case": case, "answer": answer}]})
        return engine.evaluate_trusted_attempt(attempt["id"])

    pre_test = engine.define_test(curriculum["id"], skill["id"], {"phase": "pretest", "cases": ["pre"], **common})
    pre_attempt = engine.record_attempt(pre_test["id"], {"responses": [{"case": "pre", "answer": "wrong"}]})
    pre_eval = engine.evaluate_trusted_attempt(pre_attempt["id"])
    post_eval = evaluate("posttest", "post", "right")
    retest_eval = evaluate("retest", "retest", "right")
    negative_eval = evaluate("negative", "negative", "not-applicable")

    evidence = engine.derive_candidate_behavioral_evidence(
        pretest_attempt_id=pre_attempt["id"],
        pretest_evaluation_id=pre_eval["id"],
        posttest_evaluation_id=post_eval["id"],
        retest_evaluation_id=retest_eval["id"],
        negative_evaluation_id=negative_eval["id"],
    )

    assert evidence == {
        "source_free": True,
        "novel_transfer": True,
        "negative_case": True,
        "reproducible": True,
        "delayed_retention": True,
        "contamination_status": "clean",
        "evaluator_independence": "independent",
    }