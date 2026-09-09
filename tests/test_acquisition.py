from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, cast

import pytest

from rex_learning import ExecutableGrader, LearningEngine, LearningError, LearningStore, TrustedEvaluator, learn_from_source
from rex_learning.acquisition import NonDiscriminatingControlError, _practice_revision_feedback
from rex_learning.engine import CeilingBaselineError


class SyntheticLearner:
    provider = "synthetic-provider"
    session_id = "synthetic-session-1"

    def answer(self, *, task: str, material: str = "", revision: str = "") -> dict:
        learned = "vex(a,b)" in material or "vex(a,b)" in revision or "vex(a,b)" in task
        case = next((part.split("=", 1)[1] for part in task.split("|") if part.startswith("case=")), task.split("|")[0])
        if "candidate" in task:
            assert "case=candidate" in task
            return {
                "responses": [],
                "surface_schema": "synthetic-skill-v1",
                "key": "vex-coordinate-capability",
                "claim": "Use vex with the first coordinate doubled and the second preserved.",
                "transferable_abstractions": ["Apply vex(a,b) as (2*a,b)."],
                "applicability": ["vex coordinate transformation"],
                "operational_procedure": "Parse vex(a,b), return (2*a,b), and reject non-vex expressions.",
                "preconditions": ["Input uses vex(a,b)."],
                "limitations": ["Does not apply to other operators."],
            }
        if "negative" in task:
            answer = "not-applicable" if learned else "vex(0,0)"
        elif "posttest" in task or "retest" in task or "practice" in task or "fresh" in task:
            answer = "vex(14,3)=(28,3)" if learned else "unknown"
        else:
            answer = "unknown"
        return {"responses": [{"case": case, "answer": answer}]}


def _explicit_reviewer(candidate: Mapping[str, Any], evidence: Mapping[str, Any]) -> dict[str, Any]:
    proposed = candidate["proposed_skill"]
    return {
        "candidate_id": candidate["candidate_id"],
        "decision": "PROMOTE",
        "rationale": "Synthetic reviewer explicitly bound the durable surface.",
        "accepted_claims": [str(proposed["claim"])],
        "rejected_claims": [],
        "required_changes": [],
        "source_refs": list(evidence["source"]["source_refs"]),
        "scope_limits": [],
        "reviewed_surface": {
            "claim": proposed["claim"],
            "operational_procedure": proposed["operational_procedure"],
        },
    }


class RevisionLearner(SyntheticLearner):
    def __init__(self) -> None:
        self.candidate_calls = 0

    def answer(self, *, task: str, material: str = "", revision: str = "") -> dict:
        if "candidate" in task:
            self.candidate_calls += 1
            if self.candidate_calls == 1:
                return {
                    "responses": [],
                    "capability": "Transform vex coordinates",
                    "applicability": ["vex coordinate transformation"],
                    "procedure": "Return the input unchanged.",
                    "preconditions": ["Input uses vex(a,b)."],
                    "limitations": ["Not for other operators."],
                }
            return {
                "responses": [],
                "capability": "Transform vex coordinates by doubling the first coordinate",
                "applicability": ["vex coordinate transformation"],
                "procedure": "Parse vex(a,b) and return vex(2*a,b).",
                "preconditions": ["Input uses vex(a,b)."],
                "limitations": ["Not for other operators."],
            }
        if "practice" in task and not revision:
            return {"responses": [{"case": "practice", "answer": "vex(14,3)"}]}
        return super().answer(task=task, material=material, revision=revision)


class CeilingLearner(SyntheticLearner):
    def __init__(self) -> None:
        self.candidate_calls = 0

    def answer(self, *, task: str, material: str = "", revision: str = "") -> dict:
        if "candidate" in task:
            self.candidate_calls += 1
            raise AssertionError("ceiling baseline must not enter candidate generation")
        return super().answer(task=task, material=material, revision=revision)


class SolvedControlLearner(SyntheticLearner):
    def answer(self, *, task: str, material: str = "", revision: str = "") -> dict:
        if task.startswith("control|"):
            return {"responses": [{"case": "control", "answer": "vex(14,3)=(28,3)"}]}
        return super().answer(task=task, material=material, revision=revision)


class BaselineReuseLearner(SyntheticLearner):
    def __init__(self) -> None:
        self.baseline_calls = 0
        self.candidate_calls = 0

    def answer(self, *, task: str, material: str = "", revision: str = "") -> dict:
        if task.startswith("pre|"):
            self.baseline_calls += 1
        if task.startswith("candidate|"):
            self.candidate_calls += 1
            raise AssertionError("candidate generation reached after baseline reuse")
        return super().answer(task=task, material=material, revision=revision)


def test_learn_from_source_runs_a_recorded_source_to_runtime_cycle(tmp_path: Path) -> None:
    grader = ExecutableGrader(
        evaluator_id="grader.synthetic-acquisition.v1",
        answer_key={
            "pre": "vex(14,3)=(28,3)",
            "practice": "vex(14,3)=(28,3)",
            "post": "vex(14,3)=(28,3)",
            "retest": "vex(14,3)=(28,3)",
            "negative": "not-applicable",
            "fresh": "vex(14,3)=(28,3)",
            "control": "vex(14,3)=(28,3)",
        },
        provider="independent-local-grader",
        session_id="grader-synthetic-acquisition-1",
    )
    engine = LearningEngine(LearningStore(tmp_path / "learning"), trusted_evaluators={grader.evaluator_id: grader})
    result = learn_from_source(
        engine=engine,
        learner=SyntheticLearner(),
        source_title="Vex coordinate protocol",
        source_text="The invented operator vex(a,b) returns (2*a,b). It applies only to vex expressions.",
        intent_key="vex-coordinate-capability",
        target_capability="Transform vex coordinate expressions",
        grader=grader,
        reviewer=_explicit_reviewer,
        fresh_learner=SyntheticLearner(),
    )

    assert result["schema"] == "rex-learning-acquisition-run-v1"
    assert result["candidate"]["provenance"]["provider"] == "synthetic-provider"
    assert result["candidate"]["trusted"] is True
    assert result["promotion"]["state"] == "demonstrated"
    assert result["evidence"]["source_free"] is True
    assert result["fresh_context"]["selected_skill_id"] == result["promotion"]["id"]
    assert result["fresh_context"]["application"]["outcome"]["passed"] is True
    assert result["fresh_context"]["application"]["validation_status"] == "verified"
    assert result["control"]["score"] == 0.0
    assert result["negative_applicability"] is False
    assert result["artifacts"]["source_hash"]
    assert engine.store.read("acquisition_runs", result["run_id"])["promotion"]["id"] == result["promotion"]["id"]


def test_learn_from_source_accepts_persisted_manifest_hash_for_reconstructed_material(tmp_path: Path) -> None:
    grader = ExecutableGrader(
        evaluator_id="grader.manifest-source.v1",
        answer_key={
            "pre": "vex(14,3)=(28,3)", "practice": "vex(14,3)=(28,3)",
            "post": "vex(14,3)=(28,3)", "retest": "vex(14,3)=(28,3)",
            "negative": "not-applicable", "fresh": "vex(14,3)=(28,3)",
            "control": "vex(14,3)=(28,3)",
        },
        provider="independent-local-grader", session_id="grader-manifest-source-1",
    )
    engine = LearningEngine(LearningStore(tmp_path / "learning"), trusted_evaluators=cast(dict[str, TrustedEvaluator], {grader.evaluator_id: grader}))
    curriculum = engine.create_curriculum("Manifest source", "Acquire vex transformation")
    material = "The invented operator vex(a,b) returns (2*a,b)."
    manifest_hash = hashlib.sha256(b"chapter.xhtml\x00original-directory-bytes").hexdigest()
    source = engine.add_source(curriculum["id"], "Manifest source", str(tmp_path / "source"), content_hash=manifest_hash)

    result = learn_from_source(
        engine=engine, learner=SyntheticLearner(), source_title="Manifest source",
        source_text=material, intent_key="manifest-vex", target_capability="Transform vex coordinate expressions",
        grader=grader, reviewer=_explicit_reviewer, fresh_learner=SyntheticLearner(),
        curriculum_id=curriculum["id"], source_id=source["id"], source_content_hash=manifest_hash,
    )

    assert result["source"]["id"] == source["id"]
    assert result["artifacts"]["source_hash"] == manifest_hash

    with pytest.raises(LearningError, match="material content hash mismatch"):
        learn_from_source(
            engine=engine, learner=SyntheticLearner(), source_title="Manifest source",
            source_text=material, intent_key="manifest-vex-mismatch", target_capability="Transform vex coordinate expressions",
            grader=grader, reviewer=_explicit_reviewer, fresh_learner=SyntheticLearner(),
            curriculum_id=curriculum["id"], source_id=source["id"], source_content_hash=manifest_hash,
            material_content_hash="0" * 64,
        )


def test_solved_supplied_control_is_rejected_before_candidate_generation(tmp_path: Path) -> None:
    grader = ExecutableGrader(
        evaluator_id="grader.synthetic-control-ceiling.v1",
        answer_key={case: "vex(14,3)=(28,3)" for case in ("pre", "control")},
        provider="independent-local-grader", session_id="grader-synthetic-control-ceiling-1",
    )
    learner = CeilingLearner()
    engine = LearningEngine(LearningStore(tmp_path / "learning"), trusted_evaluators={grader.evaluator_id: grader})
    with pytest.raises(NonDiscriminatingControlError):
        learn_from_source(
            engine=engine, learner=learner, source_title="Vex protocol",
            source_text="The invented operator vex(a,b) returns (2*a,b).",
            intent_key="vex-control-ceiling", target_capability="Transform vex coordinate expressions",
            grader=grader, reviewer=_explicit_reviewer, fresh_learner=SyntheticLearner(),
            control_learner=SolvedControlLearner(),
        )
    assert learner.candidate_calls == 0


def test_ceiling_baseline_is_persisted_before_candidate_generation(tmp_path: Path) -> None:
    grader = ExecutableGrader(
        evaluator_id="grader.synthetic-ceiling.v1",
        answer_key={"pre": "unknown"},
        provider="independent-local-grader",
        session_id="grader-synthetic-ceiling-1",
    )
    learner = CeilingLearner()
    engine = LearningEngine(LearningStore(tmp_path / "learning"), trusted_evaluators=cast(dict[str, TrustedEvaluator], {grader.evaluator_id: grader}))

    with pytest.raises(CeilingBaselineError) as caught:
        learn_from_source(
            engine=engine,
            learner=learner,
            source_title="Vex coordinate protocol",
            source_text="The invented operator vex(a,b) returns (2*a,b).",
            intent_key="vex-ceiling",
            target_capability="Transform vex coordinate expressions",
            grader=cast(TrustedEvaluator, grader),
            reviewer=_explicit_reviewer,
            fresh_learner=SyntheticLearner(),
        )

    assert learner.candidate_calls == 0
    preflights = engine.store.list("baseline_preflights")
    assert len(preflights) == 1
    assert preflights[0]["score"] == 1.0
    assert preflights[0]["passed"] is True
    assert caught.value.diagnostic()["baseline_preflight_id"] == preflights[0]["id"]


def test_acquisition_reuses_persisted_baseline_without_rerunning_learner(tmp_path: Path) -> None:
    grader = ExecutableGrader(
        evaluator_id="grader.synthetic-reuse.v1", answer_key={"pre": "vex(14,3)=(28,3)"},
        provider="independent-local-grader", session_id="grader-synthetic-reuse-1",
    )
    engine = LearningEngine(LearningStore(tmp_path / "learning"), trusted_evaluators=cast(dict[str, TrustedEvaluator], {grader.evaluator_id: grader}))
    first = BaselineReuseLearner()
    with pytest.raises(AssertionError, match="candidate generation"):
        learn_from_source(
            engine=engine, learner=first, source_title="Vex coordinate protocol",
            source_text="The invented operator vex(a,b) returns (2*a,b).",
            intent_key="vex-reuse", target_capability="Transform vex coordinate expressions",
            grader=cast(TrustedEvaluator, grader), reviewer=_explicit_reviewer, fresh_learner=SyntheticLearner(),
        )
    preflight = engine.store.list("baseline_preflights")[0]
    curriculum = engine.store.list("curricula")[0]
    source = engine.store.list("sources")[0]
    assert first.baseline_calls == 1

    resumed = BaselineReuseLearner()
    with pytest.raises(AssertionError, match="candidate generation"):
        learn_from_source(
            engine=engine, learner=resumed, source_title="Vex coordinate protocol",
            source_text="The invented operator vex(a,b) returns (2*a,b).",
            intent_key="vex-reuse", target_capability="Transform vex coordinate expressions",
            grader=grader, reviewer=_explicit_reviewer, fresh_learner=SyntheticLearner(),
            curriculum_id=curriculum["id"], source_id=source["id"],
            baseline_preflight_id=preflight["id"],
        )
    assert resumed.baseline_calls == 0
    assert resumed.candidate_calls == 1


def test_baseline_reuse_rejects_mismatched_context_before_learner_call(tmp_path: Path) -> None:
    grader = ExecutableGrader(
        evaluator_id="grader.synthetic-reuse-mismatch.v1", answer_key={"pre": "vex(14,3)=(28,3)"},
        provider="independent-local-grader", session_id="grader-synthetic-reuse-mismatch-1",
    )
    engine = LearningEngine(LearningStore(tmp_path / "learning"), trusted_evaluators={grader.evaluator_id: grader})
    first = BaselineReuseLearner()
    with pytest.raises(AssertionError):
        learn_from_source(
            engine=engine, learner=first, source_title="Vex coordinate protocol",
            source_text="The invented operator vex(a,b) returns (2*a,b).",
            intent_key="vex-reuse-mismatch", target_capability="Transform vex coordinate expressions",
            grader=grader, reviewer=_explicit_reviewer, fresh_learner=SyntheticLearner(),
        )
    preflight = engine.store.list("baseline_preflights")[0]
    curriculum = engine.store.list("curricula")[0]
    source = engine.store.list("sources")[0]
    learner = BaselineReuseLearner()
    with pytest.raises(LearningError, match="does not match acquisition context"):
        learn_from_source(
            engine=engine, learner=learner, source_title="Vex coordinate protocol",
            source_text="The invented operator vex(a,b) returns (2*a,b).",
            intent_key="vex-reuse-mismatch", target_capability="Different capability",
            grader=grader, reviewer=_explicit_reviewer, fresh_learner=SyntheticLearner(),
            curriculum_id=curriculum["id"], source_id=source["id"], baseline_preflight_id=preflight["id"],
        )
    assert learner.baseline_calls == 0
    assert learner.candidate_calls == 0


def test_acquisition_revises_candidate_after_practice_failure_and_preserves_failure(tmp_path: Path) -> None:
    grader = ExecutableGrader(
        evaluator_id="grader.synthetic-revision.v1",
        answer_key={
            "pre": "vex(14,3)=(28,3)",
            "practice": "vex(14,3)=(28,3)",
            "post": "vex(14,3)=(28,3)",
            "retest": "vex(14,3)=(28,3)",
            "negative": "not-applicable",
            "fresh": "vex(14,3)=(28,3)",
            "control": "vex(14,3)=(28,3)",
        },
        provider="independent-local-grader",
        session_id="grader-synthetic-revision-1",
    )
    learner = RevisionLearner()
    engine = LearningEngine(LearningStore(tmp_path / "learning"), trusted_evaluators={grader.evaluator_id: grader})

    result = learn_from_source(
        engine=engine,
        learner=learner,
        source_title="Vex coordinate protocol",
        source_text="The invented operator vex(a,b) returns (2*a,b). It applies only to vex expressions.",
        intent_key="vex-coordinate-revision",
        target_capability="Transform vex coordinate expressions",
        grader=grader,
        reviewer=_explicit_reviewer,
        fresh_learner=SyntheticLearner(),
    )

    assert result["practice"]["revision"]
    assert result["practice"]["failed_evaluation"]["passed"] is False
    assert result["practice"]["evaluation"]["passed"] is True
    assert result["candidate"]["revision_history"][0]["failed_evaluation_id"] == result["practice"]["failed_evaluation"]["id"]
    assert result["promotion"]["operational_procedure"] == "Parse vex(a,b) and return vex(2*a,b)."
    assert result["fresh_context"]["application"]["outcome"]["passed"] is True


def test_practice_revision_feedback_preserves_diagnostics_without_answer_scaffolding() -> None:
    feedback = _practice_revision_feedback(
        {
            "case_results": [
                {"case": "practice", "passed": False, "failed_fields": ["preserves_existing_mapping"], "rationale": "Check state after denial."},
            ],
            "expected_case_results": [{"case": "practice", "passed": True}],
        },
        1,
        skill_kind="procedure",
        target_capability="Preserve mappings during bounded installation",
    )

    assert "preserves_existing_mapping" in feedback
    assert "Check state after denial." in feedback
    assert "expected_case_results" not in feedback
    assert "answer_key" not in feedback
    assert "state changes" in feedback
    assert "observable checks" in feedback


def test_practice_revision_feedback_preserves_diagnostic_and_failed_assertions() -> None:
    feedback = _practice_revision_feedback(
        {
            "case_results": [{
                "case": "practice",
                "diagnosis": "The state-preservation invariant was omitted.",
                "failed_assertions": ["$.state_preserved must be true"],
            }]
        },
        1,
        skill_kind="procedure",
        target_capability="Preserve state across rejected operations",
    )

    assert "state-preservation invariant was omitted" in feedback
    assert "$.state_preserved must be true" in feedback
    assert "answer_key" not in feedback
    assert "expected_case_results" not in feedback


@pytest.mark.parametrize(
    ("skill_kind", "required_terms"),
    [
        ("workflow", ("decision points", "failure handling")),
        ("declarative", ("distinguishing conditions", "counterexample")),
    ],
)
def test_practice_revision_feedback_selects_capability_type_treatment(
    skill_kind: str, required_terms: tuple[str, str]
) -> None:
    feedback = _practice_revision_feedback(
        {"case_results": [{"case": "practice", "missing_propositions": ["boundary"]}]},
        1,
        skill_kind=skill_kind,
        target_capability="A bounded capability",
    )

    assert all(term in feedback for term in required_terms)
    assert "answer_key" not in feedback


class MethodReviewer:
    provenance = {"provider": "method-reviewer", "session_id": "review-method-1"}

    def review(self, *, candidate: dict, evidence: dict) -> dict:
        return {
            "candidate_id": candidate["candidate_id"],
            "decision": "PROMOTE",
            "rationale": "The candidate is source-grounded; behavioral evidence remains authoritative.",
            "accepted_claims": ["vex capability"],
            "rejected_claims": [],
            "required_changes": [],
            "source_refs": list(evidence["source"]["source_refs"]),
            "scope_limits": [],
            "reviewed_surface": {"claim": "vex capability", "operational_procedure": "Apply vex(a,b) as (2*a,b)."},
        }


def _reviewer_test_run(tmp_path: Path, reviewer: Any) -> dict:
    grader = ExecutableGrader(
        evaluator_id="grader.synthetic-reviewer-injection.v1",
        answer_key={
            "pre": "vex(14,3)=(28,3)",
            "practice": "vex(14,3)=(28,3)",
            "post": "vex(14,3)=(28,3)",
            "retest": "vex(14,3)=(28,3)",
            "negative": "not-applicable",
            "fresh": "vex(14,3)=(28,3)",
            "control": "vex(14,3)=(28,3)",
        },
        provider="independent-local-grader",
        session_id="grader-reviewer-injection-1",
    )
    engine = LearningEngine(LearningStore(tmp_path / "learning"), trusted_evaluators=cast(dict[str, TrustedEvaluator], {grader.evaluator_id: grader}))
    return learn_from_source(
        engine=engine,
        learner=SyntheticLearner(),
        source_title="Vex coordinate protocol",
        source_text="The invented operator vex(a,b) returns (2*a,b). It applies only to vex expressions.",
        intent_key="vex-reviewer-injection",
        target_capability="Transform vex coordinate expressions",
        grader=grader,
        reviewer=reviewer,
        fresh_learner=SyntheticLearner(),
    )


def test_callable_reviewer_invocation_and_provenance_are_persisted(tmp_path: Path) -> None:
    calls: list[str] = []

    def reviewer(candidate: dict, evidence: dict) -> dict:
        calls.append(candidate["candidate_id"])
        return {
            "candidate_id": candidate["candidate_id"],
            "decision": "PROMOTE",
            "rationale": "Callable reviewer accepted the bounded candidate.",
            "accepted_claims": ["vex capability"],
            "rejected_claims": [],
            "required_changes": [],
            "source_refs": list(evidence["source"]["source_refs"]),
            "scope_limits": [],
            "reviewed_surface": {"claim": "vex capability", "operational_procedure": "Apply vex(a,b) as (2*a,b)."},
        }

    result = _reviewer_test_run(tmp_path, reviewer)
    assert calls == [result["candidate"]["candidate_id"]]
    assert result["candidate"]["review"]["reviewer"]["invocation"] == "callable"
    assert result["promotion"]["state"] == "demonstrated"


def test_review_method_invocation_and_declared_provenance_are_persisted(tmp_path: Path) -> None:
    result = _reviewer_test_run(tmp_path, MethodReviewer())
    reviewer = result["candidate"]["review"]["reviewer"]
    assert reviewer["provider"] == "method-reviewer"
    assert reviewer["session_id"] == "review-method-1"
    assert reviewer["invocation"] == "review_method"
    assert result["promotion"]["state"] == "demonstrated"


def test_reviewer_metadata_without_judgment_defers_without_promotion(tmp_path: Path) -> None:
    result = _reviewer_test_run(tmp_path, {"provider": "metadata-only", "session_id": "review-metadata-1"})
    assert result["promotion"] == {"state": "deferred", "reason": "explicit reviewer judgment required"}
    assert result["candidate"]["trusted"] is False
    assert result["candidate"]["promoted"] is False


@pytest.mark.parametrize("bad_decision", [None, {"decision": "PROMOTE"}])
def test_injected_reviewer_output_still_cannot_bypass_review_validation(tmp_path: Path, bad_decision: object) -> None:
    def reviewer(candidate: dict, evidence: dict) -> object:
        return bad_decision

    with pytest.raises(LearningError):
        _reviewer_test_run(tmp_path, reviewer)


def test_acquisition_consumes_bound_qualification_tasks_and_preserves_design_lineage(tmp_path: Path) -> None:
    class TraceLearner(SyntheticLearner):
        def __init__(self) -> None:
            self.tasks: list[str] = []

        def answer(self, *, task: str, material: str = "", revision: str = "") -> dict:
            self.tasks.append(task)
            return super().answer(task=task, material=material, revision=revision)

    grader = ExecutableGrader(
        evaluator_id="grader.synthetic-bound-design.v1",
        answer_key={case: "vex(14,3)=(28,3)" if case != "negative" else "not-applicable" for case in ("pre", "practice", "post", "retest", "negative", "fresh", "control")},
        provider="independent-grader",
        session_id="grader-bound-design-1",
    )
    learner = TraceLearner()
    fresh = TraceLearner()
    engine = LearningEngine(LearningStore(tmp_path / "learning"), trusted_evaluators={grader.evaluator_id: grader})
    design = {
        "schema": "rex-learning-independent-test-design-v1",
        "designer_provenance": {"provider": "designer", "session_id": "designer-1", "role": "test_designer"},
        "evaluator_id": grader.evaluator_id,
        "evaluator_type": grader.evaluator_type,
        "phase_cases": {"baseline": "pre", "practice": "practice", "posttest": "post", "retest": "retest", "negative": "negative", "fresh": "fresh", "control": "control"},
        "phase_tasks": {"baseline": "BOUND BASELINE TASK", "practice": "BOUND PRACTICE TASK", "posttest": "BOUND RUNTIME TASK", "retest": "BOUND RETEST TASK", "negative": "BOUND NEGATIVE TASK for vex(a,b)", "fresh": "BOUND FRESH TASK", "control": "BOUND CONTROL TASK"},
        "source_free": True, "novel_transfer": True, "evaluator_independence": "independent", "contamination_status": "clean", "negative_applicability": True,
    }
    result = learn_from_source(
        engine=engine,
        learner=learner,
        source_title="Vex coordinate protocol",
        source_text="The invented operator vex(a,b) returns (2*a,b).",
        intent_key="vex-bound-design",
        target_capability="Transform vex coordinate expressions",
        grader=grader,
        reviewer=_explicit_reviewer,
        fresh_learner=fresh,
        qualification_tasks={
            "baseline": "BOUND BASELINE TASK",
            "practice": "BOUND PRACTICE TASK",
            "runtime": "BOUND RUNTIME TASK",
            "negative": "BOUND NEGATIVE TASK",
        },
        independent_design=design,
    )
    assert any("BOUND BASELINE TASK" in task for task in learner.tasks)
    assert any("BOUND PRACTICE TASK" in task for task in learner.tasks)
    assert any("BOUND RUNTIME TASK" in task for task in fresh.tasks)
    assert result["promotion"]["state"] == "demonstrated"
    tests = engine.store.list("tests")
    baseline_test = next(test for test in tests if test["phase"] == "pretest")
    assert baseline_test["independent_design_lineage"]
    assert baseline_test["independent_design_provenance"]["designer"]["session_id"] == "designer-1"
    assert baseline_test["phase_tasks"] == {"baseline": "BOUND BASELINE TASK"}


def test_acquisition_executes_canonical_phase_bound_independent_design(tmp_path: Path) -> None:
    class CanonicalLearner(SyntheticLearner):
        def __init__(self) -> None:
            self.tasks: list[str] = []
            self.materials: list[str] = []

        def answer(self, *, task: str, material: str = "", revision: str = "") -> dict:
            self.tasks.append(task)
            self.materials.append(material)
            if "candidate" in task:
                return super().answer(task=task, material=material, revision=revision)
            case = next(part.split("=", 1)[1] for part in task.split("|") if part.startswith("case="))
            answer = "unknown" if case == "baseline-case" else ("not-applicable" if case == "negative-case" else "vex(14,3)=(28,3)")
            return {"responses": [{"case": case, "answer": answer}]}

    grader = ExecutableGrader(
        evaluator_id="grader.canonical-independent.v1",
        answer_key={
            "baseline-case": "vex(14,3)=(28,3)",
            "practice-case": "vex(14,3)=(28,3)",
            "posttest-case": "vex(14,3)=(28,3)",
            "retest-case": "vex(14,3)=(28,3)",
            "negative-case": "not-applicable",
            "fresh-case": "vex(14,3)=(28,3)",
            "control-case": "vex(14,3)=(28,3)",
        },
        provider="trusted-independent-evaluator",
        session_id="canonical-evaluator-1",
    )
    learner = CanonicalLearner()
    fresh = CanonicalLearner()
    design = {
        "schema": "rex-learning-independent-test-design-v1",
        "designer_provenance": {"provider": "designer", "session_id": "designer-canonical-1", "role": "test_designer"},
        "evaluator_id": grader.evaluator_id,
        "evaluator_type": grader.evaluator_type,
        "phase_cases": {"baseline": "baseline-case", "practice": "practice-case", "posttest": "posttest-case", "retest": "retest-case", "negative": "negative-case", "fresh": "fresh-case", "control": "control-case"},
        "phase_tasks": {"baseline": "DESIGNED BASELINE", "practice": "DESIGNED PRACTICE", "posttest": "DESIGNED POSTTEST", "retest": "DESIGNED RETEST", "negative": "DESIGNED NEGATIVE", "fresh": "DESIGNED FRESH", "control": "DESIGNED CONTROL"},
        "source_free": True,
        "novel_transfer": True,
        "evaluator_independence": "independent",
        "contamination_status": "clean",
        "negative_applicability": True,
    }
    engine = LearningEngine(LearningStore(tmp_path / "learning"), trusted_evaluators={grader.evaluator_id: grader})
    result = learn_from_source(
        engine=engine, learner=learner, source_title="Vex protocol",
        source_text="The invented operator vex(a,b) returns (2*a,b).", intent_key="canonical-independent",
        target_capability="Transform vex coordinate expressions", grader=grader,
        reviewer=_explicit_reviewer, fresh_learner=fresh,
        independent_design=design,
    )
    independent_tests = [item for item in engine.store.list("tests") if item.get("independent_design_lineage")]
    assert {item["phase"]: item["cases"] for item in independent_tests} == {
        "pretest": ["baseline-case"], "practice": ["practice-case"], "posttest": ["posttest-case"],
        "retest": ["retest-case"], "negative": ["negative-case"], "edge": ["fresh-case"], "adversarial": ["control-case"],
    }
    assert all("answer_key" not in item for item in independent_tests)
    by_phase = {item["phase"]: item for item in independent_tests}
    assert by_phase["practice"]["source_free"] is False
    assert all(by_phase[phase]["source_free"] is True for phase in ("pretest", "posttest", "retest", "negative", "edge", "adversarial"))
    assert any("vex(a,b)" in material for material in learner.materials)
    assert all(not material for material in fresh.materials)
    assert all(any("DESIGNED " + phase.upper() in task for task in learner.tasks + fresh.tasks) for phase in ("BASELINE", "PRACTICE", "POSTTEST", "RETEST", "NEGATIVE"))
    assert sum("DESIGNED BASELINE" in task for task in learner.tasks) == 1
    assert sum("DESIGNED PRACTICE" in task for task in learner.tasks) == 1
    assert result["promotion"]["state"] == "demonstrated"


def test_invalid_independent_design_is_rejected_before_learner_exposure(tmp_path: Path) -> None:
    class CountingLearner(SyntheticLearner):
        def __init__(self) -> None:
            self.calls = 0

        def answer(self, *, task: str, material: str = "", revision: str = "") -> dict:
            self.calls += 1
            return super().answer(task=task, material=material, revision=revision)

    learner = CountingLearner()
    grader = ExecutableGrader(
        evaluator_id="grader.pre-exposure-gate.v1",
        answer_key={case: "vex(14,3)=(28,3)" for case in ("baseline", "practice", "posttest", "retest", "negative", "fresh", "control")},
        provider="independent-grader",
        session_id="pre-exposure-gate-grader-1",
    )
    design = {
        "schema": "rex-learning-independent-test-design-v1",
        "designer_provenance": {"provider": "designer", "session_id": "designer-gate-1", "role": "invalid"},
        "evaluator_id": grader.evaluator_id,
        "evaluator_type": grader.evaluator_type,
        "phase_cases": {phase: f"{phase}-case" for phase in ("baseline", "practice", "posttest", "retest", "negative", "fresh", "control")},
        "phase_tasks": {phase: f"DESIGNED {phase.upper()}" for phase in ("baseline", "practice", "posttest", "retest", "negative", "fresh", "control")},
        "source_free": True,
        "novel_transfer": True,
        "evaluator_independence": "independent",
        "contamination_status": "clean",
        "negative_applicability": True,
    }
    engine = LearningEngine(LearningStore(tmp_path / "learning"), trusted_evaluators=cast(dict[str, TrustedEvaluator], {grader.evaluator_id: grader}))
    with pytest.raises(LearningError, match="designer role is invalid"):
        learn_from_source(
            engine=engine,
            learner=learner,
            source_title="Vex coordinate protocol",
            source_text="The invented operator vex(a,b) returns (2*a,b).",
            intent_key="pre-exposure-gate",
            target_capability="Transform vex coordinate expressions",
            grader=grader,
            reviewer=_explicit_reviewer,
            fresh_learner=CountingLearner(),
            independent_design=design,
        )
    assert learner.calls == 0
