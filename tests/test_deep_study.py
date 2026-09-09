from __future__ import annotations

from pathlib import Path

import pytest

from rex_learning import DeepStudyEngine, LearningError, LearningStore


def _run(tmp_path: Path) -> tuple[DeepStudyEngine, dict, dict]:
    engine = DeepStudyEngine(LearningStore(tmp_path / "learning"))
    run = engine.create_run(
        title="Bootstrap Deep Study",
        source_id="source-book-v2",
        source_hash="sha256-book",
        method_version="rex.deep-study.v1",
        provider_policy={"primary": "qwen-local", "auxiliary": ["luna"]},
    )
    unit = engine.create_unit(
        run["id"],
        key="book/chapter-1/section-1",
        kind="section",
        title="Section 1",
        parent_key="book/chapter-1",
        source_refs=["book.xhtml#Section 1:0"],
        source_hash="section-sha",
    )
    return engine, run, unit


def test_ingestion_does_not_imply_read_or_completion(tmp_path: Path) -> None:
    engine, run, unit = _run(tmp_path)

    engine.mark_ingested(unit["id"], evidence={"source_refs": unit["source_refs"], "deterministic": True})
    current = engine.unit(unit["id"])

    assert current["state"] == "ingested"
    assert current["state"] != "close_read"
    assert current["state"] != "complete"
    assert engine.progress(run["id"])["counts"]["ingested"] == 1


def test_close_reading_requires_provider_exposure_and_retrieval_can_advance(tmp_path: Path) -> None:
    engine, run, unit = _run(tmp_path)
    engine.mark_ingested(unit["id"], evidence={"source_refs": unit["source_refs"]})

    with pytest.raises(LearningError, match="exposure"):
        engine.transition(unit["id"], "close_reading")

    exposure = engine.record_exposure(
        run["id"], unit["id"], phase="close_reading", provider="qwen-local",
        session_id="qwen-session-1", source_refs=unit["source_refs"],
        input_hash="input-close-1", material_kind="raw_source", output_ref="provider/close-1",
    )
    assert exposure["provider"] == "qwen-local"
    engine.transition(unit["id"], "retrieval_due", reason="close reading completed", evidence_ids=[exposure["id"]])

    retrieval = engine.record_retrieval_attempt(
        run["id"], unit["id"], provider="qwen-local", session_id="qwen-session-1",
        source_free=True, input_hash="input-retrieval-1", score=0.9,
        passed=True, failed_question_ids=[], evidence_ref="provider/retrieval-1",
    )
    assert retrieval["passed"] is True
    assert engine.unit(unit["id"])["state"] == "provisionally_understood"


def test_failed_retrieval_schedules_targeted_reread_and_resume_is_idempotent(tmp_path: Path) -> None:
    engine, run, unit = _run(tmp_path)
    engine.mark_ingested(unit["id"], evidence={"source_refs": unit["source_refs"]})
    exposure = engine.record_exposure(
        run["id"], unit["id"], phase="close_reading", provider="qwen-local",
        session_id="qwen-session-1", source_refs=unit["source_refs"],
        input_hash="input-close-1", material_kind="raw_source", output_ref="provider/close-1",
    )
    engine.transition(unit["id"], "retrieval_due", evidence_ids=[exposure["id"]])

    first = engine.record_retrieval_attempt(
        run["id"], unit["id"], provider="qwen-local", session_id="qwen-session-1",
        source_free=True, input_hash="input-retrieval-fail", score=0.25,
        passed=False, failed_question_ids=["q-1", "q-3"], evidence_ref="provider/retrieval-fail",
    )
    second = engine.record_retrieval_attempt(
        run["id"], unit["id"], provider="qwen-local", session_id="qwen-session-1",
        source_free=True, input_hash="input-retrieval-fail", score=0.25,
        passed=False, failed_question_ids=["q-1", "q-3"], evidence_ref="provider/retrieval-fail",
    )

    assert first["id"] == second["id"]
    assert engine.unit(unit["id"])["state"] == "needs_reread"
    rereads = engine.store.list("study_rereads")
    assert len(rereads) == 1
    assert rereads[0]["reason"] == "retrieval_failure"


def test_later_refinement_reopens_earlier_understanding(tmp_path: Path) -> None:
    engine, run, first = _run(tmp_path)
    later = engine.create_unit(
        run["id"], key="book/chapter-2/section-1", kind="section", title="Section 2",
        parent_key="book/chapter-2", source_refs=["book.xhtml#Section 2:0"], source_hash="later-sha",
    )
    engine.mark_ingested(first["id"], evidence={"source_refs": first["source_refs"]})
    engine.mark_ingested(later["id"], evidence={"source_refs": later["source_refs"]})
    for unit in (first, later):
        exposure = engine.record_exposure(
            run["id"], unit["id"], phase="close_reading", provider="qwen-local",
            session_id="qwen-session-1", source_refs=unit["source_refs"], input_hash=f"close-{unit['id']}",
            material_kind="raw_source", output_ref=f"provider/{unit['id']}",
        )
        engine.transition(unit["id"], "retrieval_due", evidence_ids=[exposure["id"]])
        engine.record_retrieval_attempt(
            run["id"], unit["id"], provider="qwen-local", session_id="qwen-session-1",
            source_free=True, input_hash=f"retrieve-{unit['id']}", score=1.0, passed=True,
            failed_question_ids=[], evidence_ref=f"provider/retrieve-{unit['id']}",
        )

    engine.record_relation(
        run["id"], source_unit_id=later["id"], target_unit_id=first["id"],
        relation="refines", evidence_ref="provider/relation-1", provider="qwen-local", session_id="qwen-session-1",
    )
    assert engine.unit(first["id"])["state"] == "needs_reread"
    assert engine.progress(run["id"])["counts"]["needs_reread"] == 1


def test_conflicting_material_is_recorded_and_practice_failure_traces_to_source(tmp_path: Path) -> None:
    engine, run, unit = _run(tmp_path)
    engine.mark_ingested(unit["id"], evidence={"source_refs": unit["source_refs"]})
    engine.record_conflict(run["id"], unit["id"], claim_ref="knowledge/new", prior_ref="knowledge/old", evidence_ref="provider/conflict", provider="luna", session_id="luna-1")
    task = engine.create_practice_task(run["id"], unit["id"], claim_ref="skill/candidate", task_ref="practice/task-1", source_refs=unit["source_refs"])
    result = engine.record_practice_result(run["id"], task["id"], passed=False, score=0.2, provider="qwen-local", session_id="qwen-session-1", failure_reason="missed condition")
    assert result["passed"] is False
    assert engine.unit(unit["id"])["state"] == "needs_reread"
    assert engine.store.list("study_conflicts")[0]["provider"] == "luna"
    assert engine.store.list("study_rereads")[0]["reason"] == "practice_failure"


def test_progress_view_reports_provider_provenance_and_does_not_promote_luna(tmp_path: Path) -> None:
    engine, run, unit = _run(tmp_path)
    engine.mark_ingested(unit["id"], evidence={"source_refs": unit["source_refs"]})
    exposure = engine.record_exposure(
        run["id"], unit["id"], phase="inspection", provider="luna", session_id="luna-1",
        source_refs=unit["source_refs"], input_hash="inspect-1", material_kind="raw_source", output_ref="provider/inspect-1",
    )
    engine.transition(unit["id"], "inspected", evidence_ids=[exposure["id"]])
    view = engine.human_progress(run["id"])
    assert "provider=luna" in view
    assert "inspected=1" in view
    assert engine.unit(unit["id"])["capability_evidence"] == []


def test_method_and_exposure_replay_are_idempotent_across_process_restart(tmp_path: Path) -> None:
    engine, run, unit = _run(tmp_path)
    method = engine.create_method_version(
        method_id="rex.deep-study", version="v-test", parent_version=None,
        policy={"retrieval": True}, provenance={"provider": "deterministic"},
    )
    replayed_method = engine.create_method_version(
        method_id="rex.deep-study", version="v-test", parent_version=None,
        policy={"retrieval": True}, provenance={"provider": "deterministic"},
    )
    assert method["id"] == replayed_method["id"]
    exposure = engine.record_exposure(
        run["id"], unit["id"], phase="inspection", provider="luna", session_id="luna-1",
        source_refs=unit["source_refs"], input_hash="inspect-replay", material_kind="raw_source", output_ref="provider/inspect-replay",
    )
    replayed_exposure = engine.record_exposure(
        run["id"], unit["id"], phase="inspection", provider="luna", session_id="luna-1",
        source_refs=unit["source_refs"], input_hash="inspect-replay", material_kind="raw_source", output_ref="provider/inspect-replay",
    )
    assert exposure["id"] == replayed_exposure["id"]


def test_checkpoint_is_persisted_and_replayed_from_store(tmp_path: Path) -> None:
    engine, run, _ = _run(tmp_path)
    checkpoint = engine.checkpoint(run["id"], phase="source_free_retrieval", unit_id="unit-1", artifacts={"stage": {"value": 1}})
    reopened = DeepStudyEngine(LearningStore(tmp_path / "learning"))
    assert checkpoint["checkpoint"] == reopened.run(run["id"])["checkpoint"]
    assert checkpoint["checkpoint"]["last_phase"] == "source_free_retrieval"
    assert reopened.run(run["id"])["artifacts"] == {"stage": {"value": 1}}


def test_unseen_test_is_persisted_without_provider_capability_promotion(tmp_path: Path) -> None:
    engine, run, unit = _run(tmp_path)
    engine.mark_ingested(unit["id"], evidence={"source_refs": unit["source_refs"]})
    task = engine.create_practice_task(run["id"], unit["id"], claim_ref="claim-1", task_ref="practice-1", source_refs=unit["source_refs"])
    engine.record_practice_result(run["id"], task["id"], passed=True, score=1.0, provider="qwen-local", session_id="qwen-1")
    test = engine.record_test_result(run["id"], unit["id"], test_kind="retest", unseen=True, passed=True, score=1.0, provider="qwen-local", session_id="qwen-1", evidence_ref="provider/retest-1")
    assert test["capability_evidence"] is False
    assert engine.unit(unit["id"])["state"] == "tested"
    assert engine.store.list("study_tests")[0]["evaluator_class"] == "provider_structural"


def test_semantic_evaluation_persists_provenance_feedback_and_does_not_mutate_lifecycle(tmp_path: Path) -> None:
    engine, run, unit = _run(tmp_path)
    judgment = {
        "label": "correct_but_incomplete",
        "dimensions": {"semantic_correctness": "pass", "procedure_execution": "pass", "applicability_judgment": "uncertain", "uncertainty_calibration": "pass"},
        "diagnoses": ["essential_omission"], "missing_propositions": ["the boundary condition"],
        "unsupported_claims": [], "feedback": "Retrieve the boundary condition and retest an unseen case.", "confidence": 0.9,
    }
    kwargs = {
        "test_ref": "held-out-1", "attempt_id": "attempt-1",
        "learner": {"provider": "qwen-local", "session_id": "qwen-1", "source_free": True},
        "evaluator": {"provider": "openai-codex", "model": "gpt-5.6-luna", "session_id": "luna-1"},
        "judgment": judgment, "contamination": {"status": "clean", "method": "attested"},
        "protocol_ref": "bootstrap-evaluation-protocol-v2.json",
        "remediation": {"action": "targeted_retrieval", "status": "scheduled"},
    }
    record = engine.record_semantic_evaluation(run["id"], unit["id"], **kwargs)
    replay = engine.record_semantic_evaluation(run["id"], unit["id"], **kwargs)
    assert record["id"] == replay["id"]
    assert record["feedback"]["action"] == "targeted_retrieval"
    assert record["capability_evidence_boundary"] == "learner_behavior_only"
    assert engine.unit(unit["id"])["state"] == "discovered"
    assert len(engine.store.list("study_semantic_evaluations")) == 1


def test_semantic_evaluation_rejects_immutable_replay_drift(tmp_path: Path) -> None:
    engine, run, unit = _run(tmp_path)
    kwargs = {
        "test_ref": "held-out-1", "attempt_id": "attempt-1",
        "learner": {"provider": "qwen-local", "session_id": "qwen-1"},
        "evaluator": {"provider": "openai-codex", "session_id": "luna-1"},
        "judgment": {"label": "correct", "dimensions": {"semantic_correctness": "pass"}, "diagnoses": [], "missing_propositions": [], "unsupported_claims": [], "feedback": "Pass.", "confidence": 0.9},
        "contamination": {"status": "clean"}, "protocol_ref": "protocol-v2",
    }
    engine.record_semantic_evaluation(run["id"], unit["id"], **kwargs)
    with pytest.raises(LearningError, match="immutable"):
        engine.record_semantic_evaluation(run["id"], unit["id"], **{**kwargs, "contamination": {"status": "unknown"}})
