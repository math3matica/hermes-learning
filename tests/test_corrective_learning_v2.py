from __future__ import annotations

import json
from pathlib import Path

import pytest

from rex_learning.corrective_learning import canonicalize_learner_answer
from rex_learning.engine import LearningEngine, LearningError
from rex_learning.store import LearningStore


def test_canonicalizer_includes_answer_fields_after_first_response() -> None:
    result = {
        "responses": [
            {"case": "problem", "answer": "state the problem"},
            {"case": "term_1", "answer": "remote triage"},
            {"case": "qualification", "answer": "staffing limits generalization"},
        ],
        "_provider_response": {"usage": {"total_tokens": 999}},
    }

    canonical = canonicalize_learner_answer(result)

    assert "term_1:\nremote triage" in canonical
    assert "qualification:\nstaffing limits generalization" in canonical
    assert "total_tokens" not in canonical


def test_canonicalizer_handles_natural_answer_nested_values_and_is_stable() -> None:
    result = {"answer": {"steps": ["retrieve", "check feedback"], "boundary": "model first"}}

    first = canonicalize_learner_answer(result)
    second = canonicalize_learner_answer(json.loads(json.dumps(result)))

    assert first == second
    assert "answer:" in first
    assert "retrieve" in first and "model first" in first


def test_learner_revision_is_versioned_and_selectively_reused_across_engine_instances(tmp_path: Path) -> None:
    engine = LearningEngine(LearningStore(tmp_path / "learning"))
    curriculum = engine.create_curriculum("Corrective", "Persist a procedure")
    skill = engine.create_skill_hypothesis(
        curriculum["id"],
        {"key": "study.retrieval", "claim": "Use retrieval with feedback", "applicability": ["retrieval planning"]},
    )

    revision = engine.consolidate_learner_revision(
        skill["id"],
        {
            "claim": "Use source-free retrieval followed by feedback and spacing.",
            "operational_procedure": "Retrieve first; inspect feedback; target failed propositions; space later attempts.",
            "applicability": ["retrieval planning"],
            "non_applicability": ["no coherent initial model"],
            "boundaries": ["give a brief model before difficult retrieval when no coherent model exists"],
            "uncertainty": "Qualified for study-planning cases; not a universal ordering rule.",
        },
        learner_provenance={"provider": "qwen-local", "session_id": "qwen-revision-1", "response_digest": "abc"},
        source_provenance={"source_refs": ["source://targeted"], "source_hash": "source-hash"},
        failure_ref="historical-failure-1",
        revision_type="corrective_revision",
    )
    assert revision["version"] == 1
    assert revision["author"] == "learner"
    assert revision["qualification_status"] == "candidate"
    assert engine.store.read("skill_versions", f"{skill['id']}.v0")["superseded_by"] == 1
    engine.qualify_learner_revision(skill["id"], {
        "provider": "openai-codex", "session_id": "luna-validate-1", "independence": "independent",
        "contamination": {"status": "clean"}, "judgment": "supported",
    })

    fresh = LearningEngine(LearningStore(tmp_path / "learning"))
    selected = fresh.discover_applicable_skills({"requirements": ["retrieval planning"]})
    assert selected[0]["skill_version"] == 1
    assert "Retrieve first" in selected[0]["operational_context"]["procedure"]
    assert fresh.discover_applicable_skills({"requirements": ["unrelated deployment"]}) == []


def test_luna_diagnosis_cannot_be_consolidated_as_learner_revision(tmp_path: Path) -> None:
    engine = LearningEngine(LearningStore(tmp_path / "learning"))
    curriculum = engine.create_curriculum("Corrective", "Persist a procedure")
    skill = engine.create_skill_hypothesis(curriculum["id"], {"key": "k", "claim": "claim"})

    with pytest.raises(LearningError, match="learner-authored"):
        engine.consolidate_learner_revision(
            skill["id"],
            {"claim": "Luna says add feedback", "operational_procedure": "add feedback"},
            learner_provenance={"provider": "openai-codex", "session_id": "luna-1", "role": "evaluator"},
            source_provenance={"source_refs": ["source://x"]},
            failure_ref="f",
            revision_type="diagnosis",
        )