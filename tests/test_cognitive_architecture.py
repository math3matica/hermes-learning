from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from rex_learning.cognitive import (
    CognitiveContextPolicy,
    CognitiveProvider,
    CognitiveRouter,
    ProviderPolicy,
)
from rex_learning.engine import LearningEngine, LearningError
from rex_learning.deep_study import DeepStudyEngine
from rex_learning.store import LearningStore


@dataclass
class FixtureProvider(CognitiveProvider):
    provider_id: str
    model_id: str

    def invoke(self, request):
        return {"operation": request.operation_type, "provider": self.provider_id, "answer": "portable result"}


@dataclass
class FailingProvider(CognitiveProvider):
    provider_id: str
    model_id: str

    def invoke(self, request):
        raise RuntimeError("provider unavailable")


def test_same_operation_contract_routes_to_different_providers_and_persists_provenance(tmp_path: Path) -> None:
    store = LearningStore(tmp_path / "learning")
    router = CognitiveRouter(
        store,
        providers={"local": FixtureProvider("local", "model-a"), "cloud": FixtureProvider("cloud", "model-b")},
        policy=ProviderPolicy({"default": {"provider": "local"}, "semantic_evaluation": {"provider": "cloud"}}),
    )

    local = router.run("close_read", {"text": "material"})
    cloud = router.run("semantic_evaluation", {"answer": "attempt", "reference": "hidden"}, context=CognitiveContextPolicy.for_operation("semantic_evaluation"))

    assert local.output["operation"] == "close_read"
    assert cloud.output["operation"] == "semantic_evaluation"
    assert local.provenance["provider"] == "local"
    assert cloud.provenance["provider"] == "cloud"
    assert local.provenance["operation_type"] == "close_read"
    assert cloud.provenance["context_policy"]["hidden_reference_visible"] is True
    assert len(store.list("cognitive_operations")) == 2


def test_context_policy_rejects_source_leakage_into_source_free_retrieval(tmp_path: Path) -> None:
    router = CognitiveRouter(
        LearningStore(tmp_path / "learning"),
        providers={"local": FixtureProvider("local", "model-a")},
        policy=ProviderPolicy({"default": {"provider": "local"}}),
    )

    with pytest.raises(LearningError, match="source-free"):
        router.run(
            "retrieve",
            {"source": "forbidden", "task": "retrieve"},
            context=CognitiveContextPolicy(source_visible=True),
        )


def test_same_model_isolated_evaluation_has_explicit_evidence_strength(tmp_path: Path) -> None:
    router = CognitiveRouter(
        LearningStore(tmp_path / "learning"),
        providers={"local": FixtureProvider("local", "model-a")},
        policy=ProviderPolicy({"default": {"provider": "local"}}),
    )

    result = router.run(
        "semantic_evaluation",
        {"learner_answer": "attempt", "hidden_reference": "reference"},
        context=CognitiveContextPolicy.for_operation("semantic_evaluation"),
        evidence_class="same_model_isolated_context",
    )

    assert result.provenance["evidence_class"] == "same_model_isolated_context"
    assert result.provenance["context_policy"]["prior_answer_visible"] is False


def test_failed_provider_can_escalate_to_policy_fallback_with_provenance(tmp_path: Path) -> None:
    store = LearningStore(tmp_path / "learning")
    router = CognitiveRouter(
        store,
        providers={"primary": FailingProvider("primary", "model-a"), "fallback": FixtureProvider("fallback", "model-b")},
        policy=ProviderPolicy({"default": {"provider": "primary"}, "fallback": {"provider": "fallback"}}),
    )

    result = router.run("diagnose", {"attempt": "uncertain"})

    assert result.provenance["provider"] == "fallback"
    records = store.list("cognitive_operations")
    assert {record["status"] for record in records} == {"failed", "completed"}
    assert result.provenance["escalated_from"] == "primary"


def test_authorized_cloud_provider_can_create_candidate_without_being_hard_coded_as_evaluator(tmp_path: Path) -> None:
    engine = LearningEngine(LearningStore(tmp_path / "learning"))
    curriculum = engine.create_curriculum("Provider neutral", "Persist a procedure")
    skill = engine.create_skill_hypothesis(curriculum["id"], {"key": "portable", "claim": "claim"})

    revision = engine.consolidate_learner_revision(
        skill["id"],
        {"claim": "cloud-derived claim", "operational_procedure": "do the procedure", "applicability": ["portable task"]},
        learner_provenance={"provider": "openai-cloud", "model": "luna", "session_id": "study-1", "role": "study_and_consolidate"},
        source_provenance={"source_refs": ["source://book"]},
        failure_ref="failure-1",
        revision_type="corrective_revision",
    )

    assert revision["author"] == "provider_derived"
    assert revision["qualification_status"] == "candidate"
    assert revision["learner_provenance"]["provider"] == "openai-cloud"


def test_content_validation_allows_experimental_reuse_without_demonstrating_skill(tmp_path: Path) -> None:
    engine = LearningEngine(LearningStore(tmp_path / "learning"))
    curriculum = engine.create_curriculum("States", "Persist a procedure")
    skill = engine.create_skill_hypothesis(curriculum["id"], {"key": "states", "claim": "claim", "applicability": ["target task"]})
    engine.consolidate_learner_revision(
        skill["id"], {"claim": "validated claim", "operational_procedure": "bounded procedure", "applicability": ["target task"]},
        learner_provenance={"provider": "local", "session_id": "study", "role": "study_and_consolidate"},
        source_provenance={"source_refs": ["source://x"]}, failure_ref="f", revision_type="study",
    )

    validated = engine.validate_skill_content(skill["id"], {
        "provider": "local", "model": "model-a", "session_id": "validation", "independence": "same_model_isolated_context",
        "contamination": {"status": "clean"}, "judgment": "supported", "source_fidelity": "pass",
    })

    assert validated["qualification_status"] == "content_validated"
    assert validated["state"] != "demonstrated"
    selected = engine.discover_applicable_skills({"requirements": ["target task"]})
    assert selected[0]["reuse_mode"] == "experimental"


def test_skill_survives_restart_and_can_be_consumed_by_another_provider(tmp_path: Path) -> None:
    engine = LearningEngine(LearningStore(tmp_path / "learning"))
    curriculum = engine.create_curriculum("Portability", "Persist a procedure")
    skill = engine.create_skill_hypothesis(curriculum["id"], {"key": "portable", "claim": "claim", "applicability": ["portable task"]})
    engine.consolidate_learner_revision(
        skill["id"], {"claim": "portable claim", "operational_procedure": "portable procedure", "applicability": ["portable task"]},
        learner_provenance={"provider": "provider-a", "model": "a", "session_id": "a-1", "role": "study_and_consolidate"},
        source_provenance={"source_refs": ["source://x"]}, failure_ref="f", revision_type="study",
    )
    engine.validate_skill_content(skill["id"], {"provider": "provider-a", "session_id": "v", "independence": "same_model_isolated_context", "contamination": {"status": "clean"}, "judgment": "supported"})
    fresh = LearningEngine(LearningStore(tmp_path / "learning"))
    selected = fresh.discover_applicable_skills({"requirements": ["portable task"]})
    assert selected[0]["operational_context"]["procedure"] == "portable procedure"
    assert selected[0]["consumer_provider"] is None


def test_deep_study_exposure_can_carry_normalized_operation_provenance(tmp_path: Path) -> None:
    store = LearningStore(tmp_path / "learning")
    study = DeepStudyEngine(store)
    run = study.create_run(title="Provider-neutral study", source_id="source-1", source_hash="hash", method_version="m1", provider_policy={"default": {"provider": "local"}})
    unit = study.create_unit(run["id"], key="unit-1", kind="chapter", title="Unit", parent_key=None, source_refs=["source-1"], source_hash="hash")

    exposure = study.record_exposure(
        run["id"], unit["id"], phase="close_reading", provider="local", session_id="session-1",
        source_refs=["source-1"], input_hash="input", material_kind="source", output_ref="op-1",
        operation_provenance={"operation_id": "op-1", "operation_type": "close_read", "model": "model-a", "context_policy": {"source_visible": True}},
    )

    assert exposure["operation_provenance"]["operation_type"] == "close_read"
