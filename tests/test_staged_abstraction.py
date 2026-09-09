from pathlib import Path

import pytest

from rex_learning import CurriculumIntent, LearningError, LearningStore, SourceContext
from rex_learning.staged_abstraction import (
    EpistemicLedger,
    StagedAbstractionPipeline,
    validate_staged_abstraction,
)


def make_intent(**overrides):
    data = {
        "key": "target",
        "title": "Target capability",
        "target_capability": "Build and debug reliable programs.",
        "competence_level": "working",
        "intended_use": ["build", "debug"],
        "intended_transfer": ["unseen programs"],
        "environment": {"runtime": "17"},
        "temporal_requirements": "current procedures require verification",
        "evaluation_criteria": ["explain", "test"],
    }
    data.update(overrides)
    return CurriculumIntent.from_dict(data)


def make_source(**overrides):
    data = {
        "source_id": "book",
        "unit_id": "unit-1",
        "title": "Exact rule",
        "text": "The protocol field must be written before the payload.",
        "source_refs": ["book#1"],
        "source_quality": "adequate",
    }
    data.update(overrides)
    return SourceContext.from_dict(data)


def test_ledger_distinguishes_source_intent_inference_unknown_and_external():
    ledger = EpistemicLedger()
    source = ledger.add("source_fact", "s1", basis="source", support_refs=["book#1"])
    intent = ledger.add("intent_fact", "i1", basis="intent", support_refs=["target"])
    inference = ledger.add("inference", "i2", basis="s1+i1", support_refs=[source, intent])
    unknown = ledger.add("unknown", "u1", basis="not established")
    external = ledger.add("external_verification_required", "e1", basis="current status absent")

    assert [item.status for item in ledger.items()] == [
        "SOURCE_ESTABLISHED", "INTENT_ESTABLISHED", "INFERRED", "UNKNOWN", "EXTERNAL_VERIFICATION_REQUIRED"
    ]
    assert ledger.by_id()[inference].support_refs == [source, intent]
    assert ledger.by_id()[unknown].support_refs == []
    assert ledger.by_id()[external].support_refs == []


def test_validation_rejects_claim_without_valid_ledger_support():
    candidate = {
        "claims": [{"id": "c1", "claim": "invented", "support_refs": ["missing"]}],
        "relevance": {"support_refs": ["missing"]},
        "source_invariant": {"claim": "field before payload", "support_refs": ["s1"]},
        "goal_conditioned_interpretation": {"claim": "practice", "support_refs": ["i1"]},
    }
    with pytest.raises(LearningError, match="support"):
        validate_staged_abstraction(candidate, ledger_ids={"s1", "i1"})


def test_validation_blocks_source_fact_citing_intent_only():
    candidate = {
        "claims": [{"id": "c1", "claim": "the source says X", "support_refs": ["i1"], "support_type": "source_established"}],
        "relevance": {"support_refs": ["i1"]},
        "source_invariant": {"claim": "field before payload", "support_refs": ["s1"]},
        "goal_conditioned_interpretation": {"claim": "practice", "support_refs": ["i1"]},
    }
    with pytest.raises(LearningError, match="source_established"):
        validate_staged_abstraction(candidate, ledger_ids={"s1", "i1"}, ledger_kinds={"s1": "SOURCE_ESTABLISHED", "i1": "INTENT_ESTABLISHED"})


def test_pipeline_preserves_invariant_and_changes_application_by_goal():
    pipeline = StagedAbstractionPipeline()
    source = make_source()
    general = pipeline.run(make_intent(), source, provider=lambda operation, inputs: {"operation": operation, "inputs": inputs})
    maintenance = pipeline.run(make_intent(target_capability="Maintain protocol compatibility", environment={"protocol": "wire"}), source, provider=lambda operation, inputs: {"operation": operation, "inputs": inputs})

    assert general.final["source_invariant"]["claim"] == maintenance.final["source_invariant"]["claim"]
    assert general.final["goal_conditioned_interpretation"]["claim"] != maintenance.final["goal_conditioned_interpretation"]["claim"]
    assert general.provenance["stage_count"] >= 4


def test_unknown_currentness_routes_to_external_verification():
    result = StagedAbstractionPipeline().run(
        make_intent(), make_source(title="Setup", text="Use the installation procedure."),
        provider=lambda operation, inputs: {"operation": operation},
    )
    assert result.final["currentness"]["status"] == "UNKNOWN_RELEVANT"
    assert result.final.get("learning_action") in {None, "unresolved"}


def test_final_claims_are_not_persisted_when_unsupported(tmp_path: Path):
    store = LearningStore(tmp_path / "learning")
    result = StagedAbstractionPipeline().run(make_intent(), make_source(), provider=lambda operation, inputs: {"operation": operation})
    persisted = result.persist(store, intent_id="intent-1", source_hash="source-hash")
    assert persisted["durable_claims"] == []
    assert store.list("staged_abstraction_runs")[0]["schema"] == "rex-learning-staged-abstraction-run-v2"


def test_critique_blocks_overbroad_goal_claim_from_final_output():
    def provider(operation, inputs):
        if operation == "critique_instructional_abstraction":
            return {"overbroad_applicability": ["goal claim exceeds source"], "unsupported_claims": []}
        return {}

    result = StagedAbstractionPipeline().run(make_intent(), make_source(), provider=provider)
    assert result.final["goal_conditioned_interpretation"]["status"] == "UNKNOWN"


def test_replay_legacy_support_suffixes_are_resolved_without_promoting_rejected_claims():
    candidate = {
        "claims": [{
            "id": "old-claim",
            "status": "REJECTED",
            "support_type": "source_established",
            "support_refs": ["ledger-0005-deadbeef1234"],
        }],
        "source_invariant": {
            "status": "UNKNOWN",
            "support_refs": ["ledger-0005-deadbeef1234"],
        },
        "goal_conditioned_interpretation": {
            "status": "PROPOSED",
            "support_refs": ["ledger-0009-missing00000"],
        },
    }
    validate_staged_abstraction(
        candidate,
        ledger_ids={"ledger-0004-deadbeef1234"},
        ledger_kinds={"ledger-0004-deadbeef1234": "SOURCE_ESTABLISHED"},
    )


def test_non_temporal_source_has_no_universal_currentness_or_verification_action():
    result = StagedAbstractionPipeline().run(
        make_intent(temporal_requirements=""),
        make_source(title="Mathematical implication", text="If A implies B and A is true, B is true."),
        provider=lambda operation, inputs: {},
    )
    assert result.final["currentness"]["applicable"] == "FALSE"
    assert result.final["currentness"]["status"] == "NOT_APPLICABLE"
    assert result.final.get("learning_action") != "verify_current_authoritative_source"


def test_unknown_relevant_currentness_requires_explicit_applicability():
    result = StagedAbstractionPipeline().run(
        make_intent(temporal_requirements="current production procedure required"),
        make_source(title="Current API procedure", text="Call the service endpoint using this configuration."),
        provider=lambda operation, inputs: {
            "currentness": {"applicable": "true", "status": "UNKNOWN_RELEVANT", "support_refs": []}
            if operation == "apply_curriculum_intent" else {},
        },
    )
    assert result.final["currentness"]["status"] == "UNKNOWN_RELEVANT"
    assert result.final.get("learning_action") != "verify_current_authoritative_source"


def test_missing_provider_invariant_remains_unknown_and_policy_is_not_source_claim():
    result = StagedAbstractionPipeline().run(make_intent(), make_source(), provider=lambda operation, inputs: {})
    assert result.final["source_invariant"]["status"] == "unknown"
    assert result.final["source_invariant"]["support_refs"] == []
    assert "Retain only what the visible source supports." not in [
        claim.get("claim") for claim in result.final["claims"]
    ]
    assert result.final["epistemic_policy"]


def test_grounding_output_is_authoritative_over_candidate_claims():
    def provider(operation, inputs):
        if operation == "derive_candidate_abstraction":
            return {
                "claims": [{"id": "candidate-c1", "claim": "candidate fact", "support_refs": ["ledger-0002-"], "support_type": "source_established"}],
                "source_invariant": {"claim": "candidate invariant", "support_refs": ["ledger-0002-"]},
            }
        if operation == "ground_abstraction_claims":
            return {"claims": [{"id": "candidate-c1", "status": "REJECTED", "support_refs": [], "support_type": "unknown"}]}
        return {}

    # The provider fixture deliberately uses the real ledger id discovered from the input.
    def grounded_provider(operation, inputs):
        if operation == "derive_candidate_abstraction":
            source_ref = inputs["ledger"][1]["id"]
            return {"claims": [{"id": "candidate-c1", "claim": "candidate fact", "support_refs": [source_ref], "support_type": "source_established"}], "source_invariant": {"claim": "candidate invariant", "support_refs": [source_ref]}}
        if operation == "ground_abstraction_claims":
            return {"claims": [{"id": "candidate-c1", "status": "REJECTED", "support_refs": [], "support_type": "unknown"}]}
        return {}

    result = StagedAbstractionPipeline().run(make_intent(), make_source(), provider=grounded_provider)
    claim = next(item for item in result.final["claims"] if item["id"] == "candidate-c1")
    assert claim["status"] == "REJECTED"


def test_critique_lost_exact_detail_rejects_the_affected_claim():
    def provider(operation, inputs):
        if operation == "derive_candidate_abstraction":
            source_ref = inputs["ledger"][1]["id"]
            return {
                "exact_details": [{
                    "id": "detail-1",
                    "claim": "The field is written before the payload.",
                    "support_refs": [source_ref],
                    "support_type": "source_established",
                }],
            }
        if operation == "critique_instructional_abstraction":
            return {
                "findings": [{
                    "finding_id": "finding-detail-1",
                    "target_path": "exact_detail[0]",
                    "finding_type": "LOST_EXACT_DETAIL",
                    "severity": "error",
                    "recommended_disposition": "REJECT",
                }],
            }
        return {}

    result = StagedAbstractionPipeline().run(make_intent(), make_source(), provider=provider)
    claim = next(item for item in result.final["claims"] if item["id"] == "detail-1")
    assert claim["status"] == "REJECTED"


def test_critique_is_surgical_and_unsupported_action_cannot_survive():
    def provider(operation, inputs):
        if operation == "derive_candidate_abstraction":
            source_ref = inputs["ledger"][1]["id"]
            return {"claims": [
                {"id": "supported-c1", "claim": "supported fact", "support_refs": [source_ref], "support_type": "source_established"},
                {"id": "broad-c2", "claim": "broad fact", "support_refs": [source_ref], "support_type": "source_established"},
            ]}
        if operation == "apply_curriculum_intent":
            return {"learning_action": "verify_current_authoritative_source", "currentness": {"applicable": "false", "status": "NOT_APPLICABLE", "support_refs": []}}
        if operation == "critique_instructional_abstraction":
            return {"findings": [{"finding_id": "f1", "target_claim_id": "broad-c2", "finding_type": "OVERBROAD", "severity": "error", "rationale": "too broad", "recommended_disposition": "REJECT"}, {"finding_id": "f2", "target_path": "learning_action", "finding_type": "ACTION_UNSUPPORTED", "severity": "error", "rationale": "not applicable", "recommended_disposition": "UNRESOLVED"}]}
        return {}

    result = StagedAbstractionPipeline().run(make_intent(), make_source(), provider=provider)
    statuses = {item["id"]: item["status"] for item in result.final["claims"]}
    assert statuses["supported-c1"] == "SUPPORTED"
    assert statuses["broad-c2"] == "REJECTED"
    assert result.final["learning_action"] == "unresolved"


def test_critique_unsupported_evidence_path_is_removed_from_final_output():
    def provider(operation, inputs):
        if operation == "select_learning_evidence":
            return {"evidence_path": [{"step": "invented practice route"}]}
        if operation == "critique_instructional_abstraction":
            return {"findings": [{
                "finding_id": "evidence-path-1",
                "target_path": "evidence_path",
                "finding_type": "EVIDENCE_PATH_UNSUPPORTED",
                "severity": "error",
                "recommended_disposition": "REJECT",
            }]}
        return {}

    result = StagedAbstractionPipeline().run(make_intent(), make_source(), provider=provider)
    assert result.final["evidence_path"] == []


def test_unknown_currentness_is_not_external_verification_or_an_automatic_action():
    def provider(operation, inputs):
        if operation == "apply_curriculum_intent":
            return {"currentness": {"applicable": "true", "status": "UNKNOWN", "support_refs": []}}
        return {}

    result = StagedAbstractionPipeline().run(
        make_intent(temporal_requirements="current production procedure required"),
        make_source(title="Service procedure", text="Call the service endpoint."),
        provider=provider,
    )
    assert result.final["currentness"]["status"] == "UNKNOWN"
    assert result.final.get("learning_action") != "verify_current_authoritative_source"


def test_external_verification_status_is_distinct_and_routes_to_verification():
    def provider(operation, inputs):
        if operation == "apply_curriculum_intent":
            return {"currentness": {"applicable": "true", "status": "EXTERNAL_VERIFICATION_REQUIRED", "support_refs": []}}
        return {}

    result = StagedAbstractionPipeline().run(
        make_intent(temporal_requirements="current production procedure required"),
        make_source(title="Service procedure", text="Call the service endpoint."),
        provider=provider,
    )
    assert result.final["currentness"]["status"] == "EXTERNAL_VERIFICATION_REQUIRED"
    assert result.final["learning_action"] == "verify_current_authoritative_source"


def test_lost_exact_detail_narrowing_preserves_claim_as_bounded_inference():
    def provider(operation, inputs):
        if operation == "derive_candidate_abstraction":
            source_ref = inputs["ledger"][1]["id"]
            return {"exact_details": [{"id": "detail-1", "claim": "The field precedes the payload.", "support_refs": [source_ref], "support_type": "source_established"}]}
        if operation == "critique_instructional_abstraction":
            return {"findings": [{"finding_id": "f1", "target_claim_id": "detail-1", "finding_type": "LOST_EXACT_DETAIL", "recommended_disposition": "NARROW"}]}
        return {}

    result = StagedAbstractionPipeline().run(make_intent(), make_source(), provider=provider)
    claim = next(item for item in result.final["claims"] if item["id"] == "detail-1")
    assert claim["status"] == "BOUNDED_INFERENCE"


def test_action_unsupported_accepts_action_alias_and_is_surgical():
    def provider(operation, inputs):
        if operation == "derive_candidate_abstraction":
            source_ref = inputs["ledger"][1]["id"]
            return {"claims": [{"id": "supported-c1", "claim": "supported fact", "support_refs": [source_ref], "support_type": "source_established"}]}
        if operation == "select_learning_evidence":
            return {"learning_action": "practice", "evidence_path": [{"step": "test"}]}
        if operation == "critique_instructional_abstraction":
            return {"findings": [{"finding_id": "f1", "target_path": "action", "finding_type": "ACTION_UNSUPPORTED", "recommended_disposition": "REJECT"}]}
        return {}

    result = StagedAbstractionPipeline().run(make_intent(), make_source(), provider=provider)
    assert result.final["learning_action"] == "unresolved"
    assert next(item for item in result.final["claims"] if item["id"] == "supported-c1")["status"] == "SUPPORTED"
