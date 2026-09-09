from rex_learning import CurriculumIntent, SourceContext
from rex_learning.lightweight_abstraction import (
    assess_risks,
    apply_targeted_critic,
    normalize_integrated_answer,
    validate_integrated_answer,
)


def make_intent(**overrides):
    data = {
        "key": "target",
        "title": "Target capability",
        "target_capability": "Implement and test reliable protocol clients.",
        "competence_level": "working",
        "intended_use": ["implement", "test"],
        "intended_transfer": ["unseen clients"],
        "environment": {},
        "temporal_requirements": "",
        "evaluation_criteria": ["compile", "test"],
    }
    data.update(overrides)
    return CurriculumIntent.from_dict(data)


def make_source(**overrides):
    data = {
        "source_id": "book",
        "unit_id": "unit-1",
        "title": "Protocol rule",
        "text": "The two-byte field uses network byte order and precedes the payload.",
        "source_refs": ["book#1"],
        "source_quality": "adequate",
    }
    data.update(overrides)
    return SourceContext.from_dict(data)


def test_normalization_preserves_integrated_semantics_and_adds_claim_ids():
    answer = {
        "source_purpose": "Specify a wire-format rule",
        "source_facts": ["The field precedes the payload"],
        "source_invariant": {"claim": "The field precedes the payload", "support_refs": ["book#1"]},
        "exact_details": ["two-byte", "network byte order"],
        "transferable_abstractions": ["Respect explicit wire ordering when parsing"],
        "goal_relevance": {"classification": "core", "support_refs": ["target"]},
        "currentness_applicable": False,
        "currentness_status": "UNKNOWN",
        "learning_action": {"action": "practice", "basis_refs": ["book#1", "target"]},
        "evidence_path": {"evidence": ["compile", "test"], "basis_refs": ["target"]},
        "claims": [{"claim": "The field precedes the payload", "support_refs": ["book#1"], "support_type": "source_established"}],
    }
    normalized = normalize_integrated_answer(answer)
    assert normalized["claims"][0]["id"] == "claim-1"
    assert normalized["exact_details"] == ["two-byte", "network byte order"]
    assert normalized["learning_action"]["action"] == "practice"


def test_validation_quarantines_bad_support_without_inventing_a_replacement():
    answer = normalize_integrated_answer({
        "claims": [
            {"id": "c1", "claim": "The source recommends this current procedure", "support_refs": [], "support_type": "source_established"},
            {"id": "c2", "claim": "The learner should test it", "support_refs": ["target"], "support_type": "intent_established"},
        ]
    })
    result = validate_integrated_answer(answer, make_source(), make_intent())
    assert result.canonical["claims"][0]["status"] == "QUARANTINED"
    assert result.canonical["claims"][0]["claim"] == "The source recommends this current procedure"
    assert result.canonical["claims"][1]["status"] == "ADMISSIBLE"
    assert any(issue["code"] == "MISSING_SUPPORT" for issue in result.issues)


def test_risk_detector_is_selective_and_does_not_trigger_for_simple_stable_material():
    safe = normalize_integrated_answer({
        "source_purpose": "Explain a stable mathematical implication",
        "source_invariant": {"claim": "If A implies B and A is true, B is true", "support_refs": ["book#1"]},
        "source_facts": ["A implies B"],
        "goal_relevance": {"classification": "core", "support_refs": ["target"]},
        "currentness_applicable": False,
        "currentness_status": "NOT_APPLICABLE",
        "learning_action": {"action": "normal-study", "basis_refs": ["target"]},
        "evidence_path": {"evidence": ["explain"], "basis_refs": ["target"]},
    })
    result = validate_integrated_answer(safe, make_source(text="If A implies B and A is true, B is true."), make_intent())
    assert assess_risks(result.canonical, make_source(text="If A implies B and A is true, B is true."), make_intent()) == []


def test_risk_detector_flags_unresolved_currentness_and_weak_action_support():
    answer = normalize_integrated_answer({
        "source_facts": ["Use endpoint /v2/items"],
        "currentness_applicable": True,
        "currentness_status": "UNKNOWN",
        "learning_action": {"action": "execute", "basis_refs": []},
        "evidence_path": {"evidence": [], "basis_refs": []},
    })
    source = make_source(title="API procedure", text="Use endpoint /v2/items in production.")
    validated = validate_integrated_answer(answer, source, make_intent(temporal_requirements="current production use required"))
    risks = assess_risks(validated.canonical, source, make_intent(temporal_requirements="current production use required"))
    assert {risk["code"] for risk in risks} >= {"CURRENTNESS_RELEVANT_BUT_UNRESOLVED", "ACTION_SUPPORT_WEAK", "EVIDENCE_PATH_SUPPORT_WEAK"}


def test_targeted_critic_only_changes_named_claim_and_preserves_unrelated_claims():
    answer = normalize_integrated_answer({
        "claims": [
            {"id": "c1", "claim": "The endpoint is current", "support_refs": ["book#1"], "support_type": "source_established"},
            {"id": "c2", "claim": "The endpoint path is /v2/items", "support_refs": ["book#1"], "support_type": "source_established"},
        ],
        "learning_action": {"action": "execute", "basis_refs": ["book#1"]},
    })
    revised = apply_targeted_critic(answer, {
        "target_claims": [{"id": "c1", "decision": "mark_unknown", "reasoning_basis": "No current evidence supplied", "support_refs": []}],
        "affected_learning_action": {"action": "verify_current_source", "basis_refs": []},
    })
    assert revised["claims"][0]["status"] == "UNKNOWN"
    assert revised["claims"][1]["status"] == "PROPOSED"
    assert revised["learning_action"]["action"] == "verify_current_source"


def test_normalization_preserves_aliases_and_scalar_semantics_with_provenance():
    raw = {
        "source_summary": "POST /v3/items accepts JSON field name",
        "invariants": [{"claim": "Use POST /v3/items", "support_refs": ["book#1"]}],
        "goal_relevance": "Implement and test the current items API.",
        "learning_action": "Write compatibility tests for POST /v3/items.",
        "evidence_path": "Run the compatibility suite and inspect the wire request.",
        "currentness": {"status": "CURRENT_AS_OF_SOURCE_DATE", "support_refs": ["book#1"]},
    }
    normalized = normalize_integrated_answer(raw)
    assert normalized["source_invariant"]["claim"] == "Use POST /v3/items"
    assert normalized["goal_relevance"]["value"] == raw["goal_relevance"]
    assert normalized["learning_action"]["value"] == raw["learning_action"]
    assert normalized["evidence_path"]["value"] == raw["evidence_path"]
    assert normalized["representation_fidelity"]["status"] == "PRESERVED_WITH_ADAPTER"
    assert normalized["representation_fidelity"]["raw_fields"]["learning_action"] == raw["learning_action"]


def test_evidence_path_scalar_list_is_wrapped_without_schema_loss():
    raw = {"evidence_path": ["v5://v5-03", "cap-03"]}
    normalized = normalize_integrated_answer(raw)
    assert normalized["evidence_path"]["evidence"] == raw["evidence_path"]
    assert normalized["evidence_path"]["raw_value"] == raw["evidence_path"]
    assert normalized["representation_fidelity"]["status"] == "PRESERVED_WITH_ADAPTER"
    assert normalized["representation_fidelity"]["incompatibilities"] == []


def test_unknown_shape_is_retained_and_explicitly_incompatible():
    raw = {"learning_action": ["write tests", {"unexpected": True}]}
    normalized = normalize_integrated_answer(raw)
    assert normalized["learning_action"]["raw_value"] == raw["learning_action"]
    assert normalized["representation_fidelity"]["status"] == "SCHEMA_INCOMPATIBLE"
    assert normalized["representation_fidelity"]["incompatibilities"][0]["field"] == "learning_action"


def test_faithful_surface_retains_provider_answer_verbatim():
    raw = {"invariant_core": "Keep --workers 4 exactly", "learning_action": "Verify current docs."}
    normalized = normalize_integrated_answer(raw)
    surface = normalized["representation_fidelity"]["faithful_surface"]
    assert surface["invariant_core"] == raw["invariant_core"]
    assert surface["learning_action"] == raw["learning_action"]
