from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile
from typing import Any, cast

import pytest

from rex_learning.learner import OpenAICompatibleLearner

from rex_learning import ExecutableArtifactGrader, ExecutableGrader, LearningEngine, LearningError, LearningStore, TrustedEvaluator, bind_independent_competence_test_design, build_learning_treatments, define_test_from_independent_design, discover_capabilities, discover_capabilities_from_units, derive_independent_behavioral_evidence, execute_bound_independent_test, execute_learning_treatments, integrate_existing_skill_plan, plan_existing_skill_integrations, preserve_conflict_plan, requalify_existing_skill_plan, select_learning_treatments
from rex_learning.ingestion import ingest_epub
from rex_learning.capability_discovery import _behavioral_evidence_candidate, _coherent_independent_phase, _missing_learner_contract_requirements, build_competence_test_requests, normalize_independent_test_design, select_learning_proposal_keys
from rex_learning.engine import CeilingBaselineError


def test_normalize_independent_test_design_accepts_strict_phases_envelope() -> None:
    phases = ("baseline", "practice", "posttest", "retest", "negative", "fresh", "control")
    raw = {
        "phases": [
            {
                "phase": phase,
                "case_id": f"{phase}-case",
                "learner_visible_task": f"LEARNER-VISIBLE FIXTURE: complete this {phase} unit-test task. OUTPUT JSON SHAPE: responses.",
                "artifact_schema": {
                    "artifact_type": "object",
                    "required_fields": ["assertions"],
                    "field_types": {"assertions": "array"},
                    "assertions": [
                        {"op": "min_items", "field": "assertions", "value": 1},
                        {"field": "assertions", "type_is": "array"},
                    ],
                },
            }
            for phase in phases
        ]
    }
    normalized = normalize_independent_test_design(raw)
    assert set(normalized["phase_cases"]) == set(phases)
    assert normalized["phase_tasks"]["posttest"].startswith("LEARNER-VISIBLE FIXTURE:")
    contract = __import__("json").loads(normalized["evaluation_contracts"]["baseline"])
    assert contract["assertions"] == [
        {"path": "$.assertions", "operator": "min_items", "value": 1},
        {"path": "$.assertions", "operator": "type_is", "value": "array"},
    ]


def test_normalize_independent_test_design_accepts_assertion_type_aliases() -> None:
    phases = ("baseline", "practice", "posttest", "retest", "negative", "fresh", "control")
    raw = {
        "phases": [
            {
                "phase": phase,
                "case_id": f"{phase}-case",
                "task": f"LEARNER-VISIBLE FIXTURE: complete this {phase} task. OUTPUT JSON SHAPE: object.",
                "artifact_schema": {
                    "artifact_type": "object",
                    "required_fields": ["decision"],
                    "field_types": {"decision": "string"},
                    "assertions": [{"assertion_type": "type_is", "field": "decision", "value_type": "string"}],
                },
            }
            for phase in phases
        ]
    }
    normalized = normalize_independent_test_design(raw)
    contract = json.loads(normalized["evaluation_contracts"]["baseline"])
    assert contract["assertions"] == [{"path": "$.decision", "operator": "type_is", "value": "string"}]


def test_normalize_independent_test_design_accepts_phase_keyed_envelope() -> None:
    phases = ("baseline", "practice", "posttest", "retest", "negative", "fresh", "control")
    raw = {
        phase: {
            "case_id": f"{phase}-case",
            "task": f"LEARNER-VISIBLE FIXTURE: complete this {phase} decision task. OUTPUT JSON SHAPE: responses.",
            "artifact_schema": {
                "artifact_type": "decision_record",
                "required_fields": ["decision"],
                "field_types": {"decision": "string"},
                "assertions": [{"path": "decision", "operator": "type_is", "value": "string"}],
            },
        }
        for phase in phases
    }
    normalized = normalize_independent_test_design(raw)
    assert set(normalized["phase_cases"]) == set(phases)
    assert normalized["phase_cases"]["control"] == "control-case"


def test_normalize_independent_test_design_accepts_fixture_task_alias() -> None:
    phases = ("baseline", "practice", "posttest", "retest", "negative", "fresh", "control")
    raw = {
        "phases": [
            {
                "phase": phase,
                "case_id": f"{phase}-case",
                "learner_visible_fixture_task": f"LEARNER-VISIBLE FIXTURE: complete this {phase} task. OUTPUT JSON SHAPE: object.",
                "artifact_schema": {
                    "artifact_type": "object",
                    "required_fields": ["decision"],
                    "field_types": {"decision": "string"},
                    "assertions": [{"field": "decision", "type_is": "string"}],
                },
            }
            for phase in phases
        ]
    }
    normalized = normalize_independent_test_design(raw)
    assert normalized["phase_tasks"]["baseline"].startswith("LEARNER-VISIBLE FIXTURE:")


def test_normalize_independent_test_design_rejects_incomplete_phase_envelope() -> None:
    with pytest.raises(LearningError, match="exactly seven"):
        normalize_independent_test_design({"phases": []})


def test_normalize_independent_test_design_rejects_solution_scaffolding_in_baseline() -> None:
    phases = ("baseline", "practice", "posttest", "retest", "negative", "fresh", "control")
    raw = {
        "phases": [
            {
                "phase": phase,
                "case_id": f"{phase}-case",
                "learner_visible_task": (
                    "LEARNER-VISIBLE FIXTURE: "
                    + ("{\"formula\":\"WIP = throughput * cycle time\",\"stages\":[]}" if phase == "baseline" else "{}")
                    + " OUTPUT JSON SHAPE: responses. decision and rationale."
                ),
                "artifact_schema": {
                    "artifact_type": "object",
                    "required_fields": ["decision", "rationale"],
                    "field_types": {"decision": "string", "rationale": "string"},
                    "assertions": [],
                },
            }
            for phase in phases
        ]
    }
    with pytest.raises(LearningError, match="solution scaffolding"):
        normalize_independent_test_design(raw)


def test_normalize_independent_test_design_rejects_text_solution_scaffolding_in_baseline() -> None:
    phases = ("baseline", "practice", "posttest", "retest", "negative", "fresh", "control")
    raw = {
        "phases": [
            {
                "phase": phase,
                "case_id": f"{phase}-case",
                "learner_visible_task": (
                    "LEARNER-VISIBLE FIXTURE: {} OUTPUT JSON SHAPE: responses. "
                    + ("Formula: WIP = throughput * cycle time. " if phase == "baseline" else "")
                    + "Return decision and rationale."
                ),
                "artifact_schema": {
                    "artifact_type": "object",
                    "required_fields": ["decision", "rationale"],
                    "field_types": {"decision": "string", "rationale": "string"},
                    "assertions": [],
                },
            }
            for phase in phases
        ]
    }
    with pytest.raises(LearningError, match="solution scaffolding"):
        normalize_independent_test_design(raw)


def test_normalize_independent_test_design_rejects_solution_conclusions_in_any_phase() -> None:
    phases = ("baseline", "practice", "posttest", "retest", "negative", "fresh", "control")
    raw = {
        "phases": [
            {
                "phase": phase,
                "case_id": f"{phase}-case",
                "learner_visible_task": (
                    "LEARNER-VISIBLE FIXTURE: {} OUTPUT JSON SHAPE: responses. "
                    + (
                        "The real dependency defect is the changed condition. "
                        if phase == "posttest"
                        else "Return decision and rationale."
                    )
                ),
                "artifact_schema": {
                    "artifact_type": "object",
                    "required_fields": ["decision", "rationale"],
                    "field_types": {"decision": "string", "rationale": "string"},
                    "assertions": [],
                },
            }
            for phase in phases
        ]
    }
    with pytest.raises(LearningError, match="expected answer"):
        normalize_independent_test_design(raw)


def test_normalize_independent_test_design_rejects_mutation_observations_in_any_phase() -> None:
    phases = ("baseline", "practice", "posttest", "retest", "negative", "fresh", "control")
    raw = {
        "phases": [
            {
                "phase": phase,
                "case_id": f"{phase}-case",
                "learner_visible_task": (
                    "LEARNER-VISIBLE FIXTURE: {} OUTPUT JSON SHAPE: responses. "
                    + (
                        "Mutation observation: the boundary mutant survived the existing test. "
                        if phase == "retest"
                        else "Return decision and rationale."
                    )
                ),
                "artifact_schema": {
                    "artifact_type": "object",
                    "required_fields": ["decision", "rationale"],
                    "field_types": {"decision": "string", "rationale": "string"},
                    "assertions": [],
                },
            }
            for phase in phases
        ]
    }
    with pytest.raises(LearningError, match="expected answer"):
        normalize_independent_test_design(raw)


def test_normalize_independent_test_design_rejects_conflicting_todo_instructions() -> None:
    phases = ("baseline", "practice", "posttest", "retest", "negative", "fresh", "control")
    raw = {
        "phases": [
            {
                "phase": phase,
                "case_id": f"{phase}-case",
                "learner_visible_task": (
                    "LEARNER-VISIBLE FIXTURE: replace the TODO with the required assertion, "
                    "but preserve the literal TODO in the submitted artifact."
                ),
                "artifact_schema": {
                    "artifact_type": "object",
                    "required_fields": ["decision"],
                    "field_types": {"decision": "string"},
                    "assertions": [],
                },
            }
            for phase in phases
        ]
    }
    with pytest.raises(LearningError, match="contradictory"):
        normalize_independent_test_design(raw)


def test_normalize_independent_test_design_accepts_pending_fixture_decision_rule() -> None:
    phases = ("baseline", "practice", "posttest", "retest", "negative", "fresh", "control")
    raw = {
        "phases": [
            {
                "phase": phase,
                "case_id": f"{phase}-case",
                "learner_visible_task": (
                    "LEARNER-VISIBLE FIXTURE: "
                    + ("{\"evaluation\": {\"decision_rule\": \"pending\"}}" if phase == "baseline" else "{}")
                    + " OUTPUT JSON SHAPE: decision and rationale."
                ),
                "artifact_schema": {
                    "artifact_type": "object",
                    "required_fields": ["decision", "rationale"],
                    "field_types": {"decision": "string", "rationale": "string"},
                    "assertions": [{"field": "decision", "type_is": "string"}],
                },
            }
            for phase in phases
        ]
    }
    normalized = normalize_independent_test_design(raw)
    assert normalized["phase_cases"]["baseline"] == "baseline-case"


def test_normalize_independent_test_design_rejects_implementation_task_with_decision_only_artifact() -> None:
    phases = ("baseline", "practice", "posttest", "retest", "negative", "fresh", "control")
    raw = {
        "phases": [
            {
                "phase": phase,
                "case_id": f"{phase}-case",
                "learner_visible_task": (
                    "LEARNER-VISIBLE FIXTURE: implement and verify the requested pipeline. "
                    "Report the chosen decision and rationale without supplying implementation output."
                ),
                "artifact_schema": {
                    "artifact_type": "object",
                    "required_fields": ["decision", "rationale"],
                    "field_types": {"decision": "string", "rationale": "string"},
                    "assertions": [],
                },
            }
            for phase in phases
        ]
    }
    with pytest.raises(LearningError, match="implementation artifact surface"):
        normalize_independent_test_design(raw)


def test_learner_contract_requirements_reject_hidden_cardinality_and_type_constraints() -> None:
    contract = json.dumps({
        "artifact_type": "object",
        "required_fields": ["before_after", "behavior_preserved"],
        "field_types": {"before_after": "array", "behavior_preserved": "boolean"},
        "assertions": [
            {"path": "$.before_after", "operator": "min_items", "value": 2},
            {"path": "$.behavior_preserved", "operator": "type_is", "value": "boolean"},
        ],
    })
    task = "Return before/after material and a behavior-preservation claim. REQUIRED FIELDS: before_after, behavior_preserved"
    gaps = _missing_learner_contract_requirements(task, contract)
    assert "$.before_after requires an array with at least 2 items" in gaps
    assert "$.behavior_preserved requires a boolean value" in gaps


def test_learner_contract_requirements_accept_explicit_min_items_operator() -> None:
    contract = json.dumps({
        "artifact_type": "object",
        "required_fields": ["constraints"],
        "field_types": {"constraints": "array"},
        "assertions": [{"path": "$.constraints", "operator": "min_items", "value": 3}],
    })
    task = "REQUIRED OUTPUT FIELDS: constraints (array). CONSTRAINTS: $.constraints must use type_is array and min_items 3."
    assert _missing_learner_contract_requirements(task, contract) == []


def test_learner_contract_requirements_accept_plural_boolean_declaration() -> None:
    contract = json.dumps({
        "artifact_type": "object",
        "required_fields": ["original_passes", "mutant_fails"],
        "field_types": {"original_passes": "boolean", "mutant_fails": "boolean"},
        "assertions": [],
    })
    task = "verification must contain original_passes and mutant_fails booleans"
    assert _missing_learner_contract_requirements(task, contract) == []


def test_learner_contract_requirements_keep_exact_scalar_values_evaluator_only() -> None:
    contract = json.dumps({
        "artifact_type": "object",
        "required_fields": ["mode", "enabled"],
        "field_types": {"mode": "string", "enabled": "boolean"},
        "assertions": [
            {"path": "$.mode", "operator": "value_is", "value": "target_unit_testing"},
            {"path": "$.enabled", "operator": "value_is", "value": False},
        ],
    })
    gaps = _missing_learner_contract_requirements(
        "REQUIRED FIELDS: mode, enabled. mode is a string; enabled is boolean.",
        contract,
    )
    assert gaps == []


def test_treatment_selection_requires_explicit_behavioral_candidate_marker() -> None:
    proposals = [
        {"key": "implicit", "actionable": True, "kind": "procedure", "procedure": "do it"},
        {"key": "explicit", "actionable": True, "kind": "procedure", "procedure": "do it", "behavioral_evidence_candidate": True},
    ]
    assert select_learning_proposal_keys(proposals) == ["explicit"]


def test_treatment_selection_accepts_actionable_non_procedure_behavioral_capability() -> None:
    proposals = [
        {
            "key": "diagnose_mocking_gap",
            "actionable": True,
            "kind": "declarative",
            "procedure": "Compare the mocked and real dependency behavior, then identify the observable failure.",
            "behavioral_evidence_candidate": True,
        },
    ]
    assert select_learning_proposal_keys(proposals) == ["diagnose_mocking_gap"]


def test_treatment_selection_accepts_declarative_behavioral_capability_without_procedure() -> None:
    proposals = [
        {
            "key": "distinguish_defect_from_failure",
            "actionable": True,
            "kind": "declarative",
            "behavioral_evidence_candidate": True,
        },
    ]
    assert select_learning_proposal_keys(proposals) == ["distinguish_defect_from_failure"]


def test_treatment_selection_prioritizes_explicit_baseline_headroom_without_treating_it_as_evidence() -> None:
    proposals = [
        {"key": "ceiling", "actionable": True, "kind": "procedure", "procedure": "do it", "behavioral_evidence_candidate": True, "baseline_headroom": 0.0},
        {"key": "learnable", "actionable": True, "kind": "procedure", "procedure": "do it", "behavioral_evidence_candidate": True, "baseline_headroom": 0.8},
        {"key": "unknown", "actionable": True, "kind": "procedure", "procedure": "do it", "behavioral_evidence_candidate": True},
    ]
    assert select_learning_proposal_keys(proposals, limit=2) == ["learnable", "ceiling"]


def test_treatment_selection_defers_measured_ceiling_and_keeps_invalid_scores_unknown() -> None:
    proposals = [
        {"key": "ceiling", "actionable": True, "kind": "procedure", "procedure": "do it", "behavioral_evidence_candidate": True},
        {"key": "learnable", "actionable": True, "kind": "procedure", "procedure": "do it", "behavioral_evidence_candidate": True},
        {"key": "unknown", "actionable": True, "kind": "procedure", "procedure": "do it", "behavioral_evidence_candidate": True},
    ]
    scores = {"ceiling": 1.0, "learnable": 0.0, "unknown": "not-a-score"}
    assert select_learning_proposal_keys(proposals, limit=2, baseline_scores=scores) == ["learnable", "unknown"]
    persisted_scores = {"ceiling": {"score": 1.0}, "learnable": {"score": 0.0}}
    assert select_learning_proposal_keys(proposals, limit=2, baseline_scores=persisted_scores) == ["learnable", "unknown"]


def test_treatment_selection_breaks_unknown_headroom_ties_by_stable_key() -> None:
    proposals = [
        {"key": "zulu", "actionable": True, "kind": "procedure", "procedure": "do it", "behavioral_evidence_candidate": True},
        {"key": "alpha", "actionable": True, "kind": "procedure", "procedure": "do it", "behavioral_evidence_candidate": True},
    ]
    assert select_learning_proposal_keys(proposals, limit=None) == ["alpha", "zulu"]


def test_behavioral_evidence_screen_accepts_explicit_correctness_criteria() -> None:
    from rex_learning.capability_discovery import _behavioral_evidence_candidate

    assert _behavioral_evidence_candidate(
        "Calculate the result and correctly identify the appropriate action in a novel case."
    ) is True


def test_behavioral_evidence_screen_accepts_inflected_observable_action() -> None:
    from rex_learning.capability_discovery import _behavioral_evidence_candidate

    assert _behavioral_evidence_candidate(
        "The learner correctly identifies the strategy and cites two constraints."
    ) is True


def test_treatment_selection_forwards_trusted_baseline_scores() -> None:
    discovery = {
        "schema": "rex-learning-capability-discovery-v1",
        "proposals": [
            {
                "id": "proposal-ceiling",
                "key": "ceiling",
                "curriculum_id": "curriculum-1",
                "source_id": "source-1",
                "source_hash": "hash-1",
                "capability": "Already solved procedure",
                "kind": "procedure",
                "actionable": True,
                "behavioral_evidence_candidate": True,
                "procedure": "Do the procedure.",
                "proposed_practice": "Practice the procedure.",
                "proposed_evidence": "Complete the procedure.",
                "relationship": "acquire",
            },
            {
                "id": "proposal-headroom",
                "key": "headroom",
                "curriculum_id": "curriculum-1",
                "source_id": "source-1",
                "source_hash": "hash-1",
                "capability": "Unsolved procedure",
                "kind": "procedure",
                "actionable": True,
                "behavioral_evidence_candidate": True,
                "procedure": "Do the other procedure.",
                "proposed_practice": "Practice the other procedure.",
                "proposed_evidence": "Complete the other procedure.",
                "relationship": "acquire",
            },
        ],
    }
    selected = select_learning_treatments(
        discovery,
        limit=1,
        baseline_scores={"ceiling": 1.0, "headroom": 0.0},
    )
    assert [item["intent_key"] for item in selected] == ["headroom"]
from scripts.run_preserved_discovery_treatment_experiment import request_suffix, select_experiment_proposals


class DiscoveryLearner:
    provider = "fixture-provider"
    session_id = "discovery-session-1"

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def answer(self, *, task: str, material: str = "", revision: str = "") -> dict:
        self.calls.append((task, material))
        return {
            "responses": [
                {
                    "case": "capability_discovery",
                    "answer": {
                        "capabilities": [
                            {
                                "key": "extract-functions",
                                "capability": "Recognize repeated logic and extract a function",
                                "kind": "procedure",
                                "instructional_purpose": "Turn repeated logic into reusable behavior.",
                                "generalizable_principle": "Extract a function when a named behavior is repeated.",
                                "procedure": ["Identify repeated logic", "Choose parameters", "Return the computed value"],
                                "prerequisites": ["Understand control flow"],
                                "limitations": ["Do not extract trivial one-off expressions automatically."],
                                "proposed_practice": "Refactor a repeated block into a tested function.",
                                "proposed_evidence": "Compile and run tests; the expected assertions pass for the refactored program.",
                            },
                            {
                                "key": "function-reference",
                                "capability": "Understand that a function reference is not its return value",
                                "kind": "declarative",
                                "instructional_purpose": "Distinguish callable behavior from a computed result.",
                                "exact_details": ["Calling requires parentheses in the language example."],
                                "uncertainties": ["Exact syntax depends on the language version."],
                            },
                            {
                                "key": "author-biography",
                                "capability": "Know the author's biography",
                                "kind": "context",
                                "instructional_purpose": "Provide historical context.",
                            },
                        ]
                    },
                }
            ]
        }


def test_live_discovery_prompt_requires_single_wrapped_response(tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    def requester(_url: str, body: bytes, _headers: dict[str, str], _timeout: float) -> bytes:
        payload = json.loads(body)
        captured["task"] = payload["messages"][1]["content"]
        content = json.dumps({"responses": [{"case": "capability_discovery", "answer": {"capabilities": [{
            "key": "observed-procedure", "capability": "Perform an observed procedure", "kind": "procedure",
            "procedure": ["Observe", "Act"], "proposed_practice": "Perform the procedure on a fresh case",
            "proposed_evidence": "The procedure executes and the output is correct",
        }]}}]})
        return json.dumps({"choices": [{"message": {"content": content}}]}).encode()

    result = discover_capabilities(
        engine=LearningEngine(LearningStore(tmp_path / "learning")),
        learner=OpenAICompatibleLearner("http://test", "test-model", provider="test", session_id="discovery-wire", requester=requester),
        source_title="Wire contract", source_text="Instructional material.",
        educational_objective="Identify transferable procedures",
    )
    assert "exactly one response object" in captured["task"]
    assert "do not repeat the case ID" in captured["task"]
    assert result["actionable_proposal_keys"] == ["observed-procedure"]


def test_empty_discovery_response_is_persisted_as_failed_attempt(tmp_path: Path) -> None:
    class EmptyDiscoveryLearner:
        provider = "fixture-empty-provider"
        session_id = "empty-discovery-session"

        def answer(self, *, task: str, material: str = "", revision: str = "") -> dict[str, Any]:
            return {"responses": [{"case": "capability_discovery", "answer": {"capabilities": []}}]}

    engine = LearningEngine(LearningStore(tmp_path / "learning"))
    with pytest.raises(LearningError, match="non-empty capabilities list") as raised:
        discover_capabilities(
            engine=engine, learner=EmptyDiscoveryLearner(), source_title="Empty discovery",
            source_text="Instructional material.", educational_objective="Find transferable capabilities",
        )
    assert raised.value.details["code"] == "empty_capabilities"
    attempts = engine.store.list("capability_discovery_attempts")
    assert len(attempts) == 1
    assert attempts[0]["status"] == "failed"
    assert attempts[0]["raw_surface"] == {"capabilities": []}
    assert attempts[0]["provider"]["session_id"] == "empty-discovery-session"


def test_large_discovery_material_is_packetized_and_aggregated(tmp_path: Path) -> None:
    class PacketLearner:
        provider = "fixture-packet-provider"
        session_id = "packet-discovery-session"

        def __init__(self) -> None:
            self.materials: list[str] = []

        def answer(self, *, task: str, material: str = "", revision: str = "") -> dict[str, Any]:
            self.materials.append(material)
            index = len(self.materials)
            return {"responses": [{"case": "capability_discovery", "answer": {"capabilities": [{
                "key": f"packet-{index}", "capability": f"Perform packet procedure {index}", "kind": "procedure",
                "procedure": ["Observe", "Act"],
                "proposed_practice": "Perform this procedure on a fresh case",
                "proposed_evidence": "The procedure executes and produces the expected output",
            }]}}]}

    learner = PacketLearner()
    engine = LearningEngine(LearningStore(tmp_path / "learning"))
    result = discover_capabilities(
        engine=engine,
        learner=learner,
        source_title="Large source",
        source_text="instructional material " * 2000,
        educational_objective="Find transferable procedures",
    )

    assert len(learner.materials) > 1
    assert all(len(material) <= 12000 for material in learner.materials)
    assert len(result["proposals"]) == len(learner.materials)
    attempt = engine.store.list("capability_discovery_attempts")[0]
    assert attempt["status"] == "completed"
    assert attempt["provider"]["packet_count"] == len(learner.materials)
    assert len(attempt["raw_provider_results"]) == len(learner.materials)


def test_coverage_analysis_evidence_is_behaviorally_selectable() -> None:
    assert _behavioral_evidence_candidate(
        "Calculate statement and branch coverage for fresh test inputs and report the expected coverage results."
    ) is True


def test_coverage_analysis_source_evidence_with_uncovered_paths_is_behaviorally_selectable() -> None:
    assert _behavioral_evidence_candidate(
        "Correct calculation of coverage percentages and identification of uncovered paths."
    ) is True


class SingleCapabilityLearner(DiscoveryLearner):
    def answer(self, *, task: str, material: str = "", revision: str = "") -> dict:
        self.calls.append((task, material))
        return {"responses": [{"case": "capability_discovery", "answer": {
            "key": "single-capability", "capability": "Perform a bounded diagnostic procedure",
            "kind": "procedure", "source-grounded procedure": ["Observe", "Test"],
            "proposed_practice": "Diagnose a fresh case", "proposed_evidence": "The cause is identified",
        }}]}


class MultiResponseCapabilityLearner(DiscoveryLearner):
    def answer(self, *, task: str, material: str = "", revision: str = "") -> dict:
        self.calls.append((task, material))
        return {"responses": [
            {"case": "capability_discovery", "answer": {"key": "one", "capability": "Perform inspection", "kind": "procedure", "procedure": ["Inspect"], "proposed_practice": "Inspect a case", "proposed_evidence": "Inspection is complete"}},
            {"case": "capability_discovery", "answer": {"key": "two", "capability": "Perform verification", "kind": "procedure", "procedure": ["Verify"], "proposed_practice": "Verify a case", "proposed_evidence": "Verification is complete"}},
        ]}


def test_discovery_normalizes_underscore_source_grounded_procedure_alias(tmp_path: Path) -> None:
    class UnderscoreProcedureLearner(SingleCapabilityLearner):
        def answer(self, *, task: str, material: str = "", revision: str = "") -> dict:
            self.calls.append((task, material))
            return {"responses": [{"case": "capability_discovery", "answer": {
                "key": "single-capability", "capability": "Perform a bounded diagnostic procedure",
                "kind": "procedure", "source_grounded_procedure": ["Observe", "Test"],
                "proposed_practice": "Diagnose a fresh case", "proposed_evidence": "The cause is identified",
            }}]}

    learner = UnderscoreProcedureLearner()
    engine = LearningEngine(LearningStore(tmp_path / "learning"))
    curriculum = engine.create_curriculum("Procedure chapter", "Improve diagnosis")

    result = discover_capabilities(
        engine=engine,
        learner=learner,
        source_title="Procedure chapter",
        source_text="Observe and test the case.",
        educational_objective="Improve diagnosis",
        curriculum_id=curriculum["id"],
    )

    assert result["proposals"][0]["procedure"] == "1. Observe\n2. Test"


def test_discovery_persists_durable_curriculum_intent_and_binds_proposals(tmp_path: Path) -> None:
    engine = LearningEngine(LearningStore(tmp_path / "learning"))
    curriculum = engine.create_curriculum("Functions chapter", "Become better at programming")

    result = discover_capabilities(
        engine=engine,
        learner=DiscoveryLearner(),
        source_title="Functions chapter",
        source_text="Repeated logic can become a function.",
        educational_objective="Become better at programming",
        curriculum_id=curriculum["id"],
    )

    intent = result["curriculum_intent"]
    assert intent["schema"] == "rex-learning-curriculum-intent-v1"
    assert intent["intent"]["target_capability"] == "Become better at programming"
    assert intent["provenance"]["source"] == "capability_discovery"
    assert len(engine.store.list("curriculum_intents")) == 1
    assert all(proposal["curriculum_intent_id"] == intent["id"] for proposal in result["proposals"])


def test_discovery_reuses_curriculum_intent_across_source_units(tmp_path: Path) -> None:
    engine = LearningEngine(LearningStore(tmp_path / "learning"))
    curriculum = engine.create_curriculum("Programming course", "Become better at programming")

    first = discover_capabilities(
        engine=engine,
        learner=DiscoveryLearner(),
        source_title="Chapter one",
        source_text="Repeated logic can become a function.",
        educational_objective="Become better at programming",
        curriculum_id=curriculum["id"],
    )
    second = discover_capabilities(
        engine=engine,
        learner=DiscoveryLearner(),
        source_title="Chapter two",
        source_text="Parameters make reusable behavior flexible.",
        educational_objective="Become better at programming",
        curriculum_id=curriculum["id"],
    )

    assert first["curriculum_intent"]["id"] == second["curriculum_intent"]["id"]
    assert second["curriculum_intent"]["intent"]["title"] == curriculum["title"]
    assert len(engine.store.list("curriculum_intents")) == 1
    assert all(proposal["curriculum_intent_id"] == first["curriculum_intent"]["id"] for proposal in second["proposals"])


class AcquisitionLearner:
    provider = "acquisition-fixture"

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.materials: list[str] = []

    def answer(self, *, task: str, material: str = "", revision: str = "") -> dict:
        self.materials.append(material)
        if task.startswith("candidate"):
            return {"responses": [], "capability": "Perform inspection", "procedure": "Inspect the case and record the result.", "applicability": ["inspection tasks"]}
        case = task.split("|case=", 1)[1].split("|", 1)[0]
        return {"responses": [{"case": case, "answer": "not-applicable" if case == "negative" else ("wrong" if case == "pre" else "inspection-pass")}]}


class NoTreatmentControlLearner(AcquisitionLearner):
    def answer(self, *, task: str, material: str = "", revision: str = "") -> dict:
        case = task.split("|case=", 1)[1].split("|", 1)[0]
        return {"responses": [{"case": case, "answer": "not-applicable" if case == "negative" else "wrong"}]}


def test_request_suffix_uses_explicit_behavior_fields_even_when_task_has_no_keyword() -> None:
    suffix = request_suffix("pre|case=pre|perform a held-out task", ["questions", "retrieval_result"])
    assert '"questions":["..."]' in suffix
    assert '"retrieval_result":["..."]' in suffix


def test_selection_without_limit_returns_all_behaviorally_screened_proposals() -> None:
    from rex_learning.capability_discovery import select_learning_proposal_keys

    proposals = [
        {"key": "first", "actionable": True, "kind": "procedure", "procedure": "Inspect", "behavioral_evidence_candidate": True},
        {"key": "second", "actionable": True, "kind": "workflow", "procedure": "Verify", "behavioral_evidence_candidate": True},
        {"key": "rejected", "actionable": True, "kind": "procedure", "procedure": "Record", "behavioral_evidence_candidate": False},
    ]

    assert select_learning_proposal_keys(proposals, limit=None) == ["first", "second"]


def test_discovery_proposes_capabilities_without_target_labels_and_persists_relationships(tmp_path: Path) -> None:
    learner = DiscoveryLearner()
    engine = LearningEngine(LearningStore(tmp_path / "learning"))
    curriculum = engine.create_curriculum("Functions chapter", "Become better at programming")
    existing = engine.create_skill_hypothesis(
        curriculum["id"],
        {
            "key": "function-decomposition",
            "kind": "procedure",
            "claim": "Extract repeated logic into functions",
            "applicability": ["reusable behavior"],
        },
    )
    existing["state"] = "demonstrated"
    engine.store.write("skills", existing["id"], existing)

    result = discover_capabilities(
        engine=engine,
        learner=learner,
        source_title="Functions chapter",
        source_text="Repeated logic can become a function with useful parameters and a return value.",
        educational_objective="Become better at programming",
        curriculum_id=curriculum["id"],
        source_locator="book.xhtml#functions",
    )

    assert len(result["proposals"]) == 3
    assert {proposal["relationship"] for proposal in result["proposals"]} >= {"refine", "context"}
    refined = next(item for item in result["proposals"] if item["key"] == "extract-functions")
    assert refined["relationship"] == "refine"
    assert refined["existing_skill_ids"] == [existing["id"]]
    assert refined["proposed_practice"]
    assert refined["source_refs"]
    assert refined["procedure"] == "1. Identify repeated logic\n2. Choose parameters\n3. Return the computed value"
    assert len(engine.store.list("capability_proposals")) == 3
    assert "extract-functions" not in learner.calls[0][0]
    assert "Become better at programming" in learner.calls[0][0]


def test_discovery_deduplicates_exact_proposals_but_preserves_raw_surface(tmp_path: Path) -> None:
    class DuplicateLearner:
        provider = "duplicate-discovery-fixture"
        session_id = "duplicate-discovery-session"

        def answer(self, *, task: str, material: str = "", revision: str = "") -> dict:
            proposal = {
                "key": "repeat-procedure",
                "capability": "Perform the same diagnostic procedure",
                "kind": "procedure",
                "procedure": ["Observe", "Verify"],
                "proposed_practice": "Diagnose a fresh case",
                "proposed_evidence": "The cause is identified",
            }
            return {"responses": [{"case": "capability_discovery", "answer": {"capabilities": [proposal, dict(proposal)]}}]}

    engine = LearningEngine(LearningStore(tmp_path / "learning"))
    curriculum = engine.create_curriculum("Duplicate chapter", "Improve diagnosis")
    result = discover_capabilities(
        engine=engine,
        learner=DuplicateLearner(),
        source_title="Duplicate chapter",
        source_text="Observe symptoms and verify the suspected cause.",
        educational_objective="Improve diagnosis",
        curriculum_id=curriculum["id"],
        source_locator="lesson#diagnosis",
    )

    assert len(result["raw_surface"]["capabilities"]) == 2
    assert len(result["proposals"]) == 1
    assert len(engine.store.list("capability_proposals")) == 1


def test_discovery_preserves_materially_distinct_same_identity_proposals(tmp_path: Path) -> None:
    class VariantLearner:
        provider = "variant-discovery-fixture"
        session_id = "variant-discovery-session"

        def answer(self, *, task: str, material: str = "", revision: str = "") -> dict:
            base = {
                "key": "same-capability",
                "capability": "Perform bounded diagnosis",
                "kind": "procedure",
                "proposed_practice": "Diagnose a fresh case",
                "proposed_evidence": "Cause is identified",
            }
            first = {**base, "procedure": ["Observe", "verify"]}
            second = {**base, "procedure": ["Observe", "isolate", "verify"]}
            return {"responses": [{"case": "capability_discovery", "answer": {"capabilities": [first, second]}}]}

    engine = LearningEngine(LearningStore(tmp_path / "learning"))
    curriculum = engine.create_curriculum("Variant chapter", "Improve diagnosis")
    result = discover_capabilities(
        engine=engine,
        learner=VariantLearner(),
        source_title="Variant chapter",
        source_text="Observe symptoms and verify the suspected cause.",
        educational_objective="Improve diagnosis",
        curriculum_id=curriculum["id"],
        source_locator="lesson#diagnosis",
    )

    assert len(result["proposals"]) == 2
    assert len({proposal["id"] for proposal in result["proposals"]}) == 2
    assert {proposal["procedure"] for proposal in result["proposals"]} == {
        "1. Observe\n2. verify",
        "1. Observe\n2. isolate\n3. verify",
    }
    assert len(engine.store.list("capability_proposals")) == 2


def test_discovery_records_bounded_relationship_evidence_and_preserves_provider_labels_as_untrusted(tmp_path: Path) -> None:
    class RelationshipLearner:
        provider = "relationship-fixture"
        session_id = "relationship-session"

        def __init__(self, existing_skill_id: str) -> None:
            self.existing_skill_id = existing_skill_id

        def answer(self, *, task: str, material: str = "", revision: str = "") -> dict:
            return {"responses": [{"case": "capability_discovery", "answer": {"capabilities": [
                {
                    "key": "refined-planning", "capability": "Plan implementation interfaces",
                    "kind": "procedure", "procedure": ["Identify interfaces"],
                    "prerequisites": ["Plan implementation"],
                    "proposed_practice": "Plan interfaces for a fresh task",
                    "proposed_evidence": "The interfaces are justified",
                    "relationship": "support",
                    "contradicts": [self.existing_skill_id],
                },
                {
                    "key": "extended-planning", "capability": "Plan implementation risks",
                    "kind": "procedure", "procedure": ["Identify risks"],
                    "proposed_practice": "Plan risks for a fresh task",
                    "proposed_evidence": "The risks are justified",
                    "relationship": "not-a-supported-relationship",
                    "extends": [self.existing_skill_id, "unknown-skill-id", None],
                },
            ]}}]}

    engine = LearningEngine(LearningStore(tmp_path / "learning"))
    curriculum = engine.create_curriculum("Planning lesson", "Improve implementation planning")
    existing = engine.create_skill_hypothesis(curriculum["id"], {
        "key": "planning", "kind": "procedure", "claim": "Plan implementation", "applicability": ["software tasks"],
    })

    result = discover_capabilities(
        engine=engine, learner=RelationshipLearner(existing["id"]), source_title="Planning lesson",
        source_text="Plan implementation with interfaces and risks.", educational_objective="Improve implementation planning",
        curriculum_id=curriculum["id"],
    )

    contradiction = result["proposals"][0]
    assert contradiction["relationship"] == "contradict"
    assert contradiction["relationship_evidence"]["method"] == "explicit_existing_skill_id"
    assert contradiction["existing_skill_ids"] == [existing["id"]]
    assert contradiction["provider_relationship"] == "support"
    extension = result["proposals"][1]
    assert extension["relationship"] == "extend"
    assert extension["relationship_evidence"]["method"] == "explicit_existing_skill_id"
    assert extension["provider_relationship"] == ""


def test_discovery_ignores_malformed_relationship_id_fields_without_overriding_deterministic_result(tmp_path: Path) -> None:
    class MalformedRelationshipLearner:
        provider = "malformed-relationship-provider"
        session_id = "malformed-relationship-session"

        def answer(self, *, task: str, material: str = "", revision: str = "") -> dict:
            return {"responses": [{"case": "capability_discovery", "answer": {"capabilities": [{
                "key": "new-capability", "capability": "Perform a fresh unrelated operation", "kind": "procedure",
                "procedure": ["Perform operation"], "proposed_practice": "Perform a fresh operation",
                "proposed_evidence": "The operation succeeds", "contradicts": None, "extends": "not-a-list",
                "relationship": "contradict",
            }]}}]}

    engine = LearningEngine(LearningStore(tmp_path / "learning"))
    result = discover_capabilities(
        engine=engine, learner=MalformedRelationshipLearner(), source_title="New lesson",
        source_text="A fresh operation.", educational_objective="Learn a fresh operation",
    )

    proposal = result["proposals"][0]
    assert proposal["relationship"] == "new"
    assert proposal["existing_skill_ids"] == []
    assert proposal["relationship_evidence"]["method"] == "no_repertoire_token_match"
    assert proposal["provider_relationship"] == "contradict"


def test_discovery_preserves_raw_provider_surface_and_marks_actionable_candidates(tmp_path: Path) -> None:
    engine = LearningEngine(LearningStore(tmp_path / "learning"))
    result = discover_capabilities(
        engine=engine,
        learner=DiscoveryLearner(),
        source_title="Small lesson",
        source_text="A lesson about testing.",
        educational_objective="Improve software engineering",
    )

    assert result["provider"]["provider"] == "fixture-provider"
    assert result["provider"]["session_id"] == "discovery-session-1"
    assert result["provider"]["operation"] == "capability_discovery"
    assert result["provider"]["prompt_hash"]
    assert result["provider"]["output_hash"]
    assert result["raw_surface"]["capabilities"]
    assert result["actionable_proposal_keys"] == ["extract-functions"]
    assert result["proposals"][0]["source_hash"]


def test_discovery_preserves_valid_capability_supporting_source_refs_in_treatment(tmp_path: Path) -> None:
    class GroundedLearner:
        provider = "grounded-discovery-fixture"
        session_id = "grounded-discovery-session"

        def answer(self, *, task: str, material: str = "", revision: str = "") -> dict[str, Any]:
            return {"responses": [{"case": "capability_discovery", "answer": {"capabilities": [{
                "key": "grounded-technique", "capability": "Apply the technique to a novel case", "kind": "procedure",
                "procedure": "Identify the condition and apply the technique.",
                "supporting_source_refs": ["unit-testing#isolation"],
                "proposed_practice": "Apply the technique to a held-out case",
                "proposed_evidence": "The executable test passes for the held-out case.",
            }]}}]}

    engine = LearningEngine(LearningStore(tmp_path / "learning"))
    result = discover_capabilities(
        engine=engine, learner=GroundedLearner(), source_title="Grounded lesson",
        source_text="A lesson about isolation.", source_refs=["unit-testing#isolation", "other-unit"],
        educational_objective="Improve software engineering",
    )

    proposal = result["proposals"][0]
    assert proposal["source_refs"] == [proposal["source_id"], "unit-testing#isolation", "other-unit"]
    assert proposal["supporting_source_refs"] == ["unit-testing#isolation"]
    treatment = build_learning_treatments(result)[0]
    assert treatment["source_refs"] == ["unit-testing#isolation"]


def test_discovery_marks_behaviorally_discriminating_evidence_candidates(tmp_path: Path) -> None:
    engine = LearningEngine(LearningStore(tmp_path / "learning"))

    class EvidenceScreenLearner:
        provider = "evidence-screen-fixture"
        session_id = "evidence-screen-session"

        def answer(self, *, task: str, material: str = "", revision: str = "") -> dict[str, Any]:
            return {"responses": [{"case": "capability_discovery", "answer": {"capabilities": [
                {"key": "executable", "capability": "Implement the behavior", "kind": "procedure", "procedure": ["Implement", "run", "test"], "proposed_practice": "Implement a held-out example", "proposed_evidence": "Compile and run automated tests; the expected assertions pass."},
                {"key": "descriptive", "capability": "Explain the behavior", "kind": "procedure", "procedure": ["Explain"], "proposed_practice": "Explain a fresh example", "proposed_evidence": "The explanation is correct."},
            ]}}]}

    result = discover_capabilities(
        engine=engine,
        learner=EvidenceScreenLearner(),
        source_title="Evidence screening",
        source_text="A lesson about implementing and explaining behavior.",
        educational_objective="Improve software engineering",
    )

    candidates = {proposal["key"]: proposal for proposal in result["proposals"]}
    assert candidates["executable"]["behavioral_evidence_candidate"] is True
    assert candidates["descriptive"]["behavioral_evidence_candidate"] is False


def test_discovery_marks_procedureless_behavioral_proposals_actionable(tmp_path: Path) -> None:
    engine = LearningEngine(LearningStore(tmp_path / "learning"))

    class ProcedurelessLearner:
        provider = "fixture-provider"
        session_id = "procedureless-session"

        def answer(self, *, task: str, material: str = "", revision: str = "") -> dict:
            return {"responses": [{"case": "capability_discovery", "answer": {"key": "active-retrieval", "capability": "Use active retrieval to test understanding", "kind": "declarative", "proposed_practice": "Retrieve answers from memory on a fresh topic", "proposed_evidence": "A held-out explanation contains the required steps"}}]}

    result = discover_capabilities(
        engine=engine,
        learner=ProcedurelessLearner(),
        source_title="Learning lesson",
        source_text="Retrieval practice improves learning.",
        educational_objective="Improve learning",
    )

    assert result["actionable_proposal_keys"] == ["active-retrieval"]
    assert result["proposals"][0]["procedure"] == ""


def test_treatment_planning_rejects_unresolved_practice_and_evidence_placeholders(tmp_path: Path) -> None:
    engine = LearningEngine(LearningStore(tmp_path / "learning"))

    class PlaceholderLearner:
        provider = "placeholder-fixture"
        session_id = "placeholder-session"

        def answer(self, *, task: str, material: str = "", revision: str = "") -> dict:
            return {"responses": [{"case": "capability_discovery", "answer": {
                "key": "placeholder-procedure",
                "capability": "Perform a bounded diagnostic procedure",
                "kind": "procedure",
                "proposed_practice": "Diagnose [specific scenario] and record the result",
                "proposed_evidence": "Confirm [specific mechanism] was identified",
            }}]}

    result = discover_capabilities(
        engine=engine,
        learner=PlaceholderLearner(),
        source_title="Small lesson",
        source_text="A lesson about diagnosis.",
        educational_objective="Improve diagnosis",
    )

    assert result["actionable_proposal_keys"] == ["placeholder-procedure"]
    assert build_learning_treatments(result) == []


def test_discovery_normalizes_a_single_structured_capability_answer(tmp_path: Path) -> None:
    engine = LearningEngine(LearningStore(tmp_path / "learning"))
    result = discover_capabilities(
        engine=engine,
        learner=SingleCapabilityLearner(),
        source_title="Small lesson",
        source_text="A lesson about diagnosis.",
        educational_objective="Improve diagnosis",
    )

    assert [proposal["key"] for proposal in result["proposals"]] == ["single-capability"]
    assert result["actionable_proposal_keys"] == ["single-capability"]


def test_discovery_normalizes_multiple_capability_discovery_responses(tmp_path: Path) -> None:
    engine = LearningEngine(LearningStore(tmp_path / "learning"))
    result = discover_capabilities(
        engine=engine,
        learner=MultiResponseCapabilityLearner(),
        source_title="Small lesson",
        source_text="A lesson about inspection and verification.",
        educational_objective="Improve diagnosis",
    )

    assert [proposal["key"] for proposal in result["proposals"]] == ["one", "two"]


def test_preserved_experiment_selects_procedural_candidates_for_behavioral_acquisition() -> None:
    proposals = [
        {"key": "declarative", "actionable": True, "kind": "declarative", "procedure": ""},
        {"key": "workflow", "actionable": True, "kind": "workflow", "procedure": "1. Schedule", "behavioral_evidence_candidate": True},
        {"key": "procedure", "actionable": True, "kind": "procedure", "procedure": "1. Retrieve", "behavioral_evidence_candidate": True},
        {"key": "context", "actionable": False, "kind": "context", "procedure": ""},
    ]

    assert select_experiment_proposals(proposals) == ["workflow", "procedure"]


def test_select_learning_treatments_prefers_behavioral_candidates_without_target_labels() -> None:
    discovery = {
        "schema": "rex-learning-capability-discovery-v1",
        "proposals": [
            {"key": "context", "actionable": False, "kind": "context", "procedure": ""},
            {"key": "declarative", "actionable": True, "kind": "declarative", "procedure": ""},
            {"id": "proposal-weak", "key": "weak", "actionable": True, "behavioral_evidence_candidate": False, "kind": "procedure", "procedure": "1. Explain", "curriculum_id": "curriculum", "source_id": "source", "source_refs": ["unit#weak"], "source_hash": "hash", "capability": "Explain study", "relationship": "new", "proposed_practice": "Explain a concept", "proposed_evidence": "The explanation is correct"},
            {"id": "proposal-workflow", "key": "workflow", "actionable": True, "behavioral_evidence_candidate": True, "kind": "workflow", "procedure": "1. Schedule", "curriculum_id": "curriculum", "source_id": "source", "source_refs": ["unit#workflow"], "source_hash": "hash", "capability": "Schedule study", "relationship": "new", "proposed_practice": "Schedule a study session", "proposed_evidence": "The session is scheduled"},
            {"id": "proposal-procedure", "key": "procedure", "actionable": True, "behavioral_evidence_candidate": True, "kind": "procedure", "procedure": "1. Retrieve", "curriculum_id": "curriculum", "source_id": "source", "source_refs": ["unit#procedure"], "source_hash": "hash", "capability": "Retrieve information", "relationship": "new", "proposed_practice": "Retrieve a new concept", "proposed_evidence": "The concept is retrieved"},
        ],
    }

    selected = select_learning_treatments(discovery, limit=2)

    assert [item["intent_key"] for item in selected] == ["workflow", "procedure"]


def test_build_learning_treatments_excludes_rejected_behavioral_candidates() -> None:
    discovery = {
        "schema": "rex-learning-capability-discovery-v1",
        "proposals": [
            {
                "id": "proposal-weak",
                "key": "weak",
                "actionable": True,
                "behavioral_evidence_candidate": False,
                "kind": "procedure",
                "procedure": "1. Explain",
                "curriculum_id": "curriculum",
                "source_id": "source",
                "source_refs": ["unit#weak"],
                "source_hash": "hash",
                "capability": "Explain study",
                "relationship": "new",
                "existing_skill_ids": [],
                "proposed_practice": "Explain a concept clearly",
                "proposed_evidence": "The explanation is correct",
            },
            {
                "id": "proposal-strong",
                "key": "strong",
                "actionable": True,
                "behavioral_evidence_candidate": True,
                "kind": "procedure",
                "procedure": "1. Implement\n2. Test",
                "curriculum_id": "curriculum",
                "source_id": "source",
                "source_refs": ["unit#strong"],
                "source_hash": "hash",
                "capability": "Implement study",
                "relationship": "new",
                "existing_skill_ids": [],
                "proposed_practice": "Implement a held-out example",
                "proposed_evidence": "Run tests and verify expected output",
            },
        ],
    }

    built = build_learning_treatments(discovery)

    assert [item["intent_key"] for item in built] == ["strong"]


def test_competence_test_requests_preserve_untrusted_proposals_and_require_independent_design() -> None:
    discovery = {
        "schema": "rex-learning-capability-discovery-v1",
        "proposals": [{
            "id": "proposal-probe",
            "key": "diagnose-failure",
            "curriculum_id": "curriculum-probe",
            "curriculum_intent_id": "intent-probe",
            "source_id": "source-probe",
            "source_refs": ["source-probe"],
            "source_hash": "hash-probe",
            "capability": "Diagnose a failure by testing hypotheses",
            "kind": "procedure",
            "relationship": "new",
            "procedure": "1. Reproduce\n2. Isolate\n3. Test",
            "proposed_practice": "Diagnose a held-out failure",
            "proposed_evidence": "The root cause is identified and the repair passes tests.",
            "actionable": True,
            "behavioral_evidence_candidate": True,
            "existing_skill_ids": [],
        }],
    }

    requests = build_competence_test_requests(discovery)

    assert requests == [{
        "schema": "rex-learning-test-design-request-v1",
        "discovery_provenance": {},
        "proposal_id": "proposal-probe",
        "intent_key": "diagnose-failure",
        "curriculum_id": "curriculum-probe",
        "curriculum_intent_id": "intent-probe",
        "source_id": "source-probe",
        "source_refs": ["source-probe"],
        "source_hash": "hash-probe",
        "target_capability": "Diagnose a failure by testing hypotheses",
        "skill_kind": "procedure",
        "procedure": "1. Reproduce\n2. Isolate\n3. Test",
        "untrusted_proposed_practice": "Diagnose a held-out failure",
        "untrusted_proposed_evidence": "The root cause is identified and the repair passes tests.",
        "status": "awaiting_independent_design",
        "evaluator_requirements": {
            "source_free": True,
            "novel_transfer": True,
            "independent_evaluator": True,
            "negative_applicability": True,
        },
    }]


def test_independent_test_design_binding_rejects_learner_scaffolding_and_preserves_lineage(tmp_path: Path) -> None:
    request = {
        "schema": "rex-learning-test-design-request-v1",
        "proposal_id": "proposal-probe",
        "intent_key": "diagnose-failure",
        "curriculum_id": "curriculum-probe",
        "curriculum_intent_id": "intent-probe",
        "source_id": "source-probe",
        "source_refs": ["source-probe#diagnosis"],
        "source_hash": "hash-probe",
        "target_capability": "Diagnose a failure by testing hypotheses",
        "skill_kind": "procedure",
        "procedure": "1. Reproduce\n2. Isolate\n3. Test",
        "untrusted_proposed_practice": "Diagnose a held-out failure",
        "untrusted_proposed_evidence": "The root cause is identified and the repair passes tests.",
        "status": "awaiting_independent_design",
        "evaluator_requirements": {"source_free": True, "novel_transfer": True, "independent_evaluator": True, "negative_applicability": True},
    }
    design = {
        "proposal_id": "proposal-probe",
        "curriculum_id": "curriculum-probe",
        "curriculum_intent_id": "intent-probe",
        "source_id": "source-probe",
        "source_refs": ["source-probe#diagnosis"],
        "source_hash": "hash-probe",
        "target_capability": "Diagnose a failure by testing hypotheses",
        "skill_kind": "procedure",
        "cases": ["baseline", "practice", "posttest", "retest", "negative", "fresh", "control"],
        "phase_cases": {"baseline": "baseline", "practice": "practice", "posttest": "posttest", "retest": "retest", "negative": "negative", "fresh": "fresh", "control": "control"},
        "phase_tasks": {"baseline": "baseline task", "practice": "practice task", "posttest": "posttest task", "retest": "retest task", "negative": "negative task", "fresh": "fresh task", "control": "control task"},
        "evaluation_contracts": {phase: {"phase": phase, "required_fields": ["artifact"]} for phase in ("baseline", "practice", "posttest", "retest", "negative", "fresh", "control")},
        "evaluator_id": "grader.independent-diagnosis",
        "evaluator_type": "executable_grader",
        "success_threshold": 1.0,
        "source_free": True,
        "novel_transfer": True,
        "evaluator_independence": "independent",
        "contamination_status": "clean",
        "negative_applicability": True,
    }

    bound = bind_independent_competence_test_design(
        request=request,
        design=design,
        designer_provenance={"provider": "independent-test-designer", "session_id": "designer-session", "role": "test_designer"},
    )
    assert bound["evaluator_type"] == "executable_grader"

    with pytest.raises(LearningError, match="schema-only contract evaluator"):
        bind_independent_competence_test_design(
            request=request,
            design={**design, "evaluator_id": "contract-only", "evaluator_type": "contract_artifact_evaluator"},
            designer_provenance={"provider": "independent-test-designer", "session_id": "contract-session", "role": "test_designer"},
        )
    grader = ExecutableGrader(
        evaluator_id="grader.independent-diagnosis",
        answer_key={case: "ok" for case in design["cases"]},
        provider="independent-evaluator",
        session_id="evaluator-session",
    )
    engine = LearningEngine(LearningStore(tmp_path / "learning"), trusted_evaluators={grader.evaluator_id: grader})
    curriculum = engine.create_curriculum("probe", "diagnose failures")
    skill = engine.create_skill_hypothesis(curriculum["id"], {"key": "diagnose-failure", "claim": "Diagnose failures by testing hypotheses"})

    assert bound["schema"] == "rex-learning-independent-test-design-v1"
    assert bound["proposal_id"] == request["proposal_id"]
    assert bound["source_hash"] == request["source_hash"]
    assert bound["cases"] == design["cases"]
    assert bound["designer_provenance"]["role"] == "test_designer"
    assert "answer_key" not in bound
    assert bound["status"] == "awaiting_trusted_evaluator_binding"
    materialized = define_test_from_independent_design(
        engine=engine,
        curriculum_id=curriculum["id"],
        skill_id=skill["id"],
        design=bound,
        phase="posttest",
    )
    assert materialized["evaluation_contract"] == design["evaluation_contracts"]["posttest"]

    contradictory = {
        **design,
        "phase_tasks": {
            phase: "Return a JSON object containing a responses list. Do not return a JSON envelope."
            for phase in design["phase_tasks"]
        },
    }
    with pytest.raises(LearningError, match="contradicts the learner responses envelope"):
        bind_independent_competence_test_design(
            request=request,
            design=contradictory,
            designer_provenance={"provider": "independent-test-designer", "session_id": "designer-session", "role": "test_designer"},
        )

    with pytest.raises(LearningError, match="learner-visible fixture or starter"):
        bind_independent_competence_test_design(
            request=request,
            design={**design, "self_contained_tasks": True},
            designer_provenance={"provider": "independent-test-designer", "session_id": "designer-session", "role": "test_designer"},
        )

    with pytest.raises(LearningError, match="complete evaluation contracts"):
        bind_independent_competence_test_design(
            request=request,
            design={
                **design,
                "self_contained_tasks": True,
                "phase_tasks": {
                    phase: "Learner-visible fixture: verify the requested behavior."
                    for phase in ("baseline", "practice", "posttest", "retest", "negative", "fresh", "control")
                },
            },
            designer_provenance={"provider": "independent-test-designer", "session_id": "designer-session", "role": "test_designer"},
        )

    with pytest.raises(LearningError, match="complete phase contracts"):
        bind_independent_competence_test_design(
            request=request,
            design={key: value for key, value in design.items() if key not in {"phase_cases", "phase_tasks"}},
            designer_provenance={"provider": "independent-test-designer", "session_id": "designer-session", "role": "test_designer"},
        )

    with pytest.raises(LearningError, match="learner-authored"):
        bind_independent_competence_test_design(
            request=request,
            design={**design, "answer_key": {"posttest": "known-answer"}},
            designer_provenance={"provider": "independent-test-designer", "session_id": "designer-session", "role": "test_designer"},
        )


def test_independent_test_design_rejects_self_contained_cross_capability_phase_drift() -> None:
    request = {
        "schema": "rex-learning-test-design-request-v1",
        "proposal_id": "proposal-coherence",
        "intent_key": "mocking-diagnosis",
        "curriculum_id": "curriculum-coherence",
        "curriculum_intent_id": "intent-coherence",
        "source_id": "source-coherence",
        "source_refs": ["source-coherence#testing"],
        "source_hash": "hash-coherence",
        "target_capability": "Diagnose bugs where mocking hides integration failures in a real dependency",
        "skill_kind": "declarative",
        "procedure": "Compare isolated mocked behavior with the real integration dependency and identify the hidden bug.",
        "status": "awaiting_independent_design",
    }
    phases = ("baseline", "practice", "posttest", "retest", "negative", "fresh", "control")
    design = {
        **{field: request[field] for field in ("proposal_id", "curriculum_id", "curriculum_intent_id", "source_id", "source_hash", "target_capability", "skill_kind")},
        "source_refs": list(request["source_refs"]),
        "cases": list(phases),
        "phase_cases": {phase: phase for phase in phases},
        "phase_tasks": {
            phase: "Learner-visible fixture: diagnose timezone normalization with a real zone dependency. Return responses with the matching case_id."
            for phase in phases
        },
        "evaluation_contracts": {phase: "Require responses and a concrete observable diagnosis." for phase in phases},
        "evaluator_id": "grader.coherence",
        "evaluator_type": "semantic_checklist_grader",
        "success_threshold": 1.0,
        "source_free": True,
        "novel_transfer": True,
        "evaluator_independence": "independent",
        "contamination_status": "clean",
        "negative_applicability": True,
        "self_contained_tasks": True,
    }
    with pytest.raises(LearningError, match="not coherent"):
        bind_independent_competence_test_design(
            request=request,
            design=design,
            designer_provenance={"provider": "designer", "session_id": "coherence-session", "role": "test_designer"},
        )


def test_fresh_phase_coherence_accepts_explicit_unseen_transfer_with_one_anchor() -> None:
    relevance = {"defect", "failure", "software", "quality"}
    phase_text = (
        "Apply the same defect-versus-failure contract to this unseen domain. "
        "Analyze every case and tie failure claims to observed behavior."
    )
    assert _coherent_independent_phase(
        phase="fresh", relevance=relevance, phase_text=phase_text
    )


def test_fresh_phase_coherence_rejects_single_anchor_without_transfer_framing() -> None:
    relevance = {"defect", "failure", "software", "quality"}
    phase_text = "Learner-visible fixture: diagnose timezone normalization with a real zone dependency."
    assert not _coherent_independent_phase(
        phase="fresh", relevance=relevance, phase_text=phase_text
    )


def test_independent_test_design_rejects_boilerplate_overlap_with_unrelated_fixture_domain() -> None:
    request = {
        "schema": "rex-learning-test-design-request-v1",
        "proposal_id": "proposal-boilerplate-drift",
        "intent_key": "mocking-diagnosis",
        "curriculum_id": "curriculum-boilerplate-drift",
        "curriculum_intent_id": "intent-boilerplate-drift",
        "source_id": "source-boilerplate-drift",
        "source_refs": ["source-boilerplate-drift#testing"],
        "source_hash": "hash-boilerplate-drift",
        "target_capability": "Diagnose bugs where mocking hides integration failures in a real dependency",
        "skill_kind": "declarative",
        "procedure": "Compare isolated mocked behavior with the real integration dependency and identify the hidden bug.",
        "status": "awaiting_independent_design",
    }
    phases = ("baseline", "practice", "posttest", "retest", "negative", "fresh", "control")
    design = {
        **{field: request[field] for field in ("proposal_id", "curriculum_id", "curriculum_intent_id", "source_id", "source_hash", "target_capability", "skill_kind")},
        "source_refs": list(request["source_refs"]),
        "cases": list(phases),
        "phase_cases": {phase: phase for phase in phases},
        "phase_tasks": {
            phase: "Learner-visible fixture: diagnose this unit test and integration behavior for an invoice rounding scenario. Return responses with the matching case_id."
            for phase in phases
        },
        "evaluation_contracts": {phase: "Require responses and a concrete observable diagnosis." for phase in phases},
        "evaluator_id": "grader.boilerplate-drift",
        "evaluator_type": "semantic_checklist_grader",
        "success_threshold": 1.0,
        "source_free": True,
        "novel_transfer": True,
        "evaluator_independence": "independent",
        "contamination_status": "clean",
        "negative_applicability": True,
        "self_contained_tasks": True,
    }
    with pytest.raises(LearningError, match="not coherent"):
        bind_independent_competence_test_design(
            request=request,
            design=design,
            designer_provenance={"provider": "designer", "session_id": "boilerplate-session", "role": "test_designer"},
        )


def test_independent_test_design_rejects_explicit_fixture_capability_drift() -> None:
    request = {
        "schema": "rex-learning-test-design-request-v1",
        "proposal_id": "proposal-explicit-capability-drift",
        "intent_key": "solid-design",
        "curriculum_id": "curriculum-explicit-capability-drift",
        "curriculum_intent_id": "intent-explicit-capability-drift",
        "source_id": "source-explicit-capability-drift",
        "source_refs": ["source-explicit-capability-drift#design"],
        "source_hash": "hash-explicit-capability-drift",
        "target_capability": "Apply SOLID design principles to improve maintainability and extensibility.",
        "skill_kind": "procedure",
        "procedure": "Evaluate class design for coupling, maintainability, and extensibility using SOLID principles.",
        "status": "awaiting_independent_design",
    }
    phases = ("baseline", "practice", "posttest", "retest", "negative", "fresh", "control")
    drifted_task = (
        'LEARNER-VISIBLE FIXTURE: {"capability":"branching-strategy-selection",'
        '"projects":[{"id":"P1","constraints":"maintainability, coupling, and extensibility"}]}. '
        "Choose a branching strategy and justify the decision using maintainability, coupling, and extensibility. "
        "The answer object must contain decision and rationale. OUTPUT JSON SHAPE: responses."
    )
    design = {
        **{field: request[field] for field in ("proposal_id", "curriculum_id", "curriculum_intent_id", "source_id", "source_hash", "target_capability", "skill_kind")},
        "source_refs": list(request["source_refs"]),
        "cases": list(phases),
        "phase_cases": {phase: phase for phase in phases},
        "phase_tasks": {phase: drifted_task for phase in phases},
        "evaluation_contracts": {phase: "Require a response artifact with decision and rationale." for phase in phases},
        "evaluator_id": "grader.explicit-capability-drift",
        "evaluator_type": "semantic_checklist_grader",
        "success_threshold": 1.0,
        "source_free": True,
        "novel_transfer": True,
        "evaluator_independence": "independent",
        "contamination_status": "clean",
        "negative_applicability": True,
        "self_contained_tasks": True,
    }
    with pytest.raises(LearningError, match="not coherent"):
        bind_independent_competence_test_design(
            request=request,
            design=design,
            designer_provenance={"provider": "designer", "session_id": "explicit-drift-session", "role": "test_designer"},
        )


def test_independent_test_design_binds_only_to_registered_trusted_evaluator(tmp_path: Path) -> None:
    grader = ExecutableArtifactGrader(
        evaluator_id="grader.bound-artifact",
        required_fields={"transfer": ("result",)},
        validators={"transfer": {"result": lambda value: value == "verified"}},
        provider="independent-evaluator",
        session_id="evaluator-session",
    )
    engine = LearningEngine(LearningStore(tmp_path / "learning"), trusted_evaluators={grader.evaluator_id: grader})
    curriculum = engine.create_curriculum("Binding probe", "Practice a workflow")
    skill = engine.create_skill_hypothesis(curriculum["id"], {"key": "workflow.probe", "claim": "Apply the workflow"})
    design = {
        "schema": "rex-learning-independent-test-design-v1",
        "status": "awaiting_trusted_evaluator_binding",
        "cases": ["transfer"],
        "proposal_id": "proposal-bound",
        "intent_key": "workflow.probe",
        "curriculum_id": curriculum["id"],
        "curriculum_intent_id": "intent-bound",
        "source_id": "source-bound",
        "source_refs": ["source-bound#workflow"],
        "source_hash": "source-hash-bound",
        "target_capability": "Apply the workflow",
        "skill_kind": "procedure",
        "evaluator_id": grader.evaluator_id,
        "evaluator_type": grader.evaluator_type,
        "success_threshold": 1.0,
        "evaluator_independence": "independent",
        "contamination_status": "clean",
        "source_free": True,
        "novel_transfer": True,
        "negative_applicability": True,
        "evaluation_contracts": {"posttest": {"required_fields": ["decisive_constraints", "nearest_alternative_and_reason", "safeguard"]}},
        "discovery_provenance": {"provider": "discovery", "session_id": "discovery-session"},
        "designer_provenance": {"provider": "designer", "session_id": "designer-session", "role": "test_designer"},
    }

    test = define_test_from_independent_design(
        engine=engine,
        curriculum_id=curriculum["id"],
        skill_id=skill["id"],
        design=design,
        phase="posttest",
    )

    assert test["evaluator_id"] == grader.evaluator_id
    assert test["evaluator_type"] == grader.evaluator_type
    assert test["cases"] == ["transfer"]
    assert test["evaluation_contract"] == design["evaluation_contracts"]["posttest"]
    assert test["expected_case_results"] == []
    assert "answer_key" not in test

    with pytest.raises(LearningError, match="registered trusted evaluator"):
        define_test_from_independent_design(
            engine=engine,
            curriculum_id=curriculum["id"],
            skill_id=skill["id"],
            design={**design, "evaluator_id": "unregistered"},
            phase="retest",
        )

    with pytest.raises(LearningError, match="frozen test"):
        define_test_from_independent_design(
            engine=engine,
            curriculum_id=curriculum["id"],
            skill_id=skill["id"],
            design={**design, "source_hash": "tampered-source-hash"},
            phase="posttest",
        )

    with pytest.raises(LearningError, match="frozen test"):
        define_test_from_independent_design(
            engine=engine,
            curriculum_id=curriculum["id"],
            skill_id=skill["id"],
            design={**design, "evaluation_contracts": {"posttest": {"required_fields": ["rationale"]}}},
            phase="posttest",
        )


def test_define_test_from_independent_design_preserves_string_contract(tmp_path: Path) -> None:
    grader = ExecutableArtifactGrader(
        evaluator_id="grader.string-contract",
        required_fields={"posttest": ("artifact",)},
        validators={"posttest": {"artifact": lambda value: value == "verified"}},
        provider="independent-evaluator",
        session_id="string-contract-session",
    )
    engine = LearningEngine(LearningStore(tmp_path / "learning"), trusted_evaluators={grader.evaluator_id: grader})
    curriculum = engine.create_curriculum("String contract", "Preserve contract text")
    skill = engine.create_skill_hypothesis(curriculum["id"], {"key": "contract.probe", "claim": "Preserve contract text"})
    design = {
        "schema": "rex-learning-independent-test-design-v1",
        "status": "awaiting_trusted_evaluator_binding",
        "proposal_id": "proposal-string-contract",
        "intent_key": "contract.probe",
        "curriculum_id": curriculum["id"],
        "curriculum_intent_id": "intent-string-contract",
        "source_id": "source-string-contract",
        "source_refs": ["source-string-contract#contract"],
        "source_hash": "source-hash-string-contract",
        "target_capability": "Preserve contract text",
        "skill_kind": "procedure",
        "cases": ["posttest"],
        "phase_cases": {"posttest": "posttest"},
        "phase_tasks": {"posttest": "Learner-visible fixture: return the artifact."},
        "evaluation_contracts": {"posttest": "canonical contract text"},
        "evaluator_id": grader.evaluator_id,
        "evaluator_type": grader.evaluator_type,
        "success_threshold": 1.0,
        "source_free": True,
        "novel_transfer": True,
        "evaluator_independence": "independent",
        "contamination_status": "clean",
        "negative_applicability": True,
        "discovery_provenance": {"provider": "discovery", "session_id": "discovery-string-contract"},
        "designer_provenance": {"provider": "designer", "session_id": "designer-string-contract", "role": "test_designer"},
    }
    test = define_test_from_independent_design(
        engine=engine,
        curriculum_id=curriculum["id"],
        skill_id=skill["id"],
        design=design,
        phase="posttest",
    )
    assert test["evaluation_contract"] == "canonical contract text"


def test_define_test_from_independent_design_rejects_unbound_lineage(tmp_path: Path) -> None:
    grader = ExecutableArtifactGrader(
        evaluator_id="grader.lineage-bound",
        required_fields={"transfer": ("result",)},
        validators={"transfer": {"result": lambda value: value == "verified"}},
        provider="independent-evaluator",
        session_id="lineage-session",
    )
    engine = LearningEngine(LearningStore(tmp_path / "learning"), trusted_evaluators={grader.evaluator_id: grader})
    curriculum = engine.create_curriculum("Lineage probe", "Practice a workflow")
    skill = engine.create_skill_hypothesis(curriculum["id"], {"key": "lineage.workflow", "claim": "Apply the workflow"})
    design = {
        "schema": "rex-learning-independent-test-design-v1",
        "status": "awaiting_trusted_evaluator_binding",
        "cases": ["transfer"],
        "evaluator_id": grader.evaluator_id,
        "evaluator_type": grader.evaluator_type,
        "success_threshold": 1.0,
        "evaluator_independence": "independent",
        "contamination_status": "clean",
        "source_free": True,
        "novel_transfer": True,
        "negative_applicability": True,
    }

    with pytest.raises(LearningError, match="lineage"):
        define_test_from_independent_design(
            engine=engine,
            curriculum_id=curriculum["id"],
            skill_id=skill["id"],
            design=design,
            phase="posttest",
        )


def test_execute_bound_independent_test_records_trusted_evidence_without_qualification(tmp_path: Path) -> None:
    grader = ExecutableArtifactGrader(
        evaluator_id="grader.execute-bound",
        required_fields={"transfer": ("result",)},
        validators={"transfer": {"result": lambda value: value == "verified"}},
        provider="independent-evaluator",
        session_id="execute-grader-session",
    )
    engine = LearningEngine(LearningStore(tmp_path / "learning"), trusted_evaluators={grader.evaluator_id: grader})
    curriculum = engine.create_curriculum("Execution bridge", "Practice a workflow")
    skill = engine.create_skill_hypothesis(curriculum["id"], {"key": "execute.workflow", "claim": "Apply the workflow"})
    design = {
        "schema": "rex-learning-independent-test-design-v1",
        "status": "awaiting_trusted_evaluator_binding",
        "cases": ["transfer"],
        "proposal_id": "proposal-execute",
        "intent_key": "execute.workflow",
        "curriculum_id": curriculum["id"],
        "curriculum_intent_id": "intent-execute",
        "source_id": "source-execute",
        "source_refs": ["source-execute#unit"],
        "source_hash": "hash-execute",
        "target_capability": "Apply the workflow",
        "skill_kind": "procedure",
        "evaluator_id": grader.evaluator_id,
        "evaluator_type": grader.evaluator_type,
        "success_threshold": 1.0,
        "evaluator_independence": "independent",
        "contamination_status": "clean",
        "source_free": True,
        "novel_transfer": True,
        "negative_applicability": True,
        "discovery_provenance": {"provider": "discovery", "session_id": "discovery-execute"},
        "designer_provenance": {"provider": "designer", "session_id": "designer-execute", "role": "test_designer"},
    }
    test = define_test_from_independent_design(engine=engine, curriculum_id=curriculum["id"], skill_id=skill["id"], design=design, phase="posttest")

    class BoundLearner:
        provider = "learner-provider"
        session_id = "learner-execute"

        def answer(self, *, task: str, material: str = "", revision: str = "") -> dict[str, Any]:
            assert material == ""
            return {"responses": [{"case": "transfer", "answer": {"result": "verified"}}]}

    result = execute_bound_independent_test(engine=engine, test=test, learner=BoundLearner(), task="Apply the workflow to a novel case", task_id="execute-transfer")

    assert result["schema"] == "rex-learning-independent-test-execution-v1"
    assert result["attempt"]["test_id"] == test["id"]
    assert result["evaluation"]["validation_status"] == "verified"
    assert result["evaluation"]["passed"] is True
    assert "qualification" not in result
    assert "promotion" not in result

    with pytest.raises(LearningError, match="source-free"):
        execute_bound_independent_test(engine=engine, test=test, learner=BoundLearner(), task="contaminated", material="source text", task_id="contaminated")


def test_derive_independent_behavioral_evidence_rejects_mixed_execution_lineage(tmp_path: Path) -> None:
    engine = LearningEngine(LearningStore(tmp_path / "learning"))

    with pytest.raises(LearningError, match="execution records"):
        derive_independent_behavioral_evidence(
            engine=engine,
            executions={"pretest": {"schema": "rex-learning-independent-test-execution-v1"}},
        )


def test_derive_independent_behavioral_evidence_rejects_tampered_execution_envelope() -> None:
    class PersistedStore:
        def __init__(self) -> None:
            self.records = {
                "tests": {
                    "test-pretest": {"id": "test-pretest", "skill_id": "skill-1", "phase": "pretest", "source_free": True, "novel_transfer": False, "independent_design_lineage": {"proposal_id": "canonical"}, "independent_design_provenance": {"designer": {"role": "test_designer"}}},
                    "test-posttest": {"id": "test-posttest", "skill_id": "skill-1", "phase": "posttest", "source_free": True, "novel_transfer": True, "independent_design_lineage": {"proposal_id": "canonical"}, "independent_design_provenance": {"designer": {"role": "test_designer"}}},
                    "test-retest": {"id": "test-retest", "skill_id": "skill-1", "phase": "retest", "source_free": True, "novel_transfer": True, "independent_design_lineage": {"proposal_id": "canonical"}, "independent_design_provenance": {"designer": {"role": "test_designer"}}},
                    "test-negative": {"id": "test-negative", "skill_id": "skill-1", "phase": "negative", "source_free": True, "novel_transfer": True, "independent_design_lineage": {"proposal_id": "canonical"}, "independent_design_provenance": {"designer": {"role": "test_designer"}}},
                },
                "attempts": {f"attempt-{name}": {"id": f"attempt-{name}", "test_id": f"test-{name}", "skill_id": "skill-1"} for name in ("pretest", "posttest", "retest", "negative")},
                "evaluations": {f"evaluation-{name}": {"id": f"evaluation-{name}", "attempt_id": f"attempt-{name}", "skill_id": "skill-1", "passed": True, "score": 1.0, "validation_status": "verified", "evaluator_type": "executable_artifact_grader", "independence": "independent", "contamination": {"status": "clean"}} for name in ("pretest", "posttest", "retest", "negative")},
            }

        def read(self, collection: str, identifier: str) -> dict[str, Any]:
            return self.records[collection][identifier]

    class EvidenceEngine:
        def __init__(self) -> None:
            self.store = PersistedStore()

        def derive_candidate_behavioral_evidence(self, **kwargs: str) -> dict[str, bool]:
            return {"reproducible": True}

    def execution(name: str, phase: str) -> dict[str, Any]:
        persisted = EvidenceEngine().store.records["tests"][f"test-{name}"]
        return {
            "schema": "rex-learning-independent-test-execution-v1",
            "test": {**persisted, "independent_design_lineage": {"proposal_id": "tampered"}},
            "attempt": {"id": f"attempt-{name}", "test_id": f"test-{name}"},
            "evaluation": {"id": f"evaluation-{name}", "attempt_id": f"attempt-{name}"},
            "learner_provenance": {"role": "test_taker"},
        }

    executions = {name: execution(name, phase) for name, phase in (("pretest", "pretest"), ("posttest", "posttest"), ("retest", "retest"), ("negative", "negative"))}
    with pytest.raises(LearningError, match="persisted|tamper|lineage"):
        derive_independent_behavioral_evidence(engine=cast(Any, EvidenceEngine()), executions=executions)


def test_derive_independent_behavioral_evidence_rejects_mixed_design_provenance() -> None:
    class EvidenceEngine:
        def derive_candidate_behavioral_evidence(self, **kwargs: str) -> dict[str, bool]:
            return {"reproducible": True}

    def execution(name: str, phase: str, proposal_id: str = "proposal-1") -> dict[str, Any]:
        test_id = f"test-{name}"
        attempt_id = f"attempt-{name}"
        return {
            "schema": "rex-learning-independent-test-execution-v1",
            "test": {
                "id": test_id,
                "skill_id": "skill-1",
                "phase": phase,
                "source_free": True,
                "novel_transfer": name != "pretest",
                "independent_design_lineage": {"proposal_id": proposal_id},
                "independent_design_provenance": {"designer": {"role": "test_designer"}},
            },
            "attempt": {"id": attempt_id, "test_id": test_id},
            "evaluation": {"id": f"evaluation-{name}", "attempt_id": attempt_id},
            "learner_provenance": {"role": "test_taker"},
        }

    executions = {
        "pretest": execution("pretest", "pretest"),
        "posttest": execution("posttest", "posttest", proposal_id="proposal-2"),
        "retest": execution("retest", "retest"),
        "negative": execution("negative", "negative"),
    }

    with pytest.raises(LearningError, match="lineage"):
        derive_independent_behavioral_evidence(
            engine=cast(Any, EvidenceEngine()),
            executions=executions,
        )


def test_derive_independent_behavioral_evidence_delegates_only_valid_bound_records() -> None:
    calls: list[dict[str, str]] = []

    class PersistedStore:
        def __init__(self) -> None:
            self.records: dict[str, dict[str, dict[str, Any]]] = {"tests": {}, "attempts": {}, "evaluations": {}}

        def read(self, collection: str, identifier: str) -> dict[str, Any]:
            return self.records[collection][identifier]

    class EvidenceEngine:
        def __init__(self) -> None:
            self.store = PersistedStore()

        def derive_candidate_behavioral_evidence(self, **kwargs: str) -> dict[str, bool]:
            calls.append(kwargs)
            return {"reproducible": True}

    engine = EvidenceEngine()

    def execution(name: str, phase: str) -> dict[str, Any]:
        test_id = f"test-{name}"
        attempt_id = f"attempt-{name}"
        evaluation_id = f"evaluation-{name}"
        result = {
            "schema": "rex-learning-independent-test-execution-v1",
            "test": {
                "id": test_id,
                "skill_id": "skill-1",
                "phase": phase,
                "source_free": True,
                "novel_transfer": name != "pretest",
                "independent_design_lineage": {"proposal_id": "proposal-1"},
                "independent_design_provenance": {"designer": {"role": "test_designer"}},
            },
            "attempt": {"id": attempt_id, "test_id": test_id},
            "evaluation": {"id": evaluation_id, "attempt_id": attempt_id},
            "learner_provenance": {"role": "test_taker"},
        }
        engine.store.records["tests"][test_id] = result["test"]
        engine.store.records["attempts"][attempt_id] = result["attempt"]
        engine.store.records["evaluations"][evaluation_id] = result["evaluation"]
        return result

    result = derive_independent_behavioral_evidence(
        engine=cast(Any, engine),
        executions={
            "pretest": execution("pretest", "pretest"),
            "posttest": execution("posttest", "posttest"),
            "retest": execution("retest", "retest"),
            "negative": execution("negative", "negative"),
        },
    )

    assert result == {
        "schema": "rex-learning-independent-behavioral-evidence-v1",
        "skill_id": "skill-1",
        "execution_ids": {
            "pretest": "evaluation-pretest",
            "posttest": "evaluation-posttest",
            "retest": "evaluation-retest",
            "negative": "evaluation-negative",
        },
        "evidence": {"reproducible": True},
    }
    assert calls == [{
        "pretest_attempt_id": "attempt-pretest",
        "pretest_evaluation_id": "evaluation-pretest",
        "posttest_evaluation_id": "evaluation-posttest",
        "retest_evaluation_id": "evaluation-retest",
        "negative_evaluation_id": "evaluation-negative",
    }]


def test_preserved_experiment_candidate_contract_requires_structured_answer() -> None:
    suffix = request_suffix("candidate|identify the capability")
    assert "answer field must contain a JSON object" in suffix


def test_preserved_experiment_behavior_contract_requires_exact_json_artifact() -> None:
    retrieval = request_suffix("practice|case=practice|convert main points into questions", ["questions", "retrieval_result", "rephrased_insight", "application"])
    testing = request_suffix("practice|case=practice|use testing to retain the procedure", ["test_plan", "feedback_step", "spacing_schedule", "retrieval_prompt"])
    assert '"questions":["..."]' in retrieval
    assert '"retrieval_result":["..."]' in retrieval
    assert "never prose" in retrieval
    assert '"test_plan":"..."' in testing
    assert '"spacing_schedule":["..."]' in testing
    assert "never prose" in testing


def test_execute_learning_treatments_requires_explicit_qualification_dependencies(tmp_path: Path) -> None:
    engine = LearningEngine(LearningStore(tmp_path / "learning"))
    discovery = discover_capabilities(
        engine=engine,
        learner=DiscoveryLearner(),
        source_title="Small lesson",
        source_text="A lesson about reusable procedures.",
        educational_objective="Improve software engineering",
    )
    treatments = build_learning_treatments(discovery, selected_keys=["extract-functions"])

    with pytest.raises(Exception, match="qualification dependencies"):
        execute_learning_treatments(engine=engine, discovery=discovery, treatments=treatments, dependencies={})


def test_execute_learning_treatments_reuses_discovery_source_and_promotes(tmp_path: Path) -> None:
    grader = ExecutableGrader(
        evaluator_id="grader.discovery-treatment",
        answer_key={case: "not-applicable" if case == "negative" else "inspection-pass" for case in ("pre", "practice", "post", "retest", "negative", "fresh", "control")},
        provider="independent-grader",
        session_id="grader-discovery-treatment",
    )
    discovery_engine = LearningEngine(LearningStore(tmp_path / "learning"), trusted_evaluators=cast(dict[str, TrustedEvaluator], {grader.evaluator_id: grader}))
    discovery = discover_capabilities(
        engine=discovery_engine,
        learner=DiscoveryLearner(),
        source_title="Small lesson",
        source_text="A lesson about reusable procedures.",
        educational_objective="Improve software engineering",
    )
    treatments = build_learning_treatments(discovery, selected_keys=["extract-functions"])
    study_learner = AcquisitionLearner("study")

    def reviewer(candidate: dict, evidence: dict) -> dict:
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

    result = execute_learning_treatments(
        engine=discovery_engine,
        discovery=discovery,
        treatments=treatments,
        dependencies={
            "extract-functions": {
                "learner": study_learner,
                "grader": grader,
                "reviewer": reviewer,
                "fresh_learner": AcquisitionLearner("fresh"),
                "control_learner": NoTreatmentControlLearner("control"),
            }
        },
        source_text_by_intent={"extract-functions": "TARGET SUPPORTING MATERIAL"},
    )

    assert result["source_id"] == discovery["source"]["id"]
    assert result["runs"][0]["source"]["id"] == discovery["source"]["id"]
    assert result["runs"][0]["promotion"]["state"] == "demonstrated"
    assert "TARGET SUPPORTING MATERIAL" in study_learner.materials
    assert len(discovery_engine.store.list("sources")) == 1


def test_treatment_planning_converts_actionable_proposals_without_inventing_evaluation(tmp_path: Path) -> None:
    engine = LearningEngine(LearningStore(tmp_path / "learning"))
    result = discover_capabilities(
        engine=engine,
        learner=DiscoveryLearner(),
        source_title="Small lesson",
        source_text="A lesson about testing.",
        educational_objective="Improve software engineering",
    )

    treatments = build_learning_treatments(result)

    assert [item["intent_key"] for item in treatments] == ["extract-functions"]
    treatment = treatments[0]
    assert treatment["proposal_id"] == result["proposals"][0]["id"]
    assert treatment["target_capability"] == result["proposals"][0]["capability"]
    assert treatment["skill_kind"] == "procedure"
    assert treatment["practice_task"] == result["proposals"][0]["proposed_practice"]
    assert treatment["evidence_plan"] == result["proposals"][0]["proposed_evidence"]
    assert treatment["source_refs"] == result["proposals"][0]["source_refs"]
    assert treatment["missing_dependencies"] == ["grader", "reviewer", "fresh_learner", "control_learner"]
    assert treatment["status"] == "awaiting_qualification_dependencies"


def test_treatment_planning_does_not_route_refinement_as_new_acquisition(tmp_path: Path) -> None:
    engine = LearningEngine(LearningStore(tmp_path / "learning"))
    curriculum = engine.create_curriculum("Small lesson", "Improve software engineering")
    engine.create_skill_hypothesis(
        curriculum["id"],
        {
            "key": "existing-functions",
            "claim": "Recognize repeated logic and extract a function",
            "applicability": ["Recognize repeated logic and extract a function"],
        },
    )
    discovery = discover_capabilities(
        engine=engine,
        learner=DiscoveryLearner(),
        source_title="Small lesson",
        source_text="A lesson about reusable procedures.",
        educational_objective="Improve software engineering",
        curriculum_id=curriculum["id"],
    )

    treatment = build_learning_treatments(discovery, selected_keys=["extract-functions"])[0]

    assert treatment["relationship"] == "refine"
    assert treatment["treatment_mode"] == "integrate_existing"
    assert treatment["status"] == "awaiting_existing_skill_integration"


def test_execute_learning_treatments_stops_before_acquisition_for_refinement(tmp_path: Path) -> None:
    engine = LearningEngine(LearningStore(tmp_path / "learning"))
    curriculum = engine.create_curriculum("Small lesson", "Improve software engineering")
    engine.create_skill_hypothesis(
        curriculum["id"],
        {"key": "existing-functions", "claim": "Recognize repeated logic and extract a function"},
    )
    discovery = discover_capabilities(
        engine=engine,
        learner=DiscoveryLearner(),
        source_title="Small lesson",
        source_text="A lesson about reusable procedures.",
        educational_objective="Improve software engineering",
        curriculum_id=curriculum["id"],
    )
    treatment = build_learning_treatments(discovery, selected_keys=["extract-functions"])[0]

    with pytest.raises(LearningError, match="requires existing-skill integration or conflict review"):
        execute_learning_treatments(
            engine=engine,
            discovery=discovery,
            treatments=[treatment],
            dependencies={"extract-functions": {"grader": object(), "reviewer": object(), "fresh_learner": object(), "control_learner": object()}},
        )


def test_execute_learning_treatments_rejects_treatment_mode_substitution(tmp_path: Path) -> None:
    engine = LearningEngine(LearningStore(tmp_path / "learning"))
    curriculum = engine.create_curriculum("Small lesson", "Improve software engineering")
    engine.create_skill_hypothesis(
        curriculum["id"],
        {"key": "existing-functions", "claim": "Recognize repeated logic and extract a function"},
    )
    discovery = discover_capabilities(
        engine=engine,
        learner=DiscoveryLearner(),
        source_title="Small lesson",
        source_text="A lesson about reusable procedures.",
        educational_objective="Improve software engineering",
        curriculum_id=curriculum["id"],
    )
    planned = build_learning_treatments(discovery, selected_keys=["extract-functions"])[0]
    substituted = {**planned, "treatment_mode": "acquire"}

    with pytest.raises(LearningError, match="does not match discovery proposal"):
        execute_learning_treatments(
            engine=engine,
            discovery=discovery,
            treatments=[substituted],
            dependencies={},
        )


def test_plan_existing_skill_integrations_persists_refinement_without_mutating_skill(tmp_path: Path) -> None:
    engine = LearningEngine(LearningStore(tmp_path / "learning"))
    curriculum = engine.create_curriculum("Small lesson", "Improve software engineering")
    existing = engine.create_skill_hypothesis(
        curriculum["id"],
        {"key": "existing-functions", "claim": "Recognize repeated logic and extract a function"},
    )
    existing["version"] = 1
    existing["state"] = "demonstrated"
    engine.store.write("skills", existing["id"], existing)
    discovery = discover_capabilities(
        engine=engine,
        learner=DiscoveryLearner(),
        source_title="Small lesson",
        source_text="A lesson about reusable procedures.",
        educational_objective="Improve software engineering",
        curriculum_id=curriculum["id"],
    )
    treatment = build_learning_treatments(discovery, selected_keys=["extract-functions"])[0]

    plans = plan_existing_skill_integrations(engine=engine, discovery=discovery, treatments=[treatment])

    assert len(plans) == 1
    assert plans[0]["treatment_mode"] == "integrate_existing"
    assert plans[0]["existing_skill_ids"] == [existing["id"]]
    assert plans[0]["target_skill_versions"] == [{"skill_id": existing["id"], "version": 1}]
    assert plans[0]["status"] == "awaiting_behavioral_requalification"
    assert plans[0]["source_id"] == discovery["source"]["id"]
    assert engine.skill(existing["id"])["version"] == 1
    assert engine.skill(existing["id"])["state"] == "demonstrated"
    assert len(engine.store.list("skill_integration_plans")) == 1


def test_execute_learning_treatments_uses_success_budget_and_skips_only_ceiling(monkeypatch: pytest.MonkeyPatch) -> None:
    proposals = []
    treatments = []
    dependencies = {}
    for number in (1, 2, 3):
        key = f"capability-{number}"
        proposal = {
            "id": f"proposal-{number}", "key": key, "actionable": True,
            "relationship": "new", "curriculum_id": "curriculum-1", "source_id": "source-1",
            "source_hash": "hash-1", "capability": f"Capability {number}", "kind": "procedure",
            "procedure": f"Procedure {number}", "proposed_practice": f"Practice {number}",
            "proposed_evidence": f"Evidence {number}", "source_refs": ["source-1"],
        }
        proposals.append(proposal)
        treatments.append({
            "schema": "rex-learning-treatment-v1", "proposal_id": proposal["id"],
            "curriculum_id": proposal["curriculum_id"], "source_id": proposal["source_id"],
            "source_hash": proposal["source_hash"], "intent_key": key,
            "target_capability": proposal["capability"], "skill_kind": proposal["kind"],
            "relationship": "new", "treatment_mode": "acquire", "procedure": proposal["procedure"],
            "practice_task": proposal["proposed_practice"], "evidence_plan": proposal["proposed_evidence"],
            "source_refs": proposal["source_refs"], "existing_skill_ids": [],
        })
        dependencies[key] = {field: object() for field in ("learner", "grader", "reviewer", "fresh_learner", "control_learner")}
    discovery = {
        "schema": "rex-learning-capability-discovery-v1",
        "source": {"id": "source-1", "title": "Lesson", "text": "Text", "locator": "memory://lesson"},
        "curriculum": {"id": "curriculum-1"}, "proposals": proposals,
    }
    calls: list[str] = []

    def fake_learn_from_source(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs["intent_key"])
        if kwargs["intent_key"] == "capability-1":
            raise CeilingBaselineError("no improvement", details={"pre_score": 1.0, "post_score": 1.0})
        return {"run_id": f"run-{kwargs['intent_key']}"}

    import rex_learning.acquisition as acquisition
    monkeypatch.setattr(acquisition, "learn_from_source", fake_learn_from_source)
    result = execute_learning_treatments(
        engine=cast(Any, object()), discovery=discovery, treatments=treatments,
        dependencies=dependencies, source_text="Text", successful_acquisition_limit=1,
        continue_on_ceiling=True,
    )

    assert calls == ["capability-1", "capability-2"]
    assert result["successful_acquisitions"] == 1
    assert result["skipped"] == [{
        "proposal_id": "proposal-1", "intent_key": "capability-1",
        "disposition": "ceiling_baseline", "diagnostic": {"pre_score": 1.0, "post_score": 1.0},
    }]


def test_execute_learning_treatments_does_not_swallow_non_ceiling_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    proposal = {
        "id": "proposal-1", "key": "capability-1", "actionable": True, "relationship": "new",
        "curriculum_id": "curriculum-1", "source_id": "source-1", "source_hash": "hash-1",
        "capability": "Capability", "kind": "procedure", "procedure": "Procedure",
        "proposed_practice": "Practice", "proposed_evidence": "Evidence", "source_refs": ["source-1"],
    }
    treatment = {
        "schema": "rex-learning-treatment-v1", "proposal_id": "proposal-1", "curriculum_id": "curriculum-1",
        "source_id": "source-1", "source_hash": "hash-1", "intent_key": "capability-1",
        "target_capability": "Capability", "skill_kind": "procedure", "relationship": "new",
        "treatment_mode": "acquire", "procedure": "Procedure", "practice_task": "Practice",
        "evidence_plan": "Evidence", "source_refs": ["source-1"], "existing_skill_ids": [],
    }
    import rex_learning.acquisition as acquisition
    monkeypatch.setattr(acquisition, "learn_from_source", lambda **kwargs: (_ for _ in ()).throw(LearningError("practice failure")))
    with pytest.raises(LearningError, match="practice failure"):
        execute_learning_treatments(
            engine=cast(Any, object()),
            discovery={"schema": "rex-learning-capability-discovery-v1", "source": {"id": "source-1", "title": "Lesson", "text": "Text"}, "curriculum": {"id": "curriculum-1"}, "proposals": [proposal]},
            treatments=[treatment], dependencies={"capability-1": {field: object() for field in ("learner", "grader", "reviewer", "fresh_learner", "control_learner")}},
            source_text="Text", successful_acquisition_limit=1, continue_on_ceiling=True,
        )


def test_preserve_conflict_plan_records_both_claims_without_overwriting_skill(tmp_path: Path) -> None:
    engine = LearningEngine(LearningStore(tmp_path / "learning"))
    curriculum = engine.create_curriculum("Conflict lesson", "Improve implementation planning")
    existing = engine.create_skill_hypothesis(
        curriculum["id"], {"key": "planning", "claim": "Establish explicit boundaries early"},
    )
    existing["version"], existing["state"] = 2, "demonstrated"
    engine.store.write("skills", existing["id"], existing)
    discovery = discover_capabilities(
        engine=engine,
        learner=type("ConflictLearner", (), {
            "provider": "conflict-discovery-provider",
            "session_id": "conflict-discovery-session",
            "answer": lambda self, **kwargs: {"responses": [{"case": "capability_discovery", "answer": {"capabilities": [{
                "key": "abstraction-boundaries", "capability": "Establish explicit boundaries early",
                "kind": "procedure", "procedure": ["Establish boundaries"],
                "proposed_practice": "Choose boundaries for a fresh design",
                "proposed_evidence": "The boundaries are justified",
                "contradicts": [existing["id"]],
            }]}}]},
        })(),
        source_title="Conflict lesson", source_text="Establish explicit boundaries early.",
        educational_objective="Improve implementation planning", curriculum_id=curriculum["id"],
    )
    treatment = build_learning_treatments(discovery, selected_keys=["abstraction-boundaries"])[0]
    plan = plan_existing_skill_integrations(engine=engine, discovery=discovery, treatments=[treatment])[0]

    review = preserve_conflict_plan(engine=engine, plan_id=plan["id"])

    assert review["status"] == "unresolved"
    assert review["source_claim"] == plan["target_capability"]
    assert review["existing_skill_snapshots"] == [{"skill_id": existing["id"], "version": 2, "claim": "Establish explicit boundaries early"}]
    assert engine.skill(existing["id"])["claim"] == "Establish explicit boundaries early"
    assert engine.skill(existing["id"])["version"] == 2
    stored_plan = engine.store.read("skill_integration_plans", plan["id"])
    assert stored_plan["status"] == "conflict_preserved"
    assert preserve_conflict_plan(engine=engine, plan_id=plan["id"]) == review


def test_integrate_existing_skill_plan_creates_unqualified_version_from_trusted_evidence(tmp_path: Path) -> None:
    grader = ExecutableGrader(
        "grader.integration.v1", {"requalify": "updated"}, "independent-grader", "integration-session"
    )
    engine = LearningEngine(LearningStore(tmp_path / "learning"), trusted_evaluators={grader.evaluator_id: grader})
    curriculum = engine.create_curriculum("Small lesson", "Improve software engineering")
    existing = engine.create_skill_hypothesis(
        curriculum["id"],
        {"key": "existing-functions", "claim": "Recognize repeated logic and extract a function"},
    )
    existing["version"] = 1
    existing["state"] = "demonstrated"
    engine.store.write("skills", existing["id"], existing)
    test = engine.define_test(curriculum["id"], existing["id"], {
        "phase": "posttest", "cases": ["requalify"], "success_threshold": 1.0,
        "evaluator_type": "executable_grader", "evaluator_id": grader.evaluator_id,
        "evaluator_independence": "independent", "contamination_status": "clean",
        "source_free": True, "novel_transfer": True,
    })
    attempt = engine.record_attempt(test["id"], {"responses": [{"case": "requalify", "answer": "updated"}]})
    evaluation = engine.evaluate_trusted_attempt(attempt["id"])
    discovery = discover_capabilities(
        engine=engine, learner=DiscoveryLearner(), source_title="Small lesson",
        source_text="A lesson about reusable procedures.", educational_objective="Improve software engineering",
        curriculum_id=curriculum["id"],
    )
    treatment = build_learning_treatments(discovery, selected_keys=["extract-functions"])[0]
    plan = plan_existing_skill_integrations(engine=engine, discovery=discovery, treatments=[treatment])[0]

    updated = integrate_existing_skill_plan(
        engine=engine, plan_id=plan["id"],
        revision={"operational_procedure": "Identify repetition, extract a tested function, and verify behavior."},
        evidence_ids=[evaluation["id"]],
    )

    assert updated["id"] == existing["id"]
    assert updated["version"] == 2
    assert updated["state"] == "practicing"
    assert updated["qualification_status"] == "candidate"
    assert updated["integration"]["plan_id"] == plan["id"]
    assert updated["integration"]["source_id"] == plan["source_id"]
    assert engine.store.read("skill_versions", f"{existing['id']}.v1")["snapshot"]["version"] == 1
    assert engine.store.read("skill_integration_plans", plan["id"])["status"] == "awaiting_behavioral_requalification"
    assert integrate_existing_skill_plan(
        engine=engine, plan_id=plan["id"], revision={"claim": "ignored on idempotent replay"}, evidence_ids=[evaluation["id"]]
    ) == updated


def test_requalify_existing_skill_plan_promotes_only_after_complete_behavioral_gate(tmp_path: Path) -> None:
    grader = ExecutableGrader(
        "grader.requalification.v1",
        {"pretest": "right", "posttest": "right", "retest": "right", "negative": "not-applicable"},
        "independent-requalification-grader",
        "requalification-session",
    )
    engine = LearningEngine(LearningStore(tmp_path / "learning"), trusted_evaluators={grader.evaluator_id: grader})
    curriculum = engine.create_curriculum("Small lesson", "Improve software engineering")
    existing = engine.create_skill_hypothesis(
        curriculum["id"], {"key": "existing-functions", "claim": "Recognize repeated logic and extract a function"}
    )
    existing["version"], existing["state"] = 1, "demonstrated"
    engine.store.write("skills", existing["id"], existing)

    def evidence(phase: str, answer: str, *, novel_transfer: bool = False) -> tuple[dict, dict]:
        test = engine.define_test(curriculum["id"], existing["id"], {
            "phase": phase, "cases": [phase], "success_threshold": 1.0,
            "evaluator_type": "executable_grader", "evaluator_id": grader.evaluator_id,
            "evaluator_independence": "independent", "contamination_status": "clean",
            "source_free": True, "novel_transfer": novel_transfer,
        })
        attempt = engine.record_attempt(test["id"], {"responses": [{"case": phase, "answer": answer}]})
        return attempt, engine.evaluate_trusted_attempt(attempt["id"])

    pre_attempt, pre_eval = evidence("pretest", "wrong")
    _, post_eval = evidence("posttest", "right", novel_transfer=True)
    _, retest_eval = evidence("retest", "right", novel_transfer=True)
    _, negative_eval = evidence("negative", "not-applicable")
    discovery = discover_capabilities(
        engine=engine, learner=DiscoveryLearner(), source_title="Small lesson",
        source_text="A lesson about reusable procedures.", educational_objective="Improve software engineering",
        curriculum_id=curriculum["id"],
    )
    treatment = build_learning_treatments(discovery, selected_keys=["extract-functions"])[0]
    plan = plan_existing_skill_integrations(engine=engine, discovery=discovery, treatments=[treatment])[0]
    integrated = integrate_existing_skill_plan(
        engine=engine, plan_id=plan["id"], revision={"operational_procedure": "Extract repeated logic into a tested function."},
        evidence_ids=[post_eval["id"]],
    )

    qualified = requalify_existing_skill_plan(
        engine=engine, plan_id=plan["id"],
        pretest_attempt_id=pre_attempt["id"], pretest_evaluation_id=pre_eval["id"],
        posttest_evaluation_id=post_eval["id"], retest_evaluation_id=retest_eval["id"],
        negative_evaluation_id=negative_eval["id"],
    )

    assert integrated["version"] == 2
    assert qualified["id"] == existing["id"]
    assert qualified["version"] == 3
    assert qualified["state"] == "demonstrated"
    assert qualified["qualification_status"] == "behaviorally_requalified"
    assert qualified["integration"]["plan_id"] == plan["id"]
    stored_plan = engine.store.read("skill_integration_plans", plan["id"])
    assert stored_plan["status"] == "behaviorally_requalified"
    assert stored_plan["qualification_status"] == "promoted"
    assert engine.store.read("skill_versions", f"{existing['id']}.v2")["snapshot"]["version"] == 2
    assert requalify_existing_skill_plan(
        engine=engine, plan_id=plan["id"],
        pretest_attempt_id=pre_attempt["id"], pretest_evaluation_id=pre_eval["id"],
        posttest_evaluation_id=post_eval["id"], retest_evaluation_id=retest_eval["id"],
        negative_evaluation_id=negative_eval["id"],
    ) == qualified


def test_execute_learning_treatments_rejects_treatment_substitution(tmp_path: Path) -> None:
    engine = LearningEngine(LearningStore(tmp_path / "learning"))
    discovery = discover_capabilities(
        engine=engine,
        learner=DiscoveryLearner(),
        source_title="Small lesson",
        source_text="A lesson about reusable procedures.",
        educational_objective="Improve software engineering",
    )
    planned = build_learning_treatments(discovery, selected_keys=["extract-functions"])[0]
    substituted = {**planned, "target_capability": "Caller-invented capability"}

    with pytest.raises(LearningError, match="does not match discovery proposal"):
        execute_learning_treatments(
            engine=engine,
            discovery=discovery,
            treatments=[substituted],
            dependencies={},
        )


def test_discovery_from_source_units_preserves_unit_provenance(tmp_path: Path) -> None:
    engine = LearningEngine(LearningStore(tmp_path / "learning"))
    result = discover_capabilities_from_units(
        engine=engine,
        learner=DiscoveryLearner(),
        source_title="Functions lesson",
        source_units=[
            {"unit_id": "unit-1", "title": "Repetition", "text": "Find repeated logic.", "source_ref": "book.xhtml#one"},
            {"unit_id": "unit-2", "title": "Parameters", "text": "Choose useful parameters.", "source_ref": "book.xhtml#two"},
        ],
        educational_objective="Become better at programming",
    )

    assert "Find repeated logic." in result["source"]["text"]
    assert "Choose useful parameters." in result["source"]["text"]
    assert result["unit_refs"] == ["book.xhtml#one", "book.xhtml#two"]
    assert all("book.xhtml#one" in proposal["source_refs"] for proposal in result["proposals"])
    assert all("book.xhtml#two" in proposal["source_refs"] for proposal in result["proposals"])


def test_epub_ingestion_can_route_existing_source_into_capability_discovery(tmp_path: Path) -> None:
    epub = tmp_path / "lesson.epub"
    with zipfile.ZipFile(epub, "w") as book:
        book.writestr("OEBPS/chapter.xhtml", "<h1>Functions</h1><p>Repeated logic can become a reusable function.</p>")
    engine = LearningEngine(LearningStore(tmp_path / "learning"))
    curriculum = engine.create_curriculum("EPUB discovery", "Become better at programming")

    result = ingest_epub(
        engine,
        curriculum["id"],
        epub,
        learner=DiscoveryLearner(),
        educational_objective="Become better at programming",
    )

    discovery = result["capability_discovery"]
    assert discovery["source"]["id"] != result["source"]["id"]
    assert discovery["source"]["kind"] == "source_excerpt"
    assert discovery["source"]["parent_source_id"] == result["source"]["id"]
    assert all(proposal["source_id"] == discovery["source"]["id"] for proposal in discovery["proposals"])
    assert all(any(ref.startswith("OEBPS/chapter.xhtml#Functions:") for ref in proposal["source_refs"]) for proposal in discovery["proposals"])
    assert len(engine.store.list("sources")) == 2


def test_generic_epub_path_reaches_independent_design_binding_without_target_labels(tmp_path: Path) -> None:
    epub = tmp_path / "generic.epub"
    with zipfile.ZipFile(epub, "w") as book:
        book.writestr("OEBPS/lesson.xhtml", "<h1>Procedure</h1><p>Identify repeated logic and extract a tested function.</p>")
    engine = LearningEngine(LearningStore(tmp_path / "learning"))
    curriculum = engine.create_curriculum("Generic EPUB", "Find transferable procedures")

    ingestion = ingest_epub(
        engine,
        curriculum["id"],
        epub,
        learner=DiscoveryLearner(),
        educational_objective="Find transferable procedures",
    )
    discovery = ingestion["capability_discovery"]
    treatments = select_learning_treatments(discovery, limit=1)
    requests = build_competence_test_requests(discovery)
    request = next(item for item in requests if item["proposal_id"] == treatments[0]["proposal_id"])
    proposal = next(item for item in discovery["proposals"] if item["id"] == request["proposal_id"])
    design = {
        "proposal_id": request["proposal_id"],
        "curriculum_id": request["curriculum_id"],
        "curriculum_intent_id": request["curriculum_intent_id"],
        "source_id": request["source_id"],
        "source_refs": request["source_refs"],
        "source_hash": request["source_hash"],
        "target_capability": request["target_capability"],
        "skill_kind": request["skill_kind"],
        "cases": ["baseline", "practice", "posttest", "retest", "negative", "fresh", "control"],
        "phase_cases": {phase: phase for phase in ("baseline", "practice", "posttest", "retest", "negative", "fresh", "control")},
        "phase_tasks": {phase: f"{phase} task" for phase in ("baseline", "practice", "posttest", "retest", "negative", "fresh", "control")},
        "evaluator_id": "grader.generic-epub",
        "evaluator_type": "executable_artifact_grader",
        "success_threshold": 1.0,
        "source_free": True,
        "novel_transfer": True,
        "evaluator_independence": "independent",
        "contamination_status": "clean",
        "negative_applicability": True,
    }

    bound = bind_independent_competence_test_design(
        request=request,
        design=design,
        designer_provenance={"provider": "independent-designer", "session_id": "generic-epub-design", "role": "test_designer"},
    )

    assert proposal["key"] == treatments[0]["intent_key"]
    assert discovery["source"]["text"]
    assert bound["proposal_id"] == request["proposal_id"]
    assert bound["source_id"] == ingestion["capability_discovery"]["source"]["id"]
    assert bound["phase_cases"]["fresh"] == "fresh"


def test_unit_discovery_rejects_parent_source_from_another_curriculum(tmp_path: Path) -> None:
    engine = LearningEngine(LearningStore(tmp_path / "learning"))
    parent_curriculum = engine.create_curriculum("Parent", "Learn parent")
    target_curriculum = engine.create_curriculum("Target", "Learn target")
    parent = engine.add_source(parent_curriculum["id"], "Parent", "file:///parent.epub", kind="epub", content_hash="parent-hash")

    with pytest.raises(Exception, match="source/curriculum mismatch"):
        discover_capabilities_from_units(
            engine=engine,
            learner=DiscoveryLearner(),
            source_title="Parent",
            source_units=[{"unit_id": "u1", "title": "Unit", "text": "Instruction.", "source_ref": "parent.xhtml#unit:0"}],
            educational_objective="Learn target",
            curriculum_id=target_curriculum["id"],
            source_id=parent["id"],
        )

    assert len(engine.store.list("sources")) == 1


def test_discovery_rejects_source_id_when_supplied_text_hash_differs(tmp_path: Path) -> None:
    engine = LearningEngine(LearningStore(tmp_path / "learning"))
    curriculum = engine.create_curriculum("Hash-bound discovery", "Learn diagnosis")
    persisted_text = "Observe the symptom and test the suspected cause."
    source = engine.add_source(
        curriculum["id"],
        "Hash-bound discovery",
        "file:///lesson.txt",
        content_hash=hashlib.sha256(persisted_text.encode()).hexdigest(),
    )

    with pytest.raises(LearningError, match="source content mismatch"):
        discover_capabilities(
            engine=engine,
            learner=DiscoveryLearner(),
            source_title="Hash-bound discovery",
            source_text="Substituted instructional text.",
            educational_objective="Learn diagnosis",
            curriculum_id=curriculum["id"],
            source_id=source["id"],
        )

    assert engine.store.list("capability_proposals") == []


def test_independent_test_design_rejects_hidden_required_artifact_fields() -> None:
    request = {
        "schema": "rex-learning-test-design-request-v1",
        "proposal_id": "proposal-contract-task",
        "intent_key": "diagnose-failure",
        "curriculum_id": "curriculum-contract-task",
        "curriculum_intent_id": "intent-contract-task",
        "source_id": "source-contract-task",
        "source_refs": ["source-contract-task#diagnosis"],
        "source_hash": "hash-contract-task",
        "target_capability": "Diagnose a failure by testing hypotheses",
        "skill_kind": "procedure",
        "procedure": "Reproduce the failure and test hypotheses.",
        "status": "awaiting_independent_design",
    }
    phases = ("baseline", "practice", "posttest", "retest", "negative", "fresh", "control")
    design = {
        **{key: request[key] for key in ("proposal_id", "curriculum_id", "curriculum_intent_id", "source_id", "source_refs", "source_hash", "target_capability", "skill_kind")},
        "cases": list(phases),
        "phase_cases": {phase: phase for phase in phases},
        "phase_tasks": {phase: "LEARNER-VISIBLE FIXTURE: diagnose the failure and return an artifact." for phase in phases},
        "evaluation_contracts": {
            phase: json.dumps({"artifact_type": "json_response", "required_fields": ["artifact", "root_cause"]})
            for phase in phases
        },
        "evaluator_id": "grader.contract-task",
        "evaluator_type": "executable_grader",
        "source_free": True,
        "novel_transfer": True,
        "evaluator_independence": "independent",
        "contamination_status": "clean",
        "negative_applicability": True,
        "self_contained_tasks": True,
        "success_threshold": 1.0,
    }
    with pytest.raises(LearningError, match="learner output fields"):
        bind_independent_competence_test_design(
            request=request,
            design=design,
            designer_provenance={"provider": "independent-test-designer", "session_id": "designer-contract-task", "role": "test_designer"},
        )

    fields_visible = {
        **design,
        "phase_tasks": {
            phase: "LEARNER-VISIBLE FIXTURE: diagnose the failure by testing hypotheses; return responses, answer, artifact, and root_cause for the diagnosis."
            for phase in phases
        },
    }
    with pytest.raises(LearningError, match="OUTPUT JSON SHAPE"):
        bind_independent_competence_test_design(
            request=request,
            design=fields_visible,
            designer_provenance={"provider": "independent-test-designer", "session_id": "designer-contract-task", "role": "test_designer"},
        )

    executable_contract = json.dumps({
        "artifact_type": "diagnosis",
        "required_fields": ["artifact", "root_cause"],
        "field_types": {"artifact": "array<object>", "root_cause": "string"},
        "assertions": [{"path": "$.artifact", "operator": "min_items", "observable": 1}],
    })
    invalid_contract = {
        **fields_visible,
        "phase_tasks": {
            phase: "LEARNER-VISIBLE FIXTURE: OUTPUT JSON SHAPE: return responses, answer, artifact, and root_cause."
            for phase in phases
        },
        "evaluation_contracts": {phase: executable_contract for phase in phases},
    }
    with pytest.raises(LearningError, match="unsupported field type"):
        bind_independent_competence_test_design(
            request=request,
            design=invalid_contract,
            designer_provenance={"provider": "independent-test-designer", "session_id": "designer-contract-task", "role": "test_designer"},
        )
def test_normalize_independent_test_design_rejects_provider_authored_execution_command():
    phases = []
    for phase in ("baseline", "practice", "posttest", "retest", "negative", "fresh", "control"):
        phase = {
            "phase": phase,
            "case_id": f"{phase}-case",
            "task": f"Complete the {phase} task.",
            "artifact_schema": {
                "artifact_type": "object",
                "required_fields": ["decision"],
                "field_types": {"decision": "string"},
                "assertions": [{"field": "decision", "type_is": "string"}],
            },
        }
        if phase["phase"] == "posttest":
            phase["mutation_contract"] = {"test_command": "python3 mutation_harness.py learner_artifact.json"}
        phases.append(phase)
    design = {
        "phases": phases,
    }
    with pytest.raises(LearningError, match="provider-authored execution command"):
        normalize_independent_test_design(design)


def test_normalize_independent_test_design_rejects_negative_phase_without_applicability_decision():
    phases = ("baseline", "practice", "posttest", "retest", "negative", "fresh", "control")
    raw = {
        "self_contained_tasks": True,
        "phases": [
            {
                "phase": phase,
                "case_id": f"{phase}-case",
                "task": f"LEARNER-VISIBLE FIXTURE: complete this {phase} metrics task. OUTPUT JSON SHAPE: object.",
                "artifact_schema": {
                    "artifact_type": "metrics_report",
                    "required_fields": ["lcom" if phase != "negative" else "metric"],
                    "field_types": {"lcom" if phase != "negative" else "metric": "integer"},
                    "assertions": [{"path": "$.lcom" if phase != "negative" else "$.metric", "operator": "type_is", "value": "integer"}],
                },
            }
            for phase in phases
        ],
    }
    with pytest.raises(LearningError, match="applicability or scope decision"):
        normalize_independent_test_design(raw)
