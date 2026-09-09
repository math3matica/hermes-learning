from pathlib import Path
import sys
from types import SimpleNamespace
from zipfile import ZipFile
import importlib

import pytest

from rex_learning import ExecutableProjectContract, ExecutableProjectEvaluator, LearningEngine, LearningError, LearningStore, autonomous_study
from rex_learning.preflight import _materialize_learner_contract_declarations, _validate_executable_mutation_fixture, prepare_independent_designs, validate_pilot_preflight
from rex_learning.capability_discovery import define_test_from_independent_design


class DiscoveryLearner:
    provider = "test-discovery"
    session_id = "study-session"

    def answer(self, *, task: str, material: str = "", revision: str = "") -> dict:
        return {
            "responses": [{
                "case": "capability_discovery",
                "answer": {
                    "capabilities": [{
                        "capability": "Extract a tested function",
                        "kind": "procedure",
                        "procedure": "Identify repeated logic and extract a tested function.",
                        "proposed_practice": "Extract repeated logic in a fresh example and run its tests.",
                        "proposed_evidence": "Compile and run tests on a held-out extraction task.",
                    }],
                },
            }],
        }


def _make_epub(path: Path) -> None:
    with ZipFile(path, "w") as book:
        book.writestr("chapter.xhtml", "<h1>Procedure</h1><p>Identify repeated logic and extract a tested function.</p>")


def test_autonomous_study_routes_generic_epub_to_independent_design_binding(tmp_path: Path) -> None:
    epub = tmp_path / "guide.epub"
    _make_epub(epub)
    engine = LearningEngine(LearningStore(tmp_path / "learning"))
    learner = DiscoveryLearner()
    calls: list[dict] = []

    def design_provider(request: dict) -> dict:
        calls.append(request)
        phase_cases = {phase: f"{phase}-case" for phase in ("baseline", "practice", "posttest", "retest", "negative", "fresh", "control")}
        return {
            "design": {
                "cases": list(phase_cases.values()),
                "phase_cases": phase_cases,
                "phase_tasks": {phase: f"Learner-visible fixture: complete the {phase} task." for phase in phase_cases},
                "evaluation_contracts": {phase: "Require extracting repeated logic into a tested function." for phase in phase_cases},
                "evaluator_id": "grader.not-bound",
                "evaluator_type": "semantic_checklist_grader",
                "success_threshold": 1.0,
                "source_free": True,
                "novel_transfer": True,
                "evaluator_independence": "independent",
                "contamination_status": "clean",
                "negative_applicability": True,
                "self_contained_tasks": True,
            },
            "designer_provenance": {"provider": "test-designer", "session_id": "design-session", "role": "test_designer"},
        }

    result = autonomous_study(
        engine=engine,
        source_path=epub,
        learner=learner,
        educational_objective="Find transferable software procedures",
        design_provider=design_provider,
    )

    assert result["schema"] == "rex-learning-autonomous-study-v1"
    assert result["ingestion"]["capability_discovery"]["schema"] == "rex-learning-capability-discovery-v1"
    assert len(result["treatments"]) == len(result["requests"]) == len(result["bound_designs"]) == 1
    assert calls[0]["status"] == "awaiting_independent_design"
    assert "answer_key" not in calls[0]
    assert result["bound_designs"][0]["status"] == "awaiting_trusted_evaluator_binding"
    assert result["preflight"]["schema"] == "rex-learning-independent-design-preflight-v1"
    assert result["preflight"]["status"] == "passed"
    assert result["preflight"]["discovery_hash"]


def test_materialize_learner_contract_declarations_exposes_schema_without_answers() -> None:
    contract = '{"artifact_type":"record","required_fields":["decision"],"field_types":{"decision":"string"},"assertions":[{"path":"$.decision","operator":"value_is","value":"reject"}]}'
    task = "LEARNER-VISIBLE FIXTURE: {\"request\":\"classify\"}."
    materialized = _materialize_learner_contract_declarations(task, contract)
    assert "REQUIRED OUTPUT FIELDS: decision (string)." in materialized
    assert "value_is" not in materialized
    assert "reject" not in materialized
    assert "answer_key" not in materialized
    assert "expected_case_results" not in materialized


def test_normalize_independent_test_design_rejects_expected_value_in_task_text() -> None:
    phases = {}
    for phase in ("baseline", "practice", "posttest", "retest", "negative", "fresh", "control"):
        phases[phase] = {
            "case_id": f"{phase}-case",
            "task": (
                'LEARNER-VISIBLE FIXTURE: classify the request. '
                '$.selected_process must satisfy value_is with value "Waterfall".'
            ),
            "artifact_schema": {
                "artifact_type": "object",
                "required_fields": ["selected_process"],
                "field_types": {"selected_process": "string"},
                "assertions": [{"path": "$.selected_process", "operator": "value_is", "value": "Waterfall"}],
            },
        }
    with pytest.raises(LearningError, match="expected answer in learner-visible task"):
        from rex_learning.capability_discovery import normalize_independent_test_design

        normalize_independent_test_design({"phases": list(dict(item, phase=phase) for phase, item in phases.items())})


def test_normalize_independent_test_design_rejects_exact_value_constraint_in_task_text() -> None:
    phases = []
    for phase in ("baseline", "practice", "posttest", "retest", "negative", "fresh", "control"):
        phases.append({
            "phase": phase,
            "case_id": f"{phase}-case",
            "task": "LEARNER-VISIBLE FIXTURE: configure the experiment; power requires the exact value 0.8.",
            "artifact_schema": {
                "artifact_type": "object",
                "required_fields": ["power"],
                "field_types": {"power": "number"},
                "assertions": [{"path": "$.power", "operator": "value_is", "value": 0.8}],
            },
        })
    with pytest.raises(LearningError, match="expected answer in learner-visible task"):
        from rex_learning.capability_discovery import normalize_independent_test_design

        normalize_independent_test_design({"phases": phases})


def test_prepare_independent_designs_rejects_incomplete_provider_response(tmp_path: Path) -> None:
    source = tmp_path / "guide.md"
    source.write_text("# Procedure\n\nIdentify repeated logic and extract a tested function.\n", encoding="utf-8")
    engine = LearningEngine(LearningStore(tmp_path / "learning"))
    curriculum = engine.create_curriculum(source.name, "Find transferable software procedures")
    from rex_learning.ingestion import ingest_source

    discovery = ingest_source(
        engine,
        curriculum["id"],
        source,
        learner=DiscoveryLearner(),
        educational_objective="Find transferable software procedures",
    )["capability_discovery"]

    with pytest.raises(LearningError, match="independent design provider response is incomplete"):
        prepare_independent_designs(
            discovery=discovery,
            design_provider=lambda request: {"design": {}},
        )


def test_validate_pilot_preflight_rejects_missing_trusted_evaluator_support(tmp_path: Path) -> None:
    packet = {
        "schema": "rex-learning-independent-design-preflight-v1",
        "status": "passed",
        "bound_designs": [{"evaluator_id": "grader.missing", "evaluator_type": "executable_process_grader"}],
    }
    evaluator = SimpleNamespace(
        evaluator_type="executable_process_grader",
        command=(sys.executable, "mutation_harness.py"),
    )
    with pytest.raises(LearningError, match="support file is missing"):
        validate_pilot_preflight(
            packet,
            trusted_evaluators={"grader.missing": evaluator},
            command_cwd=tmp_path,
        )


def test_validate_pilot_preflight_accepts_hermes_owned_project_evaluator(tmp_path: Path) -> None:
    checker = tmp_path / "check.py"
    checker.write_text("import pathlib\nassert pathlib.Path('app.py').exists()\n", encoding="utf-8")
    contract = ExecutableProjectContract(
        source_files=("app.py",),
        support_files={"check.py": checker.read_text(encoding="utf-8")},
        test_command=(sys.executable, "check.py"),
        provenance="hermes-preflight-fixture-v1",
    )
    evaluator = ExecutableProjectEvaluator("project.grader", "hermes", "project-session", contract)
    packet = {
        "schema": "rex-learning-independent-design-preflight-v1",
        "status": "passed",
        "bound_designs": [{
            "evaluator_id": evaluator.evaluator_id,
            "evaluator_type": evaluator.evaluator_type,
            "project_contract": contract.to_dict(),
            "project_contract_sha256": contract.sha256,
        }],
    }
    validate_pilot_preflight(packet, trusted_evaluators={evaluator.evaluator_id: evaluator}, command_cwd=tmp_path)


def test_define_independent_project_test_preserves_frozen_project_contract(tmp_path: Path) -> None:
    contract = ExecutableProjectContract(
        source_files={"app.py": ""},
        support_files={"check.py": ""},
        test_command=(sys.executable, "check.py"),
        provenance="hermes-test-contract-v1",
    )
    evaluator = ExecutableProjectEvaluator("project.materialized", "hermes", "project-session", contract)
    engine = LearningEngine(LearningStore(tmp_path / "learning"), trusted_evaluators={evaluator.evaluator_id: evaluator})
    curriculum = engine.create_curriculum("project", "test")
    skill = engine.create_skill_hypothesis(curriculum["id"], {"key": "project", "kind": "procedure", "claim": "test", "applicability": ["test"], "operational_procedure": "test"})
    design = {
        "schema": "rex-learning-independent-test-design-v1", "status": "awaiting_trusted_evaluator_binding",
        "proposal_id": "proposal-project", "intent_key": "project", "curriculum_id": curriculum["id"],
        "curriculum_intent_id": "intent-project", "source_id": "source-project", "source_hash": "hash-project",
        "target_capability": "test", "skill_kind": "procedure", "source_refs": ["ref"],
        "source_free": True, "novel_transfer": True, "evaluator_independence": "independent",
        "contamination_status": "clean", "negative_applicability": True, "evaluator_id": evaluator.evaluator_id,
        "evaluator_type": evaluator.evaluator_type, "cases": ["case-baseline", "case-practice", "case-posttest", "case-retest", "case-negative", "case-fresh", "case-control"], "phase_cases": {phase: f"case-{phase}" for phase in ("baseline", "practice", "posttest", "retest", "negative", "fresh", "control")},
        "phase_tasks": {phase: "task" for phase in ("baseline", "practice", "posttest", "retest", "negative", "fresh", "control")},
        "designer_provenance": {"provider": "designer", "session_id": "designer-session", "role": "test_designer"},
        "discovery_provenance": {"provider": "discoverer", "session_id": "discovery-session"},
        "project_contract": contract.to_dict(), "project_contract_sha256": contract.sha256,
    }
    test = define_test_from_independent_design(engine=engine, curriculum_id=curriculum["id"], skill_id=skill["id"], design=design, phase="pretest")
    assert test["project_contract"] == contract.to_dict()
    assert test["project_contract_sha256"] == contract.sha256


def test_validate_pilot_preflight_rejects_provider_authored_execution_commands(tmp_path: Path) -> None:
    """A design provider must not choose commands for Hermes to execute."""
    evaluator_script = tmp_path / "validator.py"
    evaluator_script.write_text("import sys\n", encoding="utf-8")
    packet = {
        "schema": "rex-learning-independent-design-preflight-v1",
        "status": "passed",
        "bound_designs": [{
            "evaluator_id": "grader.present",
            "evaluator_type": "executable_process_grader",
            "mutation_contracts": {
                "posttest": {
                    "test_command": "python3 mutation_harness.py learner_artifact.json --mutant M-1"
                }
            },
        }],
    }
    evaluator = SimpleNamespace(
        evaluator_type="executable_process_grader",
        command=(sys.executable, str(evaluator_script)),
    )
    with pytest.raises(LearningError, match="provider-authored execution command"):
        validate_pilot_preflight(
            packet,
            trusted_evaluators={"grader.present": evaluator},
            command_cwd=tmp_path,
        )


def test_validate_pilot_preflight_rejects_incomplete_executable_mutation_fixture(tmp_path: Path) -> None:
    evaluator_script = tmp_path / "validator.py"
    evaluator_script.write_text("import sys\n", encoding="utf-8")
    phases = ("baseline", "practice", "posttest", "retest", "negative", "fresh", "control")
    incomplete_task = (
        "Learner-visible fixture: mutation diagnosis.\n"
        "Original production class:\n```java\npublic final class Example {}\n```\n"
        "The non-equivalent mutant is described in prose but its complete class is omitted."
    )
    packet = {
        "schema": "rex-learning-independent-design-preflight-v1",
        "status": "passed",
        "bound_designs": [{
            "evaluator_id": "grader.present",
            "evaluator_type": "executable_process_grader",
            "phase_tasks": {phase: incomplete_task for phase in phases},
        }],
    }
    evaluator = SimpleNamespace(
        evaluator_type="executable_process_grader",
        command=(sys.executable, str(evaluator_script)),
    )
    with pytest.raises(LearningError, match="complete.*mutant fixture"):
        validate_pilot_preflight(
            packet,
            trusted_evaluators={"grader.present": evaluator},
            command_cwd=tmp_path,
        )


def test_executable_mutation_fixture_gate_ignores_non_mutation_boundary_task() -> None:
    _validate_executable_mutation_fixture(
        "Learner-visible fixture: this is an integration-boundary decision. "
        "Do not apply mutation diagnosis; explain why the paired components require integration scope."
    )


def test_executable_mutation_fixture_gate_accepts_labeled_letter_variants() -> None:
    _validate_executable_mutation_fixture(
        "Learner-visible fixture: mutation diagnosis.\n"
        "Original production class:\n"
        "public final class Example {}\n"
        "Complete mutant production class A:\n"
        "public final class Example {}\n"
    )


def test_autonomous_study_accepts_markdown_source_and_preserves_lineage(tmp_path: Path) -> None:
    source = tmp_path / "guide.md"
    source.write_text("# Procedure\n\nIdentify repeated logic and extract a tested function.\n", encoding="utf-8")
    engine = LearningEngine(LearningStore(tmp_path / "learning"))
    def design_provider(request: dict) -> dict:
        phases = ("baseline", "practice", "posttest", "retest", "negative", "fresh", "control")
        return {
            "design": {
                **{key: request[key] for key in ("proposal_id", "curriculum_id", "curriculum_intent_id", "source_id", "source_hash", "target_capability", "skill_kind")},
                "intent_key": request["intent_key"], "source_refs": request["source_refs"],
                "cases": [f"{phase}-case" for phase in phases],
                "phase_cases": {phase: f"{phase}-case" for phase in phases},
                "phase_tasks": {phase: f"Learner-visible fixture: complete the {phase} task." for phase in phases},
                "evaluation_contracts": {phase: "Require extracting repeated logic into a tested function." for phase in phases},
                "evaluator_id": "grader.not-bound", "evaluator_type": "semantic_checklist_grader",
                "success_threshold": 1.0, "source_free": True, "novel_transfer": True,
                "evaluator_independence": "independent", "contamination_status": "clean",
                "negative_applicability": True, "self_contained_tasks": True,
            },
            "designer_provenance": {"provider": "test-designer", "session_id": "design-session", "role": "test_designer"},
        }

    result = autonomous_study(
        engine=engine, source_path=source, learner=DiscoveryLearner(),
        educational_objective="Find transferable software procedures", design_provider=design_provider,
    )
    assert result["ingestion"]["source"]["kind"] == "markdown"
    assert result["ingestion"]["chunks"]
    discovery = result["discovery"]
    assert discovery["source"]["id"] == result["ingestion"]["source"]["id"]
    assert all(proposal["source_id"] == result["ingestion"]["source"]["id"] for proposal in discovery["proposals"])


def test_autonomous_study_forwards_dependencies_to_qualification_executor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "guide.md"
    source.write_text("# Procedure\n\nIdentify repeated logic and extract a tested function.\n", encoding="utf-8")
    engine = LearningEngine(LearningStore(tmp_path / "learning"))
    module = importlib.import_module("rex_learning.autonomous_study")
    captured: dict = {}

    def design_provider(request: dict) -> dict:
        phases = ("baseline", "practice", "posttest", "retest", "negative", "fresh", "control")
        return {
            "design": {
                **{key: request[key] for key in ("proposal_id", "curriculum_id", "curriculum_intent_id", "source_id", "source_hash", "target_capability", "skill_kind")},
                "intent_key": request["intent_key"], "source_refs": request["source_refs"],
                "cases": [f"{phase}-case" for phase in phases],
                "phase_cases": {phase: f"{phase}-case" for phase in phases},
                "phase_tasks": {phase: f"Learner-visible fixture: complete the {phase} task." for phase in phases},
                "evaluation_contracts": {phase: "Require extracting repeated logic into a tested function." for phase in phases},
                "evaluator_id": "grader.not-bound", "evaluator_type": "semantic_checklist_grader",
                "success_threshold": 1.0, "source_free": True, "novel_transfer": True,
                "evaluator_independence": "independent", "contamination_status": "clean",
                "negative_applicability": True, "self_contained_tasks": True,
            },
            "designer_provenance": {"provider": "test-designer", "session_id": "design-session", "role": "test_designer"},
        }

    def fake_execute(**kwargs: dict) -> dict:
        captured.update(kwargs)
        return {"schema": "rex-learning-treatment-execution-v1", "runs": [], "skipped": [], "successful_acquisitions": 0}

    monkeypatch.setattr(module, "execute_learning_treatments", fake_execute)
    dependencies = {"*": {"grader": object(), "reviewer": object(), "fresh_learner": object(), "control_learner": object(), "learner": DiscoveryLearner()}}
    dependencies = type("AnyKeyDependencies", (dict,), {"get": lambda self, key, default=None: super(type(self), self).get("*", default)})(dependencies)
    result = module.autonomous_study(
        engine=engine, source_path=source, learner=DiscoveryLearner(),
        educational_objective="Find transferable software procedures", design_provider=design_provider,
        qualification_dependencies=dependencies,
    )
    assert result["execution"]["schema"] == "rex-learning-treatment-execution-v1"
    assert captured["source_text"].strip()
    assert set(captured["independent_designs"]) == {result["treatments"][0]["intent_key"]}
    assert captured["dependencies"] is dependencies
