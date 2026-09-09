from __future__ import annotations

import json
import hashlib
import sys
import zipfile
from pathlib import Path
from typing import cast

import pytest

from rex_learning import CallbackEvaluator, ContractArtifactEvaluator, ExecutableArtifactGrader, ExecutableGrader, ExecutableProcessGrader, EvaluatorInstabilityError, LearningEngine, LearningError, LearningStore, OpenAICompatibleLearner, SemanticChecklistGrader, StableTrustedEvaluator, StructuredBehaviorGrader, TrustedEvaluator, bind_independent_competence_test_design, build_competence_test_requests, discover_capabilities, ingest_source
from rex_learning import ExecutableProjectContract, ExecutableProjectEvaluator
from rex_learning.contract_validator import evaluate_envelope


def test_callback_evaluator_derives_aggregate_from_independent_case_judgments():
    calls = []

    def judge(*, test, attempt):
        calls.append((test["id"], attempt["id"]))
        return {"case_results": [{"case": "a", "passed": True, "rationale": "ok"}, {"case": "b", "passed": False, "rationale": "missing"}]}

    evaluator = CallbackEvaluator("callback", judge, "independent-provider", "evaluator-session")
    test = {"id": "test-callback", "evaluator_id": "callback", "evaluator_type": "callback_evaluator", "cases": ["a", "b"], "success_threshold": 0.5, "contamination_status": "clean", "evaluator_independence": "independent"}
    result = evaluator.evaluate(test, {"id": "attempt-callback", "responses": []})
    assert calls == [("test-callback", "attempt-callback")]
    assert result["score"] == 0.5
    assert result["passed"] is True
    assert result["case_results"][1]["rationale"] == "missing"


def test_callback_evaluator_rejects_claimed_or_malformed_case_judgments():
    test = {"id": "test-callback-invalid", "evaluator_id": "callback", "evaluator_type": "callback_evaluator", "cases": ["a"], "success_threshold": 1.0}
    evaluator = CallbackEvaluator("callback", lambda **_: {"score": 1.0, "case_results": [{"case": "a", "passed": 1}]}, "provider", "session")
    with pytest.raises(ValueError, match="boolean passed"):
        evaluator.evaluate(test, {"id": "attempt", "responses": []})


def test_callback_evaluator_enforces_frozen_artifact_shape_before_semantic_callback():
    calls = []
    test = {
        "id": "test-callback-shape",
        "evaluator_id": "callback",
        "evaluator_type": "callback_evaluator",
        "cases": ["a"],
        "success_threshold": 1.0,
        "evaluation_contract": json.dumps({
            "artifact_type": "decision-record",
            "required_fields": ["decision", "rationale"],
            "field_types": {"decision": "string", "rationale": "string"},
            "assertions": [],
        }),
    }
    evaluator = CallbackEvaluator(
        "callback",
        lambda **_: calls.append(True) or {"case_results": [{"case": "a", "passed": True}]},
        "provider",
        "session",
    )
    with pytest.raises(ValueError, match="missing required field"):
        evaluator.evaluate(test, {"id": "attempt", "responses": [{"case": "a", "answer": {"CBO": 8}}]})
    assert calls == []


def test_callback_evaluator_runs_through_engine_trusted_boundary(tmp_path):
    evaluator = CallbackEvaluator(
        "callback-engine",
        lambda **_: {"case_results": [{"case": "case", "passed": True, "rationale": "valid"}]},
        "independent-provider",
        "evaluator-session",
    )
    engine = LearningEngine(LearningStore(tmp_path / "learning"), trusted_evaluators={evaluator.evaluator_id: evaluator})
    curriculum = engine.create_curriculum("Callback", "Test provider-backed judgment")
    skill = engine.create_skill_hypothesis(curriculum["id"], {"key": "callback", "claim": "Apply a judged capability"})
    test = engine.define_test(curriculum["id"], skill["id"], {
        "phase": "posttest", "cases": ["case"], "evaluator_type": evaluator.evaluator_type,
        "evaluator_id": evaluator.evaluator_id, "success_threshold": 1.0,
        "evaluator_independence": "independent", "contamination_status": "clean",
    })
    attempt = engine.record_attempt(test["id"], {"responses": [{"case": "case", "answer": {"decision": "apply", "rationale": "because"}}]})
    assert engine.evaluate_trusted_attempt(attempt["id"])["passed"] is True


def _contract_test_fixture():
    contract = {
        "artifact_type": "decision-record",
        "required_fields": ["decision", "constraints", "options"],
        "field_types": {"decision": "string", "constraints": "array", "options": "array"},
        "assertions": [
            {"path": "$.decision", "operator": "type_is", "value": "string"},
            {"path": "$.constraints", "operator": "min_items", "value": 2},
            {"path": "$.options", "operator": "all_unique"},
        ],
    }
    test = {
        "id": "test-contract",
        "evaluator_id": "contract-evaluator",
        "evaluator_type": "contract_artifact_evaluator",
        "cases": ["case"],
        "success_threshold": 1.0,
        "contamination_status": "clean",
        "evaluator_independence": "independent",
        "artifact_schema": contract,
    }
    return test, contract


def test_contract_artifact_evaluator_derives_results_from_frozen_schema():
    test, contract = _contract_test_fixture()
    evaluator = ContractArtifactEvaluator("contract-evaluator", "trusted", "contract-session")
    attempt = {
        "id": "attempt-contract",
        "responses": [{"case": "case", "answer": {"decision": "trunk", "constraints": ["cadence", "tests"], "options": ["a", "b"]}}],
    }
    result = evaluator.evaluate(test, attempt)
    assert result["passed"] is True
    assert result["score"] == 1.0
    assert result["case_results"] == [{"case": "case", "passed": True, "failed_assertions": []}]
    assert result["provider_provenance"]["contract"] == contract


def _project_evaluator_fixture(tmp_path):
    checker = tmp_path / "check.py"
    checker.write_text("import pathlib, sys\nassert pathlib.Path('app.py').read_text() == 'print(\\\"ok\\\")\\n'\n", encoding="utf-8")
    contract = ExecutableProjectContract(
        source_files=("app.py",),
        support_files={"check.py": checker.read_text(encoding="utf-8")},
        test_command=(sys.executable, "check.py"),
        provenance="hermes-test-fixture-v1",
    )
    test = {"id": "project-test", "evaluator_id": "project", "evaluator_type": "executable_project_evaluator", "cases": ["case"], "success_threshold": 1.0, "contamination_status": "clean", "evaluator_independence": "independent", "project_contract": contract.to_dict(), "project_contract_sha256": contract.sha256}
    evaluator = ExecutableProjectEvaluator("project", "hermes", "project-session", contract)
    return evaluator, test


def test_executable_project_evaluator_rejects_claimed_booleans_without_project(tmp_path):
    evaluator, test = _project_evaluator_fixture(tmp_path)
    attempt = {"id": "booleans", "responses": [{"case": "case", "answer": {"passed": True, "compiled": True, "tests_passed": True}}]}
    result = evaluator.evaluate(test, attempt)
    assert result["passed"] is False
    assert result["score"] == 0.0


def test_executable_project_evaluator_runs_minimal_project_and_discriminates_wrong_omitted_control(tmp_path):
    evaluator, test = _project_evaluator_fixture(tmp_path)
    def attempt(answer, name):
        return evaluator.evaluate(test, {"id": name, "responses": [{"case": "case", "answer": answer}]})
    valid = attempt({"files": {"app.py": 'print("ok")\n'}}, "valid")
    wrong = attempt({"files": {"app.py": 'print("wrong")\n'}}, "wrong")
    omitted = attempt({"files": {}}, "omitted")
    control = attempt({"files": {"app.py": 'print("ok")\n'}, "passed": False}, "control")
    assert valid["passed"] is True and valid["case_results"][0]["passed"] is True
    assert wrong["passed"] is False and omitted["passed"] is False
    assert control["passed"] is True  # learner claims are ignored


def test_executable_project_evaluator_enforces_contract_provenance_and_hash(tmp_path):
    evaluator, test = _project_evaluator_fixture(tmp_path)
    attempt = {"id": "valid", "responses": [{"case": "case", "answer": {"files": {"app.py": 'print("ok")\n'}}}]}
    tampered = dict(test, project_contract_sha256="0" * 64)
    with pytest.raises(ValueError, match="contract hash"):
        evaluator.evaluate(tampered, attempt)
    tampered = dict(test, project_contract=dict(test["project_contract"], provenance="provider"))
    with pytest.raises(ValueError, match="provenance"):
        evaluator.evaluate(tampered, attempt)


def test_executable_project_evaluator_rejects_learner_overwriting_support_files(tmp_path):
    evaluator, test = _project_evaluator_fixture(tmp_path)
    attempt = {
        "id": "support-overwrite",
        "responses": [{
            "case": "case",
            "answer": {"files": {"app.py": 'print("ok")\n', "check.py": "raise SystemExit(0)\n"}},
        }],
    }
    with pytest.raises(ValueError, match="support file"):
        evaluator.evaluate(test, attempt)


def test_contract_artifact_evaluator_falls_back_from_empty_artifact_schema_to_contract():
    test, contract = _contract_test_fixture()
    test["artifact_schema"] = {}
    test["evaluation_contract"] = json.dumps(contract)
    evaluator = ContractArtifactEvaluator("contract-evaluator", "trusted", "contract-session")
    attempt = {
        "id": "attempt-contract-fallback",
        "responses": [{"case": "case", "answer": {"decision": "trunk", "constraints": ["cadence", "tests"], "options": ["a", "b"]}}],
    }
    assert evaluator.evaluate(test, attempt)["passed"] is True


def test_contract_artifact_evaluator_rejects_wrong_type_and_duplicate_values():
    test, _ = _contract_test_fixture()
    evaluator = ContractArtifactEvaluator("contract-evaluator", "trusted", "contract-session")
    attempt = {
        "id": "attempt-contract-bad",
        "responses": [{"case": "case", "answer": {"decision": 7, "constraints": ["only-one"], "options": ["a", "a"]}}],
    }
    result = evaluator.evaluate(test, attempt)
    assert result["passed"] is False
    assert result["score"] == 0.0
    assert {item["operator"] for item in result["case_results"][0]["failed_assertions"]} == {"type_is", "min_items", "all_unique"}


def test_contract_artifact_evaluator_checks_array_item_paths():
    test, _ = _contract_test_fixture()
    test["artifact_schema"]["required_fields"] = ["items"]
    test["artifact_schema"]["field_types"] = {"items": "array"}
    test["artifact_schema"]["assertions"] = [{"path": "$.items[*].name", "operator": "type_is", "value": "string"}]
    evaluator = ContractArtifactEvaluator("contract-evaluator", "trusted", "contract-session")
    attempt = {"id": "attempt-contract-wildcard", "responses": [{"case": "case", "answer": {"items": [{"name": "ok"}, {"name": "also-ok"}]}}]}
    result = evaluator.evaluate(test, attempt)
    assert result["passed"] is True
    assert result["case_results"][0]["failed_assertions"] == []


def test_contract_validator_adapter_uses_frozen_string_contract_and_wildcards():
    _, contract = _contract_test_fixture()
    contract["required_fields"] = ["items"]
    contract["field_types"] = {"items": "array"}
    contract["assertions"] = [{"path": "$.items[*].name", "operator": "type_is", "value": "string"}]
    test = {
        "id": "test-adapter",
        "evaluator_id": "pilot-executable-grader",
        "evaluator_type": "executable_process_grader",
        "cases": ["case"],
        "success_threshold": 1.0,
        "artifact_schema": {},
        "evaluation_contract": json.dumps(contract),
    }
    envelope = {"test": test, "attempt": {"id": "attempt-adapter", "responses": [{"case": "case", "answer": {"items": [{"name": "a"}, {"name": "b"}]}}]}}
    result = evaluate_envelope(envelope)
    assert result == {"case_results": [{"case": "case", "passed": True, "failed_assertions": []}]}


def test_contract_artifact_evaluator_fails_closed_for_unknown_assertion_operator():
    test, _ = _contract_test_fixture()
    test["artifact_schema"]["assertions"].append({"path": "$.decision", "operator": "execute_code", "value": "bad"})
    evaluator = ContractArtifactEvaluator("contract-evaluator", "trusted", "contract-session")
    attempt = {"id": "attempt-contract-unknown", "responses": [{"case": "case", "answer": {"decision": "x", "constraints": ["a", "b"], "options": ["a", "b"]}}]}
    with pytest.raises(ValueError, match="unsupported contract assertion operator"):
        evaluator.evaluate(test, attempt)


def test_contract_artifact_evaluator_checks_exact_scalar_values():
    test, _ = _contract_test_fixture()
    test["artifact_schema"]["assertions"] = [
        {"path": "$.decision", "operator": "value_is", "value": "trunk"},
        {"path": "$.enabled", "operator": "value_is", "value": False},
    ]
    test["artifact_schema"]["required_fields"] = ["decision", "enabled"]
    test["artifact_schema"]["field_types"] = {"decision": "string", "enabled": "boolean"}
    evaluator = ContractArtifactEvaluator("contract-evaluator", "trusted", "contract-session")
    passing = {
        "id": "attempt-contract-values-pass",
        "responses": [{"case": "case", "answer": {"decision": "trunk", "enabled": False}}],
    }
    failing = {
        "id": "attempt-contract-values-fail",
        "responses": [{"case": "case", "answer": {"decision": "branch", "enabled": True}}],
    }
    assert evaluator.evaluate(test, passing)["passed"] is True
    result = evaluator.evaluate(test, failing)
    assert result["passed"] is False
    assert [item["operator"] for item in result["case_results"][0]["failed_assertions"]] == ["value_is", "value_is"]


def test_contract_artifact_evaluator_is_registered_and_persisted_by_engine(tmp_path):
    evaluator = ContractArtifactEvaluator("contract-evaluator", "trusted", "contract-session")
    engine = LearningEngine(LearningStore(tmp_path / "learning"), trusted_evaluators={evaluator.evaluator_id: evaluator})
    curriculum = engine.create_curriculum("Contract", "Test frozen artifacts")
    skill = engine.create_skill_hypothesis(curriculum["id"], {"key": "contract", "claim": "Return structured artifacts"})
    _, contract = _contract_test_fixture()
    test = engine.define_test(curriculum["id"], skill["id"], {"phase": "posttest", "cases": ["case"], "evaluator_type": evaluator.evaluator_type, "evaluator_id": evaluator.evaluator_id, "artifact_schema": contract, "success_threshold": 1.0, "evaluator_independence": "independent", "contamination_status": "clean"})
    assert test["artifact_schema"] == contract
    attempt = engine.record_attempt(test["id"], {"responses": [{"case": "case", "answer": {"decision": "x", "constraints": ["a", "b"], "options": ["a", "b"]}}]})
    assert engine.evaluate_trusted_attempt(attempt["id"])["passed"] is True


def test_semantic_checklist_grader_verifies_open_ended_propositions(tmp_path):
    base_grader = SemanticChecklistGrader(
        evaluator_id="grader.checklist.v1",
        required_propositions={"case": ["retrieve", "feedback"]},
        provider="independent-checklist",
        session_id="checklist-session",
    )
    grader = StableTrustedEvaluator(cast(TrustedEvaluator, base_grader), base_grader.evaluator_id, base_grader.evaluator_type)
    engine = LearningEngine(LearningStore(tmp_path / "learning"), trusted_evaluators=cast(dict, {grader.evaluator_id: grader}))
    curriculum = engine.create_curriculum("Checklist", "Test open-ended grading")
    skill = engine.create_skill_hypothesis(curriculum["id"], {"key": "checklist", "claim": "Use retrieval", "applicability": ["study"]})
    test = engine.define_test(curriculum["id"], skill["id"], {"phase": "posttest", "cases": ["case"], "evaluator_type": grader.evaluator_type, "evaluator_id": grader.evaluator_id, "success_threshold": 1.0, "evaluator_independence": "independent", "contamination_status": "clean"})
    attempt = engine.record_attempt(test["id"], {"responses": [{"case": "case", "answer": "Retrieve with feedback."}]})
    evaluation = engine.evaluate_trusted_attempt(attempt["id"])
    assert evaluation["passed"] is True
    assert evaluation["evaluator_type"] == "semantic_checklist_grader"
    assert evaluation["provider_provenance"]["required_propositions"] == {"case": ["retrieve", "feedback"]}


def test_contract_artifact_replay_distinguishes_valid_retest_negative_and_malformed_artifacts():
    evaluator = ContractArtifactEvaluator(
        evaluator_id="replay.contract.v1",
        provider="hermes-replay",
        session_id="replay-session",
    )
    schema = {
        "artifact_type": "object",
        "required_fields": ["decision", "invariants"],
        "field_types": {"decision": "string", "invariants": "array"},
        "assertions": [
            {"path": "$.decision", "operator": "value_is", "value": "preserve"},
            {"path": "$.invariants", "operator": "array_length", "value": 2},
            {"path": "$.invariants[*]", "operator": "type_is", "value": "string"},
            {"path": "$.invariants", "operator": "all_unique"},
        ],
    }

    def test(case: str) -> dict:
        return {
            "id": f"test-{case}",
            "cases": [case],
            "evaluator_id": evaluator.evaluator_id,
            "evaluator_type": evaluator.evaluator_type,
            "success_threshold": 1.0,
            "artifact_schema": schema,
            "contamination_status": "clean",
            "evaluator_independence": "independent",
        }

    valid = evaluator.evaluate(
        test("posttest"),
        {"id": "attempt-posttest", "responses": [{"case": "posttest", "answer": {"decision": "preserve", "invariants": ["ordering", "state"]}}]},
    )
    assert valid["score"] == 1.0
    assert valid["passed"] is True

    retest_missing_invariant = evaluator.evaluate(
        test("retest"),
        {"id": "attempt-retest", "responses": [{"case": "retest", "answer": {"decision": "preserve", "invariants": ["ordering"]}}]},
    )
    assert retest_missing_invariant["score"] == 0.0
    assert retest_missing_invariant["passed"] is False
    assert {item["operator"] for item in retest_missing_invariant["case_results"][0]["failed_assertions"]} == {"array_length"}

    negative_wrong_decision = evaluator.evaluate(
        test("negative"),
        {"id": "attempt-negative", "responses": [{"case": "negative", "answer": {"decision": "ignore", "invariants": ["ordering", "state"]}}]},
    )
    assert negative_wrong_decision["score"] == 0.0
    assert negative_wrong_decision["passed"] is False

    malformed = evaluator.evaluate(
        test("control"),
        {"id": "attempt-control", "responses": [{"case": "control", "answer": "not an artifact"}]},
    )
    assert malformed["score"] == 0.0
    assert malformed["passed"] is False


def test_executable_process_grader_derives_score_from_hermes_owned_validator(tmp_path):
    validator = tmp_path / "validator.py"
    validator.write_text(
        "import json, sys\n"
        "payload = json.load(sys.stdin)\n"
        "case = payload['attempt']['responses'][0]['case']\n"
        "print(json.dumps({'case_results': [{'case': case, 'passed': True, 'rationale': 'validator-owned'}], 'score': 0}))\n",
        encoding="utf-8",
    )
    grader = ExecutableProcessGrader(
        evaluator_id="grader.process.v1",
        command=(sys.executable, str(validator)),
        provider="independent-executable-validator",
        session_id="process-session",
    )
    engine = LearningEngine(LearningStore(tmp_path / "learning"), trusted_evaluators=cast(dict, {grader.evaluator_id: grader}))
    curriculum = engine.create_curriculum("Process", "Test executable process grading")
    skill = engine.create_skill_hypothesis(curriculum["id"], {"key": "process", "claim": "Use a process validator"})
    test = engine.define_test(curriculum["id"], skill["id"], {"phase": "posttest", "cases": ["case"], "evaluator_type": grader.evaluator_type, "evaluator_id": grader.evaluator_id, "success_threshold": 1.0, "evaluator_independence": "independent", "contamination_status": "clean"})
    attempt = engine.record_attempt(test["id"], {"responses": [{"case": "case", "answer": {"artifact": "ok"}}]})
    evaluation = engine.evaluate_trusted_attempt(attempt["id"])
    assert evaluation["passed"] is True
    assert evaluation["score"] == 1.0
    assert evaluation["provider_provenance"]["command"] == [sys.executable, str(validator)]
    assert evaluation["provider_provenance"]["validator_sha256"] == hashlib.sha256(validator.read_bytes()).hexdigest()
    assert evaluation["provider_provenance"]["validator_stdout_sha256"] == hashlib.sha256(
        b'{"case_results": [{"case": "case", "passed": true, "rationale": "validator-owned"}], "score": 0}\n'
    ).hexdigest()
    assert evaluation["provider_provenance"]["validator_stdout_bytes"] > 0


def test_openai_learner_accepts_literal_escaped_fenced_json_content():
    content = r'```json\n{"responses":[{"case":"case","answer":{"ok":true}}]}\n```'
    response = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
    learner = OpenAICompatibleLearner(
        "http://example.invalid",
        "test-model",
        requester=lambda *_: response,
        format_retry_attempts=0,
    )
    result = learner.answer(task="|case=case|")
    assert result["responses"] == [{"case": "case", "answer": {"ok": True}}]


def test_openai_learner_accepts_direct_fenced_json_response():
    response = b'```json\\n{"responses":[{"case":"case","answer":{"ok":true}}]}\\n```'
    learner = OpenAICompatibleLearner(
        "http://example.invalid",
        "test-model",
        requester=lambda *_: response,
        format_retry_attempts=0,
    )
    result = learner.answer(task="|case=case|")
    assert result["responses"] == [{"case": "case", "answer": {"ok": True}}]


def test_openai_learner_unwraps_one_nested_response_envelope_in_answer():
    nested = json.dumps([{"case": "case", "answer": {"ok": True}}])
    content = json.dumps({"responses": [{"case": "case", "answer": nested}]})
    response = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
    learner = OpenAICompatibleLearner(
        "http://example.invalid",
        "test-model",
        requester=lambda *_: response,
        format_retry_attempts=0,
    )
    result = learner.answer(task="|case=case|")
    assert result["responses"] == [{"case": "case", "answer": {"ok": True}}]


def test_openai_learner_unwraps_one_nested_mapping_response_envelope_in_answer():
    content = json.dumps({"responses": [{"case": "case", "answer": {"case": "case", "answer": {"ok": True}}}]})
    response = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
    learner = OpenAICompatibleLearner(
        "http://example.invalid",
        "test-model",
        requester=lambda *_: response,
        format_retry_attempts=0,
    )
    result = learner.answer(task="|case=case|")
    assert result["responses"] == [{"case": "case", "answer": {"ok": True}}]


def test_openai_learner_wraps_one_top_level_response_list():
    content = json.dumps([{"case": "case", "answer": {"ok": True}}])
    response = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
    learner = OpenAICompatibleLearner(
        "http://example.invalid",
        "test-model",
        requester=lambda *_: response,
        format_retry_attempts=0,
    )
    result = learner.answer(task="|case=case|")
    assert result["responses"] == [{"case": "case", "answer": {"ok": True}}]


def test_openai_learner_prompt_treats_required_fields_as_top_level_artifact_schema():
    captured = {}
    response = json.dumps({"choices": [{"message": {"content": '{"responses":[{"case":"case","answer":{"items":[]}}]}'}}]}).encode()

    def requester(_url, body, _headers, _timeout):
        captured["payload"] = json.loads(body)
        return response

    learner = OpenAICompatibleLearner(
        "http://example.invalid",
        "test-model",
        requester=requester,
        format_retry_attempts=0,
    )
    learner.answer(task="|case=case| REQUIRED FIELDS: items, explanation")
    system = captured["payload"]["messages"][0]["content"]
    assert "REQUIRED FIELDS" in system
    assert "top-level key" in system
    assert "exact requested types" in system


def test_executable_process_grader_uses_explicit_working_directory(tmp_path):
    validator = tmp_path / "validator.py"
    validator.write_text(
        "import json, sys\n"
        "payload = json.load(sys.stdin)\n"
        "case = payload['attempt']['responses'][0]['case']\n"
        "print(json.dumps({'case_results': [{'case': case, 'passed': True}]}))\n",
        encoding="utf-8",
    )
    grader = ExecutableProcessGrader(
        evaluator_id="grader.process.cwd",
        command=(sys.executable, validator.name),
        provider="independent-executable-validator",
        session_id="process-cwd-session",
        working_directory=tmp_path,
    )
    engine = LearningEngine(LearningStore(tmp_path / "learning"), trusted_evaluators=cast(dict, {grader.evaluator_id: grader}))
    curriculum = engine.create_curriculum("Process", "Test explicit evaluator cwd")
    skill = engine.create_skill_hypothesis(curriculum["id"], {"key": "process-cwd", "claim": "Use an explicit evaluator cwd"})
    test = engine.define_test(curriculum["id"], skill["id"], {"phase": "posttest", "cases": ["case"], "evaluator_type": grader.evaluator_type, "evaluator_id": grader.evaluator_id, "success_threshold": 1.0, "evaluator_independence": "independent", "contamination_status": "clean"})
    attempt = engine.record_attempt(test["id"], {"responses": [{"case": "case", "answer": {"artifact": "ok"}}]})
    evaluation = engine.evaluate_trusted_attempt(attempt["id"])
    assert evaluation["passed"] is True
    assert evaluation["provider_provenance"]["working_directory"] == str(tmp_path)
    assert evaluation["provider_provenance"]["validator_sha256"] == hashlib.sha256(validator.read_bytes()).hexdigest()


def test_executable_process_grader_rejects_validator_mutation_after_binding(tmp_path):
    validator = tmp_path / "validator.py"
    validator.write_text(
        "import json, sys\n"
        "payload = json.load(sys.stdin)\n"
        "case = payload['attempt']['responses'][0]['case']\n"
        "print(json.dumps({'case_results': [{'case': case, 'passed': True}]}))\n",
        encoding="utf-8",
    )
    grader = ExecutableProcessGrader(
        evaluator_id="grader.process.immutable",
        command=(sys.executable, str(validator)),
        provider="independent-executable-validator",
        session_id="process-immutable-session",
    )
    validator.write_text(validator.read_text(encoding="utf-8") + "# changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="validator changed after evaluator binding"):
        grader.evaluate(
            {"id": "test", "cases": ["case"], "evaluator_type": grader.evaluator_type, "evaluator_id": grader.evaluator_id, "success_threshold": 1.0},
            {"id": "attempt", "responses": [{"case": "case", "answer": {}}]},
        )


def test_executable_process_grader_fails_closed_on_validator_cardinality(tmp_path):
    validator = tmp_path / "validator.py"
    validator.write_text("import json; print(json.dumps({'case_results': []}))\n", encoding="utf-8")
    grader = ExecutableProcessGrader(
        evaluator_id="grader.process.bad",
        command=(sys.executable, str(validator)),
        provider="independent-executable-validator",
        session_id="process-session",
    )
    engine = LearningEngine(LearningStore(tmp_path / "learning"), trusted_evaluators=cast(dict, {grader.evaluator_id: grader}))
    curriculum = engine.create_curriculum("Process", "Test fail closed")
    skill = engine.create_skill_hypothesis(curriculum["id"], {"key": "process", "claim": "Use a process validator"})
    test = engine.define_test(curriculum["id"], skill["id"], {"phase": "posttest", "cases": ["case"], "evaluator_type": grader.evaluator_type, "evaluator_id": grader.evaluator_id, "success_threshold": 1.0, "evaluator_independence": "independent", "contamination_status": "clean"})
    attempt = engine.record_attempt(test["id"], {"responses": [{"case": "case", "answer": "ok"}]})
    with pytest.raises(LearningError, match="trusted evaluator rejected attempt"):
        engine.evaluate_trusted_attempt(attempt["id"])


def test_structured_behavior_grader_checks_artifact_fields_without_lexical_answer_key(tmp_path):
    grader = StructuredBehaviorGrader(
        evaluator_id="grader.structured.v1",
        required_fields={"case": ["questions", "retrieval_result", "application"]},
        provider="independent-structured-check",
        session_id="structured-session",
    )
    engine = LearningEngine(LearningStore(tmp_path / "learning"), trusted_evaluators=cast(dict, {grader.evaluator_id: grader}))
    curriculum = engine.create_curriculum("Structured", "Test behavioral artifacts")
    skill = engine.create_skill_hypothesis(curriculum["id"], {"key": "structured", "claim": "Produce a study artifact", "applicability": ["study"]})
    test = engine.define_test(curriculum["id"], skill["id"], {"phase": "posttest", "cases": ["case"], "evaluator_type": grader.evaluator_type, "evaluator_id": grader.evaluator_id, "success_threshold": 1.0, "evaluator_independence": "independent", "contamination_status": "clean"})
    attempt = engine.record_attempt(test["id"], {"responses": [{"case": "case", "answer": '{"questions": ["q"], "retrieval_result": "answer", "application": "example"}'}]})
    evaluation = engine.evaluate_trusted_attempt(attempt["id"])
    assert evaluation["passed"] is True
    assert evaluation["evaluator_type"] == "structured_behavior_grader"
    assert evaluation["provider_provenance"]["required_fields"] == {"case": ["questions", "retrieval_result", "application"]}

    failed = engine.record_attempt(test["id"], {"responses": [{"case": "case", "answer": '{"questions": ["q"], "retrieval_result": "answer"}'}]})
    failed_evaluation = engine.evaluate_trusted_attempt(failed["id"])
    assert failed_evaluation["passed"] is False
    assert failed_evaluation["case_results"][0]["missing_fields"] == ["application"]


def test_executable_artifact_grader_validates_fields_and_fails_closed(tmp_path):
    def nonempty_questions(value):
        return isinstance(value, list) and len(value) >= 2 and all(isinstance(item, str) and "?" in item for item in value)

    def concrete_application(value):
        return isinstance(value, str) and "specific" not in value.casefold() and len(value.split()) >= 4

    grader = ExecutableArtifactGrader(
        evaluator_id="grader.executable-artifact.v1",
        required_fields={"case": ["questions", "application"]},
        validators={"case": {"questions": nonempty_questions, "application": concrete_application}},
        provider="independent-executable-artifact",
        session_id="artifact-session",
    )
    engine = LearningEngine(LearningStore(tmp_path / "learning"), trusted_evaluators=cast(dict, {grader.evaluator_id: grader}))
    curriculum = engine.create_curriculum("Executable artifact", "Test executable behavior")
    skill = engine.create_skill_hypothesis(curriculum["id"], {"key": "artifact", "claim": "Produce a concrete artifact", "applicability": ["study"]})
    test = engine.define_test(curriculum["id"], skill["id"], {"phase": "posttest", "cases": ["case"], "evaluator_type": grader.evaluator_type, "evaluator_id": grader.evaluator_id, "success_threshold": 1.0, "evaluator_independence": "independent", "contamination_status": "clean"})
    passing = engine.record_attempt(test["id"], {"responses": [{"case": "case", "answer": {"questions": ["What changed?", "How will I verify it?"], "application": "Apply the procedure to a new task and verify the result."}}]})
    evaluation = engine.evaluate_trusted_attempt(passing["id"])
    assert evaluation["passed"] is True
    assert evaluation["evaluator_type"] == "executable_artifact_grader"
    assert evaluation["provider_provenance"]["validator_names"]["case"] == {"questions": "nonempty_questions", "application": "concrete_application"}

    failing = engine.record_attempt(test["id"], {"responses": [{"case": "case", "answer": '{"questions": ["generic"], "application": "[specific scenario]"}'}]})
    failed_evaluation = engine.evaluate_trusted_attempt(failing["id"])
    assert failed_evaluation["passed"] is False
    assert failed_evaluation["case_results"][0]["failed_fields"] == ["questions", "application"]

    malformed = engine.record_attempt(test["id"], {"responses": [{"case": "case", "answer": "not-json"}]})
    malformed_evaluation = engine.evaluate_trusted_attempt(malformed["id"])
    assert malformed_evaluation["passed"] is False
    assert malformed_evaluation["case_results"][0]["failed_fields"] == ["questions", "application"]


def test_executable_artifact_grader_rejects_empty_or_unconfigured_cases():
    grader = ExecutableArtifactGrader(
        evaluator_id="grader.executable-artifact.validation.v1",
        required_fields={"known": ["artifact"]},
        validators={"known": {"artifact": lambda value: True}},
        provider="independent-executable-artifact",
        session_id="artifact-validation-session",
    )

    empty_test = {"evaluator_type": grader.evaluator_type, "evaluator_id": grader.evaluator_id, "cases": [], "success_threshold": 1.0}
    with pytest.raises(ValueError, match="frozen cases"):
        grader.evaluate(empty_test, {"responses": []})

    unconfigured_test = {"evaluator_type": grader.evaluator_type, "evaluator_id": grader.evaluator_id, "cases": ["unknown"], "success_threshold": 1.0}
    with pytest.raises(ValueError, match="required fields"):
        grader.evaluate(unconfigured_test, {"responses": [{"case": "unknown", "answer": {"artifact": "value"}}]})


def test_executable_artifact_grader_rejects_duplicate_frozen_cases():
    grader = ExecutableArtifactGrader(
        evaluator_id="grader.executable-artifact.duplicate-case.v1",
        required_fields={"case": ["artifact"]},
        validators={"case": {"artifact": lambda value: True}},
        provider="independent-executable-artifact",
        session_id="artifact-duplicate-case-session",
    )
    test = {
        "id": "duplicate-case-test",
        "evaluator_type": grader.evaluator_type,
        "evaluator_id": grader.evaluator_id,
        "cases": ["case", "case"],
        "success_threshold": 1.0,
    }
    with pytest.raises(ValueError, match="frozen cases"):
        grader.evaluate(test, {"id": "duplicate-case-attempt", "responses": [{"case": "case", "answer": {"artifact": "value"}}]})
from rex_learning.experiment import run_bootstrap_experiment
from rex_learning.production_experiment import run_production_learning_experiment
from rex_learning.behavioral_experiment import _answer_with_skill, run_behavioral_reuse_experiment
from rex_learning.ingestion import ingest_epub


def _skill(engine: LearningEngine) -> tuple[dict, dict]:
    curriculum = engine.create_curriculum("C", "O")
    skill = engine.create_skill_hypothesis(curriculum["id"], {"key": "k", "kind": "procedure", "claim": "claim", "success_criteria": ["x"]})
    return curriculum, skill


def test_learning_lifecycle_requires_pretest_and_independent_evidence(tmp_path: Path) -> None:
    engine = LearningEngine(LearningStore(tmp_path / "learning"), allow_fixture=True)
    curriculum, skill = _skill(engine)
    pre = engine.define_test(curriculum["id"], skill["id"], {"phase": "pretest", "cases": ["a"], "expected_case_results": [{"case": "a", "passed": True}], "success_threshold": 1.0, "evaluator_type": "deterministic_test", "evaluator_independence": "independent", "contamination_status": "clean"})
    pa = engine.record_attempt(pre["id"], {"score": 0.0, "case_results": [{"case": "a", "passed": False}]})
    engine.evaluate_attempt(pa["id"], {"score": 1.0, "passed": True, "evidence": [{"ref": "fake"}]})
    post = engine.define_test(curriculum["id"], skill["id"], {"phase": "posttest", "cases": ["b"], "expected_case_results": [{"case": "b", "passed": True}], "success_threshold": 1.0, "evaluator_type": "deterministic_test", "evaluator_independence": "independent", "contamination_status": "clean"})
    po = engine.record_attempt(post["id"], {"score": 1.0, "case_results": [{"case": "b", "passed": True}]})
    pe = engine.evaluate_attempt(po["id"], {"score": 1.0, "passed": True, "evidence": [{"ref": "fake"}]})
    promoted = engine.promote_demonstrated(skill["id"], pretest_attempt_id=pa["id"], posttest_evaluation_id=pe["id"])
    assert promoted["state"] == "demonstrated"
    assert promoted["baseline"]["score"] == 0.0
    assert promoted["latest_evaluation"]["score"] == 1.0


def test_contaminated_or_self_evaluated_attempt_cannot_promote(tmp_path: Path) -> None:
    engine = LearningEngine(LearningStore(tmp_path / "learning"))
    curriculum, skill = _skill(engine)
    pre = engine.define_test(curriculum["id"], skill["id"], {"phase": "pretest", "cases": ["a"], "success_threshold": 1.0, "evaluator_type": "self_evaluation", "evaluator_independence": "shared_context"})
    pa = engine.record_attempt(pre["id"], {"score": 0.0})
    engine.evaluate_attempt(pa["id"], {"score": 0.0, "evidence": [{"ref": "a"}]})
    post = engine.define_test(curriculum["id"], skill["id"], {"phase": "posttest", "cases": ["b"], "success_threshold": 1.0, "evaluator_type": "same_model_shared_context", "evaluator_independence": "none"})
    po = engine.record_attempt(post["id"], {"score": 1.0})
    pe = engine.evaluate_attempt(po["id"], {"score": 1.0, "evidence": [{"ref": "b"}], "contamination": {"status": "possible"}})
    with pytest.raises(LearningError):
        engine.promote_demonstrated(skill["id"], pretest_attempt_id=pa["id"], posttest_evaluation_id=pe["id"])


def test_production_engine_rejects_fixture_promotion(tmp_path: Path) -> None:
    engine = LearningEngine(LearningStore(tmp_path / "learning"))
    _, skill = _skill(engine)
    with pytest.raises(LearningError, match="fixture qualification"):
        engine.force_qualified_fixture(skill["id"], baseline=0.0, post=1.0)


def test_frozen_test_cannot_be_redefined_including_evaluator_id(tmp_path: Path) -> None:
    engine = LearningEngine(LearningStore(tmp_path / "learning"))
    curriculum, skill = _skill(engine)
    spec = {"phase": "posttest", "cases": ["x"], "success_threshold": 1.0, "evaluator_type": "executable_grader", "evaluator_id": "grader-a", "evaluator_independence": "independent", "contamination_status": "not_applicable"}
    first = engine.define_test(curriculum["id"], skill["id"], spec)
    assert engine.define_test(curriculum["id"], skill["id"], dict(spec)) == first
    with pytest.raises(LearningError, match="frozen test"):
        engine.define_test(curriculum["id"], skill["id"], {**spec, "evaluator_id": "grader-b"})


def test_deterministic_test_requires_expected_results_for_exact_frozen_cases(tmp_path: Path) -> None:
    engine = LearningEngine(LearningStore(tmp_path / "learning"), allow_fixture=True)
    curriculum, skill = _skill(engine)
    with pytest.raises(LearningError, match="expected results"):
        engine.define_test(curriculum["id"], skill["id"], {"phase": "pretest", "cases": ["a", "b"], "expected_case_results": [{"case": "a", "passed": True}], "success_threshold": 1.0, "evaluator_type": "deterministic_test", "evaluator_independence": "independent", "contamination_status": "clean"})
    with pytest.raises(LearningError, match="expected results"):
        engine.define_test(curriculum["id"], skill["id"], {"phase": "posttest", "cases": ["a"], "expected_case_results": [{"case": "a", "passed": "yes"}], "success_threshold": 1.0, "evaluator_type": "deterministic_test", "evaluator_independence": "independent", "contamination_status": "clean"})


def test_demonstrated_skill_discovery_is_selective_and_bounded(tmp_path: Path) -> None:
    engine = LearningEngine(LearningStore(tmp_path / "learning"), allow_fixture=True)
    curriculum, skill = _skill(engine)
    skill["applicability"] = ["independent reason for change classification"]
    skill["operational_procedure"] = "Inspect the independent reason for change; classify behavior separately from deployment topology."
    engine.store.write("skills", skill["id"], skill)
    engine.force_qualified_fixture(skill["id"], baseline=0.0, post=1.0)

    positive = engine.discover_applicable_skills({"task_id": "later-1", "requirements": ["independent reason for change classification"]})
    assert positive[0]["skill_id"] == skill["id"]
    assert positive[0]["skill_version"] == 1
    assert "operational_context" in positive[0]
    negative = engine.discover_applicable_skills({"task_id": "later-2", "requirements": ["deployment topology planning"]})
    assert negative == []
    assert len(json.dumps(positive)) < 1800


def test_runtime_discovery_accepts_compact_distinctive_requirement(tmp_path: Path) -> None:
    engine = LearningEngine(LearningStore(tmp_path / "learning"), allow_fixture=True)
    curriculum, skill = _skill(engine)
    skill["key"] = "unit_test_isolation"
    skill["applicability"] = [
        "Design and implement unit tests that verify a specific class in isolation from integration infrastructure."
    ]
    engine.store.write("skills", skill["id"], skill)
    engine.force_qualified_fixture(skill["id"], baseline=0.0, post=1.0)

    selected = engine.discover_applicable_skills({"requirements": ["unit test isolation"]})

    assert [item["key"] for item in selected] == ["unit_test_isolation"]
    assert engine.discover_applicable_skills({"requirements": ["deployment"]}) == []


def test_application_evidence_is_durable_and_attached_to_skill(tmp_path: Path) -> None:
    engine = LearningEngine(LearningStore(tmp_path / "learning"), allow_fixture=True)
    _, skill = _skill(engine)
    engine.force_qualified_fixture(skill["id"], baseline=0.0, post=1.0)
    discovery = engine.discover_applicable_skills({"task_id": "later-1", "requirements": ["claim"]})
    application = engine.apply_skill(discovery[0]["key"], "later-1", discovery[0]["operational_context"]["procedure"], outcome={"score": 1.0}, selection_reason=discovery[0]["why_selected"], operational_context=discovery[0]["operational_context"], execution_ref="execution-1", evaluator_ref="evaluation-1")
    assert application["skill_version"] == 1
    assert application["selection_reason"]
    assert application["execution_ref"] == "execution-1"
    assert application["evaluator_ref"] == "evaluation-1"
    assert application["validation_status"] == "unverified"
    assert application["id"] in engine.skill(skill["id"])["provenance"]["application_ids"]


def test_behavioral_reuse_experiment_proves_fresh_selection_and_control(tmp_path: Path) -> None:
    result = run_behavioral_reuse_experiment(tmp_path / "behavioral")
    assert result["persistence"]["fresh_context"] is True
    assert result["later_task"]["selected_skill_id"] == result["skill"]["id"]
    assert result["later_task"]["application"]["skill_version"] == result["skill"]["version"]
    assert result["ablation"]["with_skill"]["score"] > result["ablation"]["without_skill"]["score"]
    assert result["applicability"]["negative_selected"] is False
    assert result["applicability"]["transfer_selected"] is True


def test_application_evidence_rejects_mismatched_attempt_task(tmp_path: Path) -> None:
    root = tmp_path / "behavioral"
    result = run_behavioral_reuse_experiment(root)
    store = LearningStore(root / "learning")
    engine = LearningEngine(store, trusted_evaluators={"grader.behavioral-reuse.v1": ExecutableGrader(evaluator_id="grader.behavioral-reuse.v1", answer_key={"later-positive": "domain"}, provider="local-executable-grader", session_id="behavioral-reuse-grader-1")})
    application = result["later_task"]["application"]
    attempt = store.read("attempts", result["later_task"]["evaluation"]["attempt_id"])
    attempt["task_id"] = "wrong-task"
    store.write("attempts", attempt["id"], attempt)
    with pytest.raises(LearningError, match="matching trusted evaluation"):
        engine.attach_application_evidence(application["id"], execution_ref=attempt["id"], evaluation_id=result["later_task"]["evaluation"]["id"], outcome={})


def test_behavioral_procedure_controls_later_answer_without_hidden_signal() -> None:
    task = {"facts": {"reason": "domain behavior changed", "location": "unchanged"}}
    procedure = "Inspect the independent reason for change. Classify domain behavior as domain and deployment-only topology as deployment."
    assert _answer_with_skill(task, procedure) == "domain"
    assert _answer_with_skill(task, "Classify by deployment location only.") == "deployment"


def test_epub_ingestion_preserves_source_sections_notes_and_candidate_refs(tmp_path: Path) -> None:
    epub = tmp_path / "lesson.epub"
    with zipfile.ZipFile(epub, "w") as book:
        book.writestr("mimetype", "application/epub+zip")
        book.writestr("OEBPS/chapter.xhtml", "<h1>Procedure</h1><p>Classify the change by its independent reason. The procedure must inspect ambiguous cases.</p>")
    engine = LearningEngine(LearningStore(tmp_path / "learning"))
    curriculum = engine.create_curriculum("EPUB", "Ingest an instructional document")
    result = ingest_epub(engine, curriculum["id"], epub)
    assert result["source"]["kind"] == "epub"
    assert result["sections"][0]["heading"] == "Procedure"
    assert result["study_note_ids"]
    assert result["candidate_skills"][0]["source_ref"].startswith("OEBPS/chapter.xhtml#Procedure")


def test_source_locator_reuse_rejects_changed_content_hash(tmp_path: Path) -> None:
    engine = LearningEngine(LearningStore(tmp_path / "learning"))
    curriculum = engine.create_curriculum("HASH", "Reject source replacement")
    engine.add_source(curriculum["id"], "Lesson", "lesson.epub", kind="epub", content_hash="hash-a")
    with pytest.raises(LearningError, match="content hash"):
        engine.add_source(curriculum["id"], "Lesson", "lesson.epub", kind="epub", content_hash="hash-b")


def test_knowledge_and_study_note_ownership_are_preserved(tmp_path: Path) -> None:
    engine = LearningEngine(LearningStore(tmp_path / "learning"))
    first, _ = _skill(engine)
    second = engine.create_curriculum("second", "two")
    source = engine.add_source(first["id"], "S", "fixture://s")
    with pytest.raises(LearningError):
        engine.record_study_note(second["id"], source["id"], {"claim": "wrong owner"})
    a = engine.record_knowledge(first["id"], {"claim": "avoid premature abstraction", "source_id": "s1"})
    duplicate = engine.record_knowledge(first["id"], {"claim": "avoid premature abstraction", "source_id": "s2"})
    assert duplicate["relation"] == "duplicate"
    assert engine.knowledge(a["id"])["provenance"]


def test_application_evidence_requires_matching_trusted_evaluation(tmp_path: Path) -> None:
    engine, skill, tests, _ = _production_engine(tmp_path)
    pre_attempt = engine.record_attempt(tests["pre"]["id"], {"responses": [{"case": "pre", "answer": "right"}]})
    pre_eval = engine.evaluate_trusted_attempt(pre_attempt["id"])
    post_attempt = engine.record_attempt(tests["post"]["id"], {"responses": [{"case": "post-a", "answer": "right"}, {"case": "post-b", "answer": "right"}]})
    post_eval = engine.evaluate_trusted_attempt(post_attempt["id"])
    engine.promote_demonstrated(skill["id"], pretest_attempt_id=pre_attempt["id"], posttest_evaluation_id=post_eval["id"], pretest_evaluation_id=pre_eval["id"])
    application = engine.apply_skill("k", "later-evidence", "procedure", outcome={"result": "done"})
    with pytest.raises(LearningError, match="matching trusted evaluation"):
        engine.attach_application_evidence(application["id"], execution_ref="execution", evaluation_id="missing", outcome={})
    verified = engine.attach_application_evidence(application["id"], execution_ref=post_attempt["id"], evaluation_id=post_eval["id"], outcome={"result": "done"})
    assert verified["validation_status"] == "verified"
    assert verified["evaluator_ref"] == post_eval["id"]


def test_skill_application_challenge_and_bounded_context(tmp_path: Path) -> None:
    engine = LearningEngine(LearningStore(tmp_path / "learning"), allow_fixture=True)
    _, skill = _skill(engine)
    engine.force_qualified_fixture(skill["id"], baseline=0.0, post=1.0)
    applied = engine.apply_skill("k", "task-1", "procedure", outcome={"score": 1.0})
    assert applied["skill_version"] == 1
    assert engine.apply_skill("k", "task-1", "procedure", outcome={"score": 1.0})["id"] == applied["id"]
    assert engine.challenge_skill(skill["id"], "task-2", {"score": 0.0})["state"] == "challenged"
    assert len(json.dumps(engine.prepared_context(max_chars=1200))) <= 1200


def test_bootstrap_experiment_records_before_after_and_reuse(tmp_path: Path) -> None:
    result = run_bootstrap_experiment(tmp_path / "experiment")
    assert result["qualification_scope"] == "deterministic_framework_fixture_not_provider_cognition"
    assert result["method_A"]["score"] == pytest.approx(1 / 3)
    assert result["held_out_retest"]["score"] == 1.0
    assert result["later_application"]["skill_version"] == result["skill"]["version"]


def _production_engine(tmp_path: Path) -> tuple[LearningEngine, dict, dict, ExecutableGrader]:
    grader = ExecutableGrader("grader.v1", {"pre": "wrong", "post-a": "right", "post-b": "right"}, "grader-provider", "grader-session")
    engine = LearningEngine(LearningStore(tmp_path / "learning"), trusted_evaluators=cast(dict, {grader.evaluator_id: grader}))
    curriculum, skill = _skill(engine)
    pre = engine.define_test(curriculum["id"], skill["id"], {"phase": "pretest", "cases": ["pre"], "success_threshold": 1.0, "evaluator_type": "executable_grader", "evaluator_id": grader.evaluator_id, "evaluator_independence": "independent", "contamination_status": "not_applicable"})
    post = engine.define_test(curriculum["id"], skill["id"], {"phase": "retest", "cases": ["post-a", "post-b"], "success_threshold": 1.0, "evaluator_type": "executable_grader", "evaluator_id": grader.evaluator_id, "evaluator_independence": "independent", "contamination_status": "not_applicable"})
    return engine, skill, {"pre": pre, "post": post}, grader


def test_trusted_executable_grader_is_the_only_production_promotion_path(tmp_path: Path) -> None:
    engine, skill, tests, _ = _production_engine(tmp_path)
    pre_attempt = engine.record_attempt(tests["pre"]["id"], {"responses": [{"case": "pre", "answer": "right"}]})
    with pytest.raises(LearningError, match="trusted evaluator"):
        engine.evaluate_attempt(pre_attempt["id"], {"score": 1.0, "passed": True, "evidence": [{"ref": "fake"}]})
    pre_eval = engine.evaluate_trusted_attempt(pre_attempt["id"])
    assert pre_eval["score"] == 0.0
    post_attempt = engine.record_attempt(tests["post"]["id"], {"responses": [{"case": "post-a", "answer": "right"}, {"case": "post-b", "answer": "right"}]})
    post_eval = engine.evaluate_trusted_attempt(post_attempt["id"])
    assert engine.promote_demonstrated(skill["id"], pretest_attempt_id=pre_attempt["id"], posttest_evaluation_id=post_eval["id"], pretest_evaluation_id=pre_eval["id"])["state"] == "demonstrated"


def test_tampered_trusted_evaluation_cannot_unlock_promotion(tmp_path: Path) -> None:
    engine, skill, tests, _ = _production_engine(tmp_path)
    pre_attempt = engine.record_attempt(tests["pre"]["id"], {"responses": [{"case": "pre", "answer": "right"}]})
    pre_eval = engine.evaluate_trusted_attempt(pre_attempt["id"])
    post_attempt = engine.record_attempt(tests["post"]["id"], {"responses": [{"case": "post-a", "answer": "right"}, {"case": "post-b", "answer": "right"}]})
    post_eval = engine.evaluate_trusted_attempt(post_attempt["id"])
    post_eval["passed"] = False
    post_eval["score"] = 0.0
    engine.store.write("evaluations", post_eval["id"], post_eval)
    with pytest.raises(LearningError, match="canonical"):
        engine.promote_demonstrated(skill["id"], pretest_attempt_id=pre_attempt["id"], posttest_evaluation_id=post_eval["id"], pretest_evaluation_id=pre_eval["id"])


def test_semantic_evaluator_rationale_drift_does_not_change_judgment(tmp_path: Path) -> None:
    engine, skill, tests, _ = _production_engine(tmp_path)
    pre_attempt = engine.record_attempt(tests["pre"]["id"], {"responses": [{"case": "pre", "answer": "right"}]})
    pre_eval = engine.evaluate_trusted_attempt(pre_attempt["id"])
    post_attempt = engine.record_attempt(tests["post"]["id"], {"responses": [{"case": "post-a", "answer": "right"}, {"case": "post-b", "answer": "right"}]})
    post_eval = engine.evaluate_trusted_attempt(post_attempt["id"])
    original = engine.evaluate_trusted_attempt

    def re_evaluate_with_different_rationale(attempt_id: str) -> dict:
        result = original(attempt_id)
        result["case_results"] = [
            {**case, "rationale": "Equivalent independently generated explanation."}
            for case in result["case_results"]
        ]
        return result

    engine.evaluate_trusted_attempt = re_evaluate_with_different_rationale
    promoted = engine.promote_demonstrated(
        skill["id"],
        pretest_attempt_id=pre_attempt["id"],
        posttest_evaluation_id=post_eval["id"],
        pretest_evaluation_id=pre_eval["id"],
    )
    assert promoted["state"] == "demonstrated"


def test_application_evidence_accepts_semantically_identical_rationale_drift(tmp_path: Path) -> None:
    engine, skill, tests, _ = _production_engine(tmp_path)
    pre_attempt = engine.record_attempt(tests["pre"]["id"], {"responses": [{"case": "pre", "answer": "right"}]})
    pre_eval = engine.evaluate_trusted_attempt(pre_attempt["id"])
    post_attempt = engine.record_attempt(tests["post"]["id"], {"responses": [{"case": "post-a", "answer": "right"}, {"case": "post-b", "answer": "right"}]})
    post_eval = engine.evaluate_trusted_attempt(post_attempt["id"])
    engine.promote_demonstrated(
        skill["id"],
        pretest_attempt_id=pre_attempt["id"],
        posttest_evaluation_id=post_eval["id"],
        pretest_evaluation_id=pre_eval["id"],
    )
    application = engine.apply_skill("k", "later-evidence", "procedure", outcome={"result": "done"})
    original = engine.evaluate_trusted_attempt

    def re_evaluate_with_different_rationale(attempt_id: str) -> dict:
        result = original(attempt_id)
        result["case_results"] = [
            {**case, "rationale": "Equivalent independently generated explanation."}
            for case in result["case_results"]
        ]
        return result

    engine.evaluate_trusted_attempt = re_evaluate_with_different_rationale
    attached = engine.attach_application_evidence(
        application["id"],
        execution_ref=post_attempt["id"],
        evaluation_id=post_eval["id"],
        outcome={"result": "done"},
    )
    assert attached["validation_status"] == "verified"


def test_trusted_grader_rejects_missing_duplicate_and_unknown_cases(tmp_path: Path) -> None:
    engine, _, tests, _ = _production_engine(tmp_path)
    with pytest.raises(LearningError, match="exactly one"):
        engine.record_attempt(tests["post"]["id"], {"responses": [{"case": "post-a", "answer": "right"}]})
    with pytest.raises(LearningError, match="exactly one"):
        engine.record_attempt(tests["post"]["id"], {"responses": [{"case": "post-a", "answer": "right"}, {"case": "post-a", "answer": "right"}]})


def test_trusted_evaluator_registry_and_answer_key_are_not_mutable(tmp_path: Path) -> None:
    engine, _, _, grader = _production_engine(tmp_path)
    with pytest.raises(TypeError):
        engine.trusted_evaluators[grader.evaluator_id] = grader
    with pytest.raises(TypeError):
        grader.answer_key["post-a"] = "forged"


def test_trusted_evaluator_registry_binds_identity_and_provenance(tmp_path: Path) -> None:
    mismatched = ExecutableGrader("actual-id", {"x": "ok"}, "provider", "session")
    with pytest.raises(LearningError, match="registry binding"):
        LearningEngine(LearningStore(tmp_path / "mismatch"), trusted_evaluators=cast(dict, {"registered-id": mismatched}))

    class ForgedProvenanceGrader(ExecutableGrader):
        def evaluate(self, test: dict, attempt: dict) -> dict:
            result = super().evaluate(test, attempt)
            result["provider_provenance"] = {"provider": "forged", "session_id": "forged", "evaluator_id": self.evaluator_id}
            return result

    grader = ForgedProvenanceGrader("bound-id", {"x": "ok"}, "provider", "session")
    engine = LearningEngine(LearningStore(tmp_path / "forged"), trusted_evaluators=cast(dict, {grader.evaluator_id: grader}))
    curriculum, skill = _skill(engine)
    test = engine.define_test(curriculum["id"], skill["id"], {"phase": "pretest", "cases": ["x"], "success_threshold": 1.0, "evaluator_type": "executable_grader", "evaluator_id": grader.evaluator_id, "evaluator_independence": "independent", "contamination_status": "not_applicable"})
    attempt = engine.record_attempt(test["id"], {"responses": [{"case": "x", "answer": "ok"}]})
    with pytest.raises(LearningError, match="provenance"):
        engine.evaluate_trusted_attempt(attempt["id"])

    class Impostor:
        evaluator_id = "impostor-id"
        evaluator_type = "executable_grader"
        provider = "provider"
        session_id = "session"

        def evaluate(self, test: dict, attempt: dict) -> dict:
            return {}

    with pytest.raises(LearningError, match="registry binding"):
        LearningEngine(LearningStore(tmp_path / "impostor"), trusted_evaluators=cast(dict, {"impostor-id": Impostor()}))


def test_non_json_serializable_trusted_output_is_learning_error(tmp_path: Path) -> None:
    class ExtraOutputGrader(ExecutableGrader):
        def evaluate(self, test: dict, attempt: dict) -> dict:
            result = super().evaluate(test, attempt)
            result["extra"] = object()
            return result

    grader = ExtraOutputGrader("extra-id", {"x": "ok"}, "provider", "session")
    engine = LearningEngine(LearningStore(tmp_path / "extra"), trusted_evaluators=cast(dict, {grader.evaluator_id: grader}))
    curriculum, skill = _skill(engine)
    test = engine.define_test(curriculum["id"], skill["id"], {"phase": "pretest", "cases": ["x"], "success_threshold": 1.0, "evaluator_type": "executable_grader", "evaluator_id": grader.evaluator_id, "evaluator_independence": "independent", "contamination_status": "not_applicable"})
    attempt = engine.record_attempt(test["id"], {"responses": [{"case": "x", "answer": "ok"}]})
    with pytest.raises(LearningError, match="JSON-serializable"):
        engine.evaluate_trusted_attempt(attempt["id"])


def test_tampered_persisted_attempt_cannot_be_reevaluated(tmp_path: Path) -> None:
    engine, _, tests, _ = _production_engine(tmp_path)
    attempt = engine.record_attempt(tests["pre"]["id"], {"responses": [{"case": "pre", "answer": "right"}]})
    attempt["responses"][0]["answer"] = "wrong"
    engine.store.write("attempts", attempt["id"], attempt)
    with pytest.raises(LearningError, match="integrity"):
        engine.evaluate_trusted_attempt(attempt["id"])


def test_promotion_rejects_cross_skill_test_evidence(tmp_path: Path) -> None:
    engine, skill, tests, grader = _production_engine(tmp_path)
    other_curriculum = engine.create_curriculum("other", "other")
    other_skill = engine.create_skill_hypothesis(other_curriculum["id"], {"key": "other", "claim": "other"})
    other_test = engine.define_test(other_curriculum["id"], other_skill["id"], {"phase": "retest", "cases": ["post-a", "post-b"], "success_threshold": 1.0, "evaluator_type": "executable_grader", "evaluator_id": grader.evaluator_id, "evaluator_independence": "independent", "contamination_status": "not_applicable"})
    pre = engine.record_attempt(tests["pre"]["id"], {"responses": [{"case": "pre", "answer": "right"}]})
    pre_eval = engine.evaluate_trusted_attempt(pre["id"])
    post = engine.record_attempt(other_test["id"], {"responses": [{"case": "post-a", "answer": "right"}, {"case": "post-b", "answer": "right"}]})
    post_eval = engine.evaluate_trusted_attempt(post["id"])
    with pytest.raises(LearningError, match="skill evidence"):
        engine.promote_demonstrated(skill["id"], pretest_attempt_id=pre["id"], posttest_evaluation_id=post_eval["id"], pretest_evaluation_id=pre_eval["id"])


def test_application_requires_curriculum_for_ambiguous_skill_key(tmp_path: Path) -> None:
    engine = LearningEngine(LearningStore(tmp_path / "learning"), allow_fixture=True)
    curriculum, skill = _skill(engine)
    engine.force_qualified_fixture(skill["id"], baseline=0.0, post=1.0)
    other = engine.create_curriculum("other", "other")
    other_skill = engine.create_skill_hypothesis(other["id"], {"key": "k", "claim": "other claim"})
    engine.force_qualified_fixture(other_skill["id"], baseline=0.0, post=1.0)
    with pytest.raises(LearningError, match="curriculum_id"):
        engine.apply_skill("k", "task", "procedure", outcome={"score": 1.0})
    applied = engine.apply_skill("k", "task", "procedure", curriculum_id=other["id"], outcome={"score": 1.0})
    assert applied["skill_id"] == other_skill["id"]


def test_malformed_trusted_evaluator_output_is_learning_error(tmp_path: Path) -> None:
    class MalformedGrader(ExecutableGrader):
        def evaluate(self, test: dict, attempt: dict) -> dict:
            return {"case_results": [None]}

    grader = MalformedGrader("malformed.v1", {"pre": "right"}, "provider", "session")
    engine = LearningEngine(LearningStore(tmp_path / "learning"), trusted_evaluators=cast(dict, {grader.evaluator_id: grader}))
    curriculum, skill = _skill(engine)
    test = engine.define_test(curriculum["id"], skill["id"], {"phase": "pretest", "cases": ["pre"], "success_threshold": 1.0, "evaluator_type": "executable_grader", "evaluator_id": grader.evaluator_id, "evaluator_independence": "independent", "contamination_status": "not_applicable"})
    attempt = engine.record_attempt(test["id"], {"responses": [{"case": "pre", "answer": "right"}]})
    with pytest.raises(LearningError, match="trusted evaluator"):
        engine.evaluate_trusted_attempt(attempt["id"])


def test_revision_rejects_unknown_or_cross_skill_evidence(tmp_path: Path) -> None:
    engine, skill, tests, _ = _production_engine(tmp_path)
    with pytest.raises(LearningError, match="existing record"):
        engine.revise_skill(skill["id"], {"claim": "revised"}, reason="because", evidence_ids=["does-not-exist"])
    other_curriculum = engine.create_curriculum("other", "other objective")
    other_skill = engine.create_skill_hypothesis(other_curriculum["id"], {"key": "other", "kind": "procedure", "claim": "other claim"})
    other_test = engine.define_test(other_curriculum["id"], other_skill["id"], {"phase": "pretest", "cases": ["other"], "success_threshold": 1.0, "evaluator_type": "executable_grader", "evaluator_id": "grader.v1", "evaluator_independence": "independent", "contamination_status": "not_applicable"})
    other_attempt = engine.record_attempt(other_test["id"], {"responses": [{"case": "other", "answer": "right"}]})
    with pytest.raises(LearningError, match="does not belong"):
        engine.revise_skill(skill["id"], {"claim": "revised"}, reason="because", evidence_ids=[other_attempt["id"]])


def test_production_promotion_requires_improvement_and_applications_are_unverified(tmp_path: Path) -> None:
    engine, skill, tests, _ = _production_engine(tmp_path)
    pre_attempt = engine.record_attempt(tests["pre"]["id"], {"responses": [{"case": "pre", "answer": "right"}]})
    pre_eval = engine.evaluate_trusted_attempt(pre_attempt["id"])
    post_attempt = engine.record_attempt(tests["post"]["id"], {"responses": [{"case": "post-a", "answer": "right"}, {"case": "post-b", "answer": "right"}]})
    post_eval = engine.evaluate_trusted_attempt(post_attempt["id"])
    assert engine.promote_demonstrated(skill["id"], pretest_attempt_id=pre_attempt["id"], posttest_evaluation_id=post_eval["id"], pretest_evaluation_id=pre_eval["id"])["state"] == "demonstrated"
    application = engine.apply_skill("k", "task-1", "procedure", outcome={"score": 1.0})
    assert application["outcome"]["validation_status"] == "unverified"
    forged = engine.apply_skill("k", "task-2", "procedure", outcome={"score": 1.0, "validation_status": "verified"})
    assert forged["outcome"]["validation_status"] == "unverified"
    assert application["outcome"]["validation_status"] == "unverified"


def test_production_promotion_rejects_no_change_from_ceiling_baseline(tmp_path: Path) -> None:
    engine, skill, tests, _ = _production_engine(tmp_path)
    pre_attempt = engine.record_attempt(tests["pre"]["id"], {"responses": [{"case": "pre", "answer": "wrong"}]})
    pre_eval = engine.evaluate_trusted_attempt(pre_attempt["id"])
    post_attempt = engine.record_attempt(tests["post"]["id"], {"responses": [{"case": "post-a", "answer": "right"}, {"case": "post-b", "answer": "right"}]})
    post_eval = engine.evaluate_trusted_attempt(post_attempt["id"])
    with pytest.raises(LearningError, match="improvement"):
        engine.promote_demonstrated(skill["id"], pretest_attempt_id=pre_attempt["id"], posttest_evaluation_id=post_eval["id"], pretest_evaluation_id=pre_eval["id"])


def test_production_experiment_contract(tmp_path: Path) -> None:
    result = run_production_learning_experiment(tmp_path / "provider")
    assert result["baseline"]["score"] == 0.0
    assert result["post_study_attempt"]["score"] == 0.5
    assert result["held_out_retest"]["score"] == 1.0
    assert result["later_application"]["outcome"]["score"] == 1.0
    assert result["skill"]["state"] == "demonstrated"


def test_openai_compatible_learner_requires_structured_provider_output() -> None:
    from rex_learning import LearnerError, OpenAICompatibleLearner

    def requester(url: str, body: bytes, headers: dict[str, str], timeout: float) -> bytes:
        assert url.endswith("/v1/chat/completions")
        request = json.loads(body)
        assert request["response_format"] == {"type": "json_object"}
        system = request["messages"][0]["content"]
        assert "never declare, redefine, reimplement, stub, or mock their type definitions" in system
        assert "verify the exact observable output for each required input" in system
        assert "requested case identifier" in system
        assert "complete payload under the single answer key" in system
        return json.dumps({"choices": [{"message": {"content": '{"responses": [{"case": "c1", "answer": "ravel"}], "procedure": "rule"}'}}]}).encode()

    learner = OpenAICompatibleLearner(endpoint="http://provider", model="model", requester=requester)
    assert learner.answer(task="classify")["procedure"] == "rule"

    def malformed(url: str, body: bytes, headers: dict[str, str], timeout: float) -> bytes:
        return json.dumps({"choices": [{"message": {"content": "not json"}}]}).encode()

    with pytest.raises(LearnerError, match="JSON learner response") as malformed_error:
        OpenAICompatibleLearner(endpoint="http://provider", model="model", requester=malformed).answer(task="classify")
    assert '"content": "not json"' in malformed_error.value.diagnostic()["raw_response"]
    assert malformed_error.value.diagnostic()["raw_response_truncated"] is False

    def missing_responses(url: str, body: bytes, headers: dict[str, str], timeout: float) -> bytes:
        return json.dumps({"choices": [{"message": {"content": '{"answer": "prose"}'}}]}).encode()

    with pytest.raises(LearnerError) as error:
        OpenAICompatibleLearner(endpoint="http://provider", model="model", requester=missing_responses).answer(task="classify")
    assert error.value.code == "missing_responses_list"
    assert error.value.diagnostic()["actual_type"] == "NoneType"
    assert error.value.diagnostic()["parsed_keys"] == ["answer"]
    assert '"answer": "prose"' in error.value.diagnostic()["parsed_preview"]

    def artifact_without_envelope(url: str, body: bytes, headers: dict[str, str], timeout: float) -> bytes:
        return json.dumps({"choices": [{"message": {"content": '{"final_status": "complete", "tests": []}'}}]}).encode()

    wrapped = OpenAICompatibleLearner(endpoint="http://provider", model="model", requester=artifact_without_envelope)
    assert wrapped.answer(task="complete|case=artifact-2|return the artifact")["responses"] == [
        {"case": "artifact-2", "answer": {"final_status": "complete", "tests": []}}
    ]

    def fenced_artifact(url: str, body: bytes, headers: dict[str, str], timeout: float) -> bytes:
        return json.dumps({"choices": [{"message": {"content": "```java\nclass Example {}\n```"}}]}).encode()

    fenced = OpenAICompatibleLearner(endpoint="http://provider", model="model", requester=fenced_artifact)
    assert fenced.answer(task="complete|case=artifact-1|return the supplied artifact")["responses"] == [
        {"case": "artifact-1", "answer": "class Example {}"}
    ]

    def double_encoded(url: str, body: bytes, headers: dict[str, str], timeout: float) -> bytes:
        content = json.dumps(json.dumps({"responses": [{"case": "c2", "answer": "unwrapped"}]}))
        return json.dumps({"choices": [{"message": {"content": content}}]}).encode()

    assert OpenAICompatibleLearner(endpoint="http://provider", model="model", requester=double_encoded).answer(task="classify|case=c2")["responses"] == [
        {"case": "c2", "answer": "unwrapped"}
    ]


def test_openai_compatible_learner_parses_fenced_json_objects() -> None:
    from rex_learning import OpenAICompatibleLearner

    def fenced_json(url: str, body: bytes, headers: dict[str, str], timeout: float) -> bytes:
        content = "```json\n{\"responses\":[{\"case\":\"c3\",\"answer\":{\"capabilities\":[{\"key\":\"coverage\"}]}}]}\n```"
        return json.dumps({"choices": [{"message": {"content": content}}]}).encode()

    result = OpenAICompatibleLearner(endpoint="http://provider", model="model", requester=fenced_json).answer(
        task="discover|case=c3|return JSON"
    )

    assert result["responses"] == [
        {"case": "c3", "answer": {"capabilities": [{"key": "coverage"}]}}
    ]


def test_openai_compatible_learner_retries_transport_format_only() -> None:
    from rex_learning import OpenAICompatibleLearner

    responses = iter([
        json.dumps({"choices": [{"message": {"content": "not json"}}]}).encode(),
        json.dumps({"choices": [{"message": {"content": '{"responses": [{"case": "c1", "answer": "recovered"}]}'}}]}).encode(),
    ])
    calls: list[dict] = []

    def requester(url: str, body: bytes, headers: dict[str, str], timeout: float) -> bytes:
        request = json.loads(body)
        calls.append(request)
        return next(responses)

    result = OpenAICompatibleLearner(endpoint="http://provider", model="model", requester=requester).answer(task="classify|case=c1")

    assert result["responses"] == [{"case": "c1", "answer": "recovered"}]
    assert len(calls) == 2
    assert "formatting" in calls[1]["messages"][-1]["content"].lower()
    assert "answer key" not in calls[1]["messages"][-1]["content"].lower()


def test_openai_compatible_learner_does_not_retry_contract_failures() -> None:
    from rex_learning import LearnerError, OpenAICompatibleLearner

    calls = 0

    def requester(url: str, body: bytes, headers: dict[str, str], timeout: float) -> bytes:
        nonlocal calls
        calls += 1
        return json.dumps({"choices": [{"message": {"content": '{"answer": "missing responses"}'}}]}).encode()

    with pytest.raises(LearnerError) as error:
        OpenAICompatibleLearner(endpoint="http://provider", model="model", requester=requester).answer(task="classify")

    assert error.value.code == "missing_responses_list"
    assert calls == 1


def test_openai_compatible_learner_prompt_requires_silent_contract_self_check() -> None:
    from rex_learning import OpenAICompatibleLearner

    captured: dict[str, object] = {}

    def requester(url: str, body: bytes, headers: dict[str, str], timeout: float) -> bytes:
        captured.update(json.loads(body))
        return json.dumps({"choices": [{"message": {"content": '{"responses": [{"case": "c1", "answer": "ravel"}]}'}}]}).encode()

    OpenAICompatibleLearner(endpoint="http://provider", model="model", requester=requester).answer(
        task="Complete the supplied artifact and follow every explicit contract requirement.",
    )
    messages = captured["messages"]
    assert isinstance(messages, list)
    system = messages[0]["content"]
    assert "Before responding, silently self-check every explicit task requirement" in system
    assert "make every stated behavior observable with a direct assertion" in system
    assert "assert the relevant state before and after each invocation" in system
    assert "Do not emit the checklist or contradictory labels" in system


def test_openai_compatible_learner_normalizes_procedure_steps() -> None:
    from rex_learning import OpenAICompatibleLearner

    def requester(url: str, body: bytes, headers: dict[str, str], timeout: float) -> bytes:
        return json.dumps({"choices": [{"message": {"content": json.dumps({"responses": [{"case": "c1", "answer": "ravel"}], "procedure": ["step one", "step two"]})}}]}).encode()

    result = OpenAICompatibleLearner(endpoint="http://provider", model="model", requester=requester).answer(task="classify")
    assert result["procedure"] == "step one\nstep two"
    assert result["_provider_response"]["choices"][0]["message"]["content"]


class _ChangingEvaluator:
    evaluator_id = "changing-evaluator"
    evaluator_type = "changing"

    def __init__(self, results: list[dict]) -> None:
        self.results = iter(results)
        self.calls = 0

    def evaluate(self, test: dict, attempt: dict) -> dict:
        self.calls += 1
        return next(self.results)


def test_stable_trusted_evaluator_allows_rationale_drift() -> None:
    delegate = _ChangingEvaluator([
        {"score": 1.0, "passed": True, "rationale": "first"},
        {"score": 1.0, "passed": True, "rationale": "different wording"},
    ])
    grader = StableTrustedEvaluator(delegate, delegate.evaluator_id, delegate.evaluator_type)
    result = grader.evaluate({}, {})
    assert result["passed"] is True
    assert delegate.calls == 2


def test_stable_trusted_evaluator_allows_explanatory_diagnosis_drift() -> None:
    delegate = _ChangingEvaluator([
        {"score": 1.0, "passed": True, "case_results": [{"case": "c1", "passed": True, "diagnosis": "clear"}]},
        {"score": 1.0, "passed": True, "case_results": [{"case": "c1", "passed": True, "diagnosis": "sufficient"}]},
    ])
    grader = StableTrustedEvaluator(delegate, delegate.evaluator_id, delegate.evaluator_type)
    result = grader.evaluate({}, {})
    assert result["passed"] is True
    assert delegate.calls == 2


def test_stable_trusted_evaluator_ignores_auxiliary_metadata_drift() -> None:
    delegate = _ChangingEvaluator([
        {"score": 0.0, "passed": False, "case_results": [{"case": "c1", "passed": False}], "phase": "baseline"},
        {"score": 0.0, "passed": False, "case_results": [{"case": "c1", "passed": False}], "provider_note": "retry"},
    ])
    grader = StableTrustedEvaluator(delegate, delegate.evaluator_id, delegate.evaluator_type)
    result = grader.evaluate({}, {})
    assert result["passed"] is False
    assert delegate.calls == 2


def test_stable_trusted_evaluator_rejects_substantive_drift() -> None:
    delegate = _ChangingEvaluator([
        {"score": 1.0, "passed": True, "case_results": [{"passed": True}]},
        {"score": 0.0, "passed": False, "case_results": [{"passed": False}]},
    ])
    grader = StableTrustedEvaluator(delegate, delegate.evaluator_id, delegate.evaluator_type)
    with pytest.raises(EvaluatorInstabilityError, match="conflicting substantive"):
        grader.evaluate({}, {})
    assert delegate.calls == 2
