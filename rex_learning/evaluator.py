from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence


class TrustedEvaluator(Protocol):
    evaluator_id: str
    evaluator_type: str
    provider: str
    session_id: str

    def evaluate(self, test: dict[str, Any], attempt: dict[str, Any]) -> dict[str, Any]: ...


def _validate_declared_artifact_shape(test: Mapping[str, Any], attempt: Mapping[str, Any]) -> None:
    """Enforce the independently frozen output shape before semantic grading."""
    raw = test.get("artifact_schema")
    if raw is None:
        raw = test.get("evaluation_contract")
    if raw is None or raw == {}:
        return
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("declared artifact contract is not valid JSON") from exc
    if not isinstance(raw, Mapping):
        raise ValueError("declared artifact contract must be an object")
    required = raw.get("required_fields")
    field_types = raw.get("field_types")
    if not isinstance(required, list) or not isinstance(field_types, Mapping):
        raise ValueError("declared artifact contract is incomplete")

    def type_name(value: Any) -> str:
        if isinstance(value, bool):
            return "boolean"
        if isinstance(value, int):
            return "integer"
        if isinstance(value, float):
            return "number"
        if isinstance(value, str):
            return "string"
        if isinstance(value, list):
            return "array"
        if isinstance(value, Mapping):
            return "object"
        return "null"

    responses = attempt.get("responses")
    if not isinstance(responses, list):
        raise ValueError("attempt responses are required for declared artifact contracts")
    for response in responses:
        if not isinstance(response, Mapping):
            raise ValueError("attempt response is not an object")
        answer = response.get("answer")
        if isinstance(answer, str):
            try:
                answer = json.loads(answer)
            except json.JSONDecodeError:
                pass
        if not isinstance(answer, Mapping):
            raise ValueError("learner artifact does not match its declared object shape")
        for field in required:
            if not isinstance(field, str) or field not in answer:
                raise ValueError(f"learner artifact is missing required field: {field}")
            expected = field_types.get(field)
            actual = type_name(answer[field])
            if expected == "number" and actual == "integer":
                continue
            if actual != expected:
                raise ValueError(f"learner artifact field {field} has type {actual}, expected {expected}")


class EvaluatorInstabilityError(ValueError):
    """Raised when repeated evaluation changes a substantive judgment."""


@dataclass(frozen=True)
class CallbackEvaluator:
    """Provider-neutral adapter for an independently configured evaluator."""

    evaluator_id: str
    callback: Any
    provider: str
    session_id: str
    evaluator_type: str = "callback_evaluator"

    def evaluate(self, test: dict[str, Any], attempt: dict[str, Any]) -> dict[str, Any]:
        if test.get("evaluator_type") != self.evaluator_type or test.get("evaluator_id") != self.evaluator_id:
            raise ValueError("test is not bound to this trusted evaluator")
        cases = list(test.get("cases", []))
        if not cases or len(cases) != len(set(cases)):
            raise ValueError("frozen cases must be nonempty and unique")
        _validate_declared_artifact_shape(test, attempt)
        result = self.callback(test=dict(test), attempt=dict(attempt))
        if not isinstance(result, Mapping):
            raise ValueError("callback evaluator returned a non-object")
        raw_results = result.get("case_results")
        if not isinstance(raw_results, list) or len(raw_results) != len(cases):
            raise ValueError("callback evaluator returned invalid case cardinality")
        by_case = {item.get("case"): item for item in raw_results if isinstance(item, Mapping)}
        if len(by_case) != len(raw_results) or set(by_case) != set(cases):
            raise ValueError("callback evaluator cases do not match frozen cases")
        case_results = []
        for case in cases:
            item = by_case[case]
            if not isinstance(item.get("passed"), bool):
                raise ValueError("callback evaluator judgments must contain boolean passed values")
            case_results.append(dict(item))
        score = sum(item["passed"] for item in case_results) / len(case_results)
        return {
            "schema": "rex-learning-trusted-evaluation-v1",
            "evaluator_id": self.evaluator_id,
            "evaluator_type": self.evaluator_type,
            "test_spec_id": test["id"],
            "attempt_id": attempt["id"],
            "score": score,
            "passed": score >= test["success_threshold"],
            "case_results": case_results,
            "evidence_refs": [f"callback://{self.evaluator_id}/{test['id']}/{attempt['id']}/{case}" for case in cases],
            "contamination": {"status": test.get("contamination_status", "unknown")},
            "evaluator_independence": test.get("evaluator_independence", "unknown"),
            "provider_provenance": {"provider": self.provider, "session_id": self.session_id, "evaluator_id": self.evaluator_id},
        }


@dataclass(frozen=True)
class ExecutableProcessGrader:
    """Run a Hermes-owned validator process against an untrusted attempt.

    The validator is configured outside the learning store and learner prompt.
    It receives only the frozen test and attempt as JSON on stdin and must
    return one ``case_results`` entry per frozen case.  Scores and pass status
    are derived here, not trusted from validator output.  A non-zero exit,
    timeout, malformed JSON, or cardinality mismatch fails closed as an
    evaluator error rather than becoming learning evidence.
    """

    evaluator_id: str
    command: tuple[str, ...]
    provider: str
    session_id: str
    timeout_seconds: float = 30.0
    evaluator_type: str = "executable_process_grader"
    working_directory: str | Path | None = None
    validator_sha256: str | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        command = tuple(str(item) for item in self.command if str(item))
        if not command:
            raise ValueError("executable process command must be nonempty")
        if self.timeout_seconds <= 0:
            raise ValueError("executable process timeout must be positive")
        object.__setattr__(self, "command", command)
        if self.working_directory is not None:
            working_directory = Path(self.working_directory)
            if not working_directory.is_dir():
                raise ValueError("executable process working directory must exist")
            object.__setattr__(self, "working_directory", working_directory)
        validator_path = self._validator_path()
        if validator_path is not None:
            object.__setattr__(self, "validator_sha256", hashlib.sha256(validator_path.read_bytes()).hexdigest())

    def _validator_path(self) -> Path | None:
        for argument in self.command[1:]:
            path = Path(argument)
            if not path.is_absolute() and self.working_directory is not None:
                path = Path(self.working_directory) / path
            if path.is_file():
                return path
        return None

    def evaluate(self, test: dict[str, Any], attempt: dict[str, Any]) -> dict[str, Any]:
        if test.get("evaluator_type") != self.evaluator_type or test.get("evaluator_id") != self.evaluator_id:
            raise ValueError("test is not bound to this trusted evaluator")
        cases = list(test.get("cases", []))
        responses = attempt.get("responses", [])
        if not cases or len(cases) != len(set(cases)) or not isinstance(responses, list):
            raise ValueError("frozen cases and attempt responses must be nonempty")
        actual_cases = [item.get("case") for item in responses if isinstance(item, Mapping)]
        if actual_cases != cases or len(actual_cases) != len(set(actual_cases)):
            raise ValueError("attempt responses must contain exactly the frozen cases")
        payload = json.dumps({"test": test, "attempt": attempt}, ensure_ascii=False, sort_keys=True).encode()
        try:
            completed = subprocess.run(
                self.command,
                input=payload,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.timeout_seconds,
                check=False,
                cwd=self.working_directory,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ValueError("executable validator could not complete") from exc
        if completed.returncode != 0:
            raise ValueError(f"executable validator exited {completed.returncode}")
        try:
            raw = json.loads(completed.stdout.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("executable validator did not return JSON") from exc
        if not isinstance(raw, Mapping) or not isinstance(raw.get("case_results"), list):
            raise ValueError("executable validator output must contain case_results")
        results = list(raw["case_results"])
        if len(results) != len(cases) or any(not isinstance(item, Mapping) for item in results):
            raise ValueError("executable validator case cardinality is invalid")
        if [item.get("case") for item in results] != cases or any(not isinstance(item.get("passed"), bool) for item in results):
            raise ValueError("executable validator cases or pass values are invalid")
        normalized_results = [dict(item) for item in results]
        score = sum(item["passed"] for item in normalized_results) / len(cases)
        validator_path = self._validator_path()
        validator_sha256 = hashlib.sha256(validator_path.read_bytes()).hexdigest() if validator_path is not None else None
        if self.validator_sha256 is not None and validator_sha256 != self.validator_sha256:
            raise ValueError("executable validator changed after evaluator binding")
        return {
            "schema": "rex-learning-trusted-evaluation-v1",
            "evaluator_id": self.evaluator_id,
            "evaluator_type": self.evaluator_type,
            "test_spec_id": test["id"],
            "attempt_id": attempt["id"],
            "score": score,
            "passed": score >= test["success_threshold"],
            "case_results": normalized_results,
            "evidence_refs": [f"executable-process://{self.evaluator_id}/{test['id']}/{attempt['id']}/{case}" for case in cases],
            "contamination": {"status": test.get("contamination_status", "unknown")},
            "evaluator_independence": test.get("evaluator_independence", "unknown"),
            "provider_provenance": {
                "provider": self.provider,
                "session_id": self.session_id,
                "evaluator_id": self.evaluator_id,
                "command": list(self.command),
                "validator_sha256": validator_sha256,
                "validator_stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
                "validator_stdout_bytes": len(completed.stdout),
                "timeout_seconds": self.timeout_seconds,
                "validator_exit_code": completed.returncode,
                "working_directory": str(self.working_directory) if self.working_directory is not None else None,
            },
        }


def _safe_project_path(value: Any) -> str:
    if not isinstance(value, str) or not value or "\x00" in value or Path(value).is_absolute() or "\\\\" in value:
        raise ValueError("project file paths must be safe relative paths")
    path = Path(value)
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("project file paths must not traverse directories")
    return value


def _safe_project_command(command: Any) -> tuple[str, ...]:
    if not isinstance(command, (list, tuple)) or not command or any(not isinstance(item, str) or not item.strip() for item in command):
        raise ValueError("project evaluator command must be a nonempty argument list")
    if any(any(token in item for token in (";", "|", "&", ">", "<", "`", "$()")) for item in command):
        raise ValueError("unsafe project evaluator command")
    return tuple(command)


@dataclass(frozen=True)
class ExecutableProjectContract:
    """Frozen Hermes-owned project material and executable checks."""
    source_files: Mapping[str, str] | Sequence[str]
    support_files: Mapping[str, str]
    test_command: tuple[str, ...]
    provenance: str
    setup_command: tuple[str, ...] | None = None
    timeout_seconds: float = 30.0
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.support_files, Mapping) or not isinstance(self.provenance, str) or not self.provenance.strip():
            raise ValueError("project contract provenance and support files are required")
        if not isinstance(self.source_files, (Mapping, list, tuple)):
            raise ValueError("project source files must be a mapping or sequence")
        source = self.source_files.items() if isinstance(self.source_files, Mapping) else ((item, "") for item in self.source_files)
        normalized_source = {}
        for path, content in source:
            path = _safe_project_path(path)
            if not isinstance(content, str):
                raise ValueError("project fixture contents must be strings")
            normalized_source[path] = content
        if not normalized_source:
            raise ValueError("project contract must declare source files")
        normalized_support = {}
        for path, content in self.support_files.items():
            path = _safe_project_path(path)
            if not isinstance(content, str):
                raise ValueError("support file contents must be strings")
            normalized_support[path] = content
        if set(normalized_source) & set(normalized_support):
            raise ValueError("source and support files must not overlap")
        command = _safe_project_command(self.test_command)
        setup = None if self.setup_command is None else _safe_project_command(self.setup_command)
        if self.timeout_seconds <= 0:
            raise ValueError("project evaluator timeout must be positive")
        object.__setattr__(self, "source_files", MappingProxyType(normalized_source))
        object.__setattr__(self, "support_files", MappingProxyType(normalized_support))
        object.__setattr__(self, "test_command", command)
        object.__setattr__(self, "setup_command", setup)
        object.__setattr__(self, "sha256", hashlib.sha256(json.dumps(self._payload(), sort_keys=True).encode()).hexdigest())

    def _payload(self) -> dict[str, Any]:
        return {"source_files": dict(self.source_files), "support_files": dict(self.support_files), "test_command": list(self.test_command), "setup_command": list(self.setup_command) if self.setup_command else None, "timeout_seconds": self.timeout_seconds, "provenance": self.provenance}

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "sha256": self.sha256}


@dataclass(frozen=True)
class ExecutableProjectEvaluator:
    """Run evaluator-owned compile/test commands against learner project files."""
    evaluator_id: str
    provider: str
    session_id: str
    contract: ExecutableProjectContract
    evaluator_type: str = "executable_project_evaluator"

    def evaluate(self, test: dict[str, Any], attempt: dict[str, Any]) -> dict[str, Any]:
        if test.get("evaluator_type") != self.evaluator_type or test.get("evaluator_id") != self.evaluator_id:
            raise ValueError("test is not bound to this trusted evaluator")
        if test.get("project_contract") != self.contract.to_dict() or test.get("project_contract_sha256") != self.contract.sha256:
            raise ValueError("project contract hash or provenance does not match evaluator binding")
        cases, responses = test.get("cases"), attempt.get("responses")
        if not isinstance(cases, list) or not cases or len(cases) != len(set(cases)) or not isinstance(responses, list) or [r.get("case") for r in responses if isinstance(r, Mapping)] != cases:
            raise ValueError("project attempt cases are malformed")
        results = []
        for response in responses:
            answer = response.get("answer") if isinstance(response, Mapping) else None
            files = answer.get("files") if isinstance(answer, Mapping) else None
            valid = isinstance(files, Mapping) and all(isinstance(path, str) and isinstance(content, str) for path, content in files.items())
            if valid:
                try:
                    normalized = {_safe_project_path(path): content for path, content in files.items()}
                except ValueError:
                    valid = False
                else:
                    if set(normalized) & set(self.contract.support_files):
                        raise ValueError("learner artifact cannot overwrite evaluator support file")
                    valid = set(self.contract.source_files) <= set(normalized)
            passed, detail = False, "malformed or incomplete project artifact"
            if valid:
                with tempfile.TemporaryDirectory(prefix="hermes-project-eval-") as root:
                    root_path = Path(root)
                    for path, content in {**self.contract.support_files, **normalized}.items():
                        target = root_path / path; target.parent.mkdir(parents=True, exist_ok=True); target.write_text(content, encoding="utf-8")
                    try:
                        if self.contract.setup_command:
                            setup = subprocess.run(self.contract.setup_command, cwd=root_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=self.contract.timeout_seconds, check=False)
                            if setup.returncode != 0:
                                raise ValueError("project evaluator setup failed")
                        checked = subprocess.run(self.contract.test_command, cwd=root_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=self.contract.timeout_seconds, check=False)
                    except subprocess.TimeoutExpired as exc:
                        raise ValueError("project evaluator timed out") from exc
                    except OSError as exc:
                        raise ValueError("project evaluator command could not execute") from exc
                    passed, detail = checked.returncode == 0, ("subprocess passed" if checked.returncode == 0 else f"subprocess exited {checked.returncode}")
            results.append({"case": response["case"], "passed": passed, "detail": detail})
        score = sum(item["passed"] for item in results) / len(results)
        return {"schema": "rex-learning-trusted-evaluation-v1", "evaluator_id": self.evaluator_id, "evaluator_type": self.evaluator_type, "test_spec_id": test["id"], "attempt_id": attempt["id"], "score": score, "passed": score >= test["success_threshold"], "case_results": results, "evidence_refs": [f"executable-project://{self.evaluator_id}/{test['id']}/{attempt['id']}/{case}" for case in cases], "contamination": {"status": test.get("contamination_status", "unknown")}, "evaluator_independence": test.get("evaluator_independence", "unknown"), "provider_provenance": {"provider": self.provider, "session_id": self.session_id, "evaluator_id": self.evaluator_id, "contract_sha256": self.contract.sha256, "contract_provenance": self.contract.provenance}}


def _contract_value(document: Any, path: str) -> Any:
    if not isinstance(path, str) or not path.startswith("$."):
        raise ValueError("contract assertion paths must start with '$.'")

    def descend(value: Any, parts: list[str]) -> Any:
        if not parts:
            return value
        part, rest = parts[0], parts[1:]
        if part == "*":
            if not isinstance(value, list):
                return None
            return [item for child in value for item in _flatten(descend(child, rest))]
        if not isinstance(value, Mapping) or part not in value:
            return None
        return descend(value[part], rest)

    def _flatten(value: Any) -> list[Any]:
        return value if isinstance(value, list) else [value]

    return descend(document, path[2:].replace("[*]", ".*").split("."))


def _contract_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, Mapping):
        return "object"
    return "null"


@dataclass(frozen=True)
class ContractArtifactEvaluator:
    """Evaluate learner artifacts using a frozen, allowlisted machine contract."""

    evaluator_id: str
    provider: str
    session_id: str
    evaluator_type: str = "contract_artifact_evaluator"

    def _contract(self, test: dict[str, Any]) -> dict[str, Any]:
        raw = test.get("artifact_schema") or test.get("evaluation_contract")
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError("frozen artifact contract is not valid JSON") from exc
        if not isinstance(raw, Mapping):
            raise ValueError("frozen artifact contract must be a mapping")
        required = raw.get("required_fields")
        field_types = raw.get("field_types")
        assertions = raw.get("assertions")
        if not isinstance(raw.get("artifact_type"), str) or not raw["artifact_type"].strip():
            raise ValueError("frozen artifact contract lacks artifact_type")
        if not isinstance(required, list) or not required or any(not isinstance(item, str) or not item.strip() for item in required):
            raise ValueError("frozen artifact contract lacks required_fields")
        if len(required) != len(set(required)) or not isinstance(field_types, Mapping) or not isinstance(assertions, list):
            raise ValueError("frozen artifact contract is malformed")
        allowed_types = {"string", "array", "object", "boolean", "number", "integer", "null"}
        if any(field not in field_types or field_types[field] not in allowed_types for field in required):
            raise ValueError("frozen artifact contract has unsupported field type")
        for assertion in assertions:
            if not isinstance(assertion, Mapping) or not isinstance(assertion.get("path"), str) or not isinstance(assertion.get("operator"), str):
                raise ValueError("frozen artifact contract has malformed assertion")
            if assertion["operator"] not in {"type_is", "value_is", "min_items", "array_length", "all_unique"}:
                raise ValueError(f"unsupported contract assertion operator: {assertion['operator']}")
        return dict(raw)

    def evaluate(self, test: dict[str, Any], attempt: dict[str, Any]) -> dict[str, Any]:
        if test.get("evaluator_type") != self.evaluator_type or test.get("evaluator_id") != self.evaluator_id:
            raise ValueError("test is not bound to this trusted evaluator")
        cases = list(test.get("cases", []))
        responses = attempt.get("responses", [])
        if not cases or len(cases) != len(set(cases)) or not isinstance(responses, list):
            raise ValueError("frozen cases and attempt responses must be nonempty")
        if [item.get("case") for item in responses if isinstance(item, Mapping)] != cases:
            raise ValueError("attempt responses must contain exactly the frozen cases")
        contract = self._contract(test)
        required = contract["required_fields"]
        field_types = contract["field_types"]
        results = []
        for response in responses:
            answer = response.get("answer")
            if isinstance(answer, str):
                try:
                    answer = json.loads(answer)
                except json.JSONDecodeError:
                    answer = None
            failed = []
            if not isinstance(answer, Mapping):
                failed.append({"operator": "artifact_type", "path": "$", "expected": "object"})
            else:
                for field in required:
                    if field not in answer or answer[field] is None:
                        failed.append({"operator": "required", "path": f"$.{field}"})
                    elif _contract_type(answer[field]) != field_types[field]:
                        failed.append({"operator": "type_is", "path": f"$.{field}", "expected": field_types[field]})
                for assertion in contract["assertions"]:
                    value = _contract_value(answer, assertion["path"])
                    operator = assertion["operator"]
                    passed = (
                        all(_contract_type(item) == assertion.get("value") for item in value) if operator == "type_is" and "[*]" in assertion["path"] and isinstance(value, list) else
                        _contract_type(value) == assertion.get("value") if operator == "type_is" else
                        value == assertion.get("value") if operator == "value_is" else
                        isinstance(value, list) and len(value) >= assertion.get("value", 0) if operator == "min_items" else
                        isinstance(value, list) and len(value) == assertion.get("value") if operator == "array_length" else
                        isinstance(value, list) and len(value) == len({json.dumps(item, sort_keys=True) for item in value})
                    )
                    if not passed:
                        failed.append(dict(assertion))
            results.append({"case": response["case"], "passed": not failed, "failed_assertions": failed})
        score = sum(item["passed"] for item in results) / len(results)
        return {
            "schema": "rex-learning-trusted-evaluation-v1",
            "evaluator_id": self.evaluator_id,
            "evaluator_type": self.evaluator_type,
            "test_spec_id": test["id"], "attempt_id": attempt["id"],
            "score": score, "passed": score >= test["success_threshold"],
            "case_results": results,
            "evidence_refs": [f"contract://{self.evaluator_id}/{test['id']}/{attempt['id']}/{case}" for case in cases],
            "contamination": {"status": test.get("contamination_status", "unknown")},
            "evaluator_independence": test.get("evaluator_independence", "unknown"),
            "provider_provenance": {"provider": self.provider, "session_id": self.session_id, "evaluator_id": self.evaluator_id, "contract": contract},
        }


@dataclass(frozen=True)
class ExecutableGrader:
    """Objective evaluator whose answer key is outside learner-supplied attempts."""

    evaluator_id: str
    answer_key: dict[str, str]
    provider: str
    session_id: str
    evaluator_type: str = "executable_grader"

    def __post_init__(self) -> None:
        object.__setattr__(self, "answer_key", MappingProxyType(dict(self.answer_key)))

    def evaluate(self, test: dict[str, Any], attempt: dict[str, Any]) -> dict[str, Any]:
        if test.get("evaluator_type") != self.evaluator_type or test.get("evaluator_id") != self.evaluator_id:
            raise ValueError("test is not bound to this trusted evaluator")
        responses = attempt.get("responses", [])
        if not isinstance(responses, list):
            raise ValueError("attempt responses must be a list")
        by_case = {item.get("case"): item.get("answer") for item in responses if isinstance(item, dict)}
        if len(by_case) != len(responses) or set(by_case) != set(test.get("cases", [])):
            raise ValueError("attempt responses must contain exactly the frozen cases")
        if not set(test.get("cases", [])).issubset(self.answer_key):
            raise ValueError("trusted evaluator lacks an answer for a frozen case")
        results = []
        for case in test["cases"]:
            expected = self.answer_key[case]
            actual = by_case[case]
            passed = isinstance(actual, str) and actual.strip().casefold() == expected.strip().casefold()
            results.append({"case": case, "passed": passed})
        score = sum(item["passed"] for item in results) / len(results)
        return {
            "schema": "rex-learning-trusted-evaluation-v1",
            "evaluator_id": self.evaluator_id,
            "evaluator_type": self.evaluator_type,
            "test_spec_id": test["id"],
            "attempt_id": attempt["id"],
            "score": score,
            "passed": score >= test["success_threshold"],
            "case_results": results,
            "evidence_refs": [f"grader://{self.evaluator_id}/{test['id']}/{attempt['id']}/{case}" for case in test["cases"]],
            "contamination": {"status": test.get("contamination_status", "unknown")},
            "evaluator_independence": test.get("evaluator_independence", "unknown"),
            "provider_provenance": {
                "provider": self.provider,
                "session_id": self.session_id,
                "evaluator_id": self.evaluator_id,
            },
        }


@dataclass(frozen=True)
class SemanticChecklistGrader:
    """Deterministic grader for open-ended answers with frozen propositions.

    This is intentionally lexical rather than a self-evaluation path: the
    checklist is configured outside the learner and every required proposition
    is retained in the evaluator provenance/evidence.
    """

    evaluator_id: str
    required_propositions: Mapping[str, Sequence[str]]
    provider: str
    session_id: str
    evaluator_type: str = "semantic_checklist_grader"

    def __post_init__(self) -> None:
        frozen = {case: tuple(str(item).casefold().strip() for item in propositions if str(item).strip()) for case, propositions in self.required_propositions.items()}
        if not all(frozen.values()):
            raise ValueError("every semantic checklist case requires propositions")
        object.__setattr__(self, "required_propositions", MappingProxyType(frozen))

    def evaluate(self, test: dict[str, Any], attempt: dict[str, Any]) -> dict[str, Any]:
        if test.get("evaluator_type") != self.evaluator_type or test.get("evaluator_id") != self.evaluator_id:
            raise ValueError("test is not bound to this trusted evaluator")
        responses = attempt.get("responses", [])
        if not isinstance(responses, list):
            raise ValueError("attempt responses must be a list")
        by_case = {item.get("case"): item.get("answer") for item in responses if isinstance(item, dict)}
        cases = list(test.get("cases", []))
        if len(by_case) != len(responses) or set(by_case) != set(cases):
            raise ValueError("attempt responses must contain exactly the frozen cases")
        if not set(cases).issubset(self.required_propositions):
            raise ValueError("trusted evaluator lacks a checklist for a frozen case")
        results = []
        evidence_refs = []
        for case in cases:
            answer = by_case[case]
            normalized = answer.casefold() if isinstance(answer, str) else ""
            missing = [item for item in self.required_propositions[case] if item not in normalized]
            passed = not missing
            results.append({"case": case, "passed": passed, "missing_propositions": missing})
            evidence_refs.append(f"checklist://{self.evaluator_id}/{test['id']}/{attempt['id']}/{case}")
        score = sum(item["passed"] for item in results) / len(results)
        return {
            "schema": "rex-learning-trusted-evaluation-v1",
            "evaluator_id": self.evaluator_id,
            "evaluator_type": self.evaluator_type,
            "test_spec_id": test["id"],
            "attempt_id": attempt["id"],
            "score": score,
            "passed": score >= test["success_threshold"],
            "case_results": results,
            "evidence_refs": evidence_refs,
            "contamination": {"status": test.get("contamination_status", "unknown")},
            "evaluator_independence": test.get("evaluator_independence", "unknown"),
            "provider_provenance": {
                "provider": self.provider,
                "session_id": self.session_id,
                "evaluator_id": self.evaluator_id,
                "required_propositions": {case: list(self.required_propositions[case]) for case in cases},
            },
        }


@dataclass(frozen=True)
class StructuredBehaviorGrader:
    """Deterministic grader for behavior represented as a structured artifact."""

    evaluator_id: str
    required_fields: Mapping[str, Sequence[str]]
    provider: str
    session_id: str
    field_values: Mapping[str, Mapping[str, Sequence[str]]] | None = None
    evaluator_type: str = "structured_behavior_grader"

    def __post_init__(self) -> None:
        frozen = {
            case: tuple(str(field).strip() for field in fields if str(field).strip())
            for case, fields in self.required_fields.items()
        }
        if not frozen or not all(frozen.values()):
            raise ValueError("every structured behavior case requires fields")
        object.__setattr__(self, "required_fields", MappingProxyType(frozen))
        values = {
            case: {str(field).strip(): tuple(str(item).casefold().strip() for item in options if str(item).strip()) for field, options in fields.items()}
            for case, fields in (self.field_values or {}).items()
        }
        if any(not options for fields in values.values() for options in fields.values()):
            raise ValueError("structured behavior field values cannot be empty")
        object.__setattr__(self, "field_values", MappingProxyType(values))

    def evaluate(self, test: dict[str, Any], attempt: dict[str, Any]) -> dict[str, Any]:
        if test.get("evaluator_type") != self.evaluator_type or test.get("evaluator_id") != self.evaluator_id:
            raise ValueError("test is not bound to this trusted evaluator")
        responses = attempt.get("responses", [])
        if not isinstance(responses, list):
            raise ValueError("attempt responses must be a list")
        by_case = {item.get("case"): item.get("answer") for item in responses if isinstance(item, dict)}
        cases = list(test.get("cases", []))
        if len(by_case) != len(responses) or set(by_case) != set(cases):
            raise ValueError("attempt responses must contain exactly the frozen cases")
        if not set(cases).issubset(self.required_fields):
            raise ValueError("trusted evaluator lacks fields for a frozen case")
        results = []
        evidence_refs = []
        configured_values = self.field_values or {}
        for case in cases:
            answer = by_case[case]
            parsed: Any = answer if isinstance(answer, Mapping) else None
            if isinstance(answer, str):
                try:
                    parsed = __import__("json").loads(answer)
                except (TypeError, ValueError):
                    parsed = None
            missing = []
            for field in self.required_fields[case]:
                value = parsed.get(field) if isinstance(parsed, Mapping) else None
                if not _has_content(value):
                    missing.append(field)
                    continue
                allowed = configured_values.get(case, {}).get(field, ())
                if allowed and (not isinstance(value, str) or value.casefold().strip() not in allowed):
                    missing.append(field)
            results.append({"case": case, "passed": not missing, "missing_fields": missing})
            evidence_refs.append(f"structured://{self.evaluator_id}/{test['id']}/{attempt['id']}/{case}")
        score = sum(item["passed"] for item in results) / len(results)
        return {
            "schema": "rex-learning-trusted-evaluation-v1",
            "evaluator_id": self.evaluator_id,
            "evaluator_type": self.evaluator_type,
            "test_spec_id": test["id"],
            "attempt_id": attempt["id"],
            "score": score,
            "passed": score >= test["success_threshold"],
            "case_results": results,
            "evidence_refs": evidence_refs,
            "contamination": {"status": test.get("contamination_status", "unknown")},
            "evaluator_independence": test.get("evaluator_independence", "unknown"),
            "provider_provenance": {
                "provider": self.provider,
                "session_id": self.session_id,
                "evaluator_id": self.evaluator_id,
                "required_fields": {case: list(self.required_fields[case]) for case in cases},
                "field_values": {case: {field: list(values) for field, values in configured_values.get(case, {}).items()} for case in cases if case in configured_values},
            },
        }


@dataclass(frozen=True)
class ExecutableArtifactGrader:
    """Trusted evaluator for structured artifacts with executable field checks.

    Validators are configured outside learner attempts and are never serialized
    into the learner-controlled evidence. A validator exception is a failed
    field, not permission to accept the artifact.
    """

    evaluator_id: str
    required_fields: Mapping[str, Sequence[str]]
    validators: Mapping[str, Mapping[str, Any]]
    provider: str
    session_id: str
    evaluator_type: str = "executable_artifact_grader"

    def __post_init__(self) -> None:
        frozen = {
            case: tuple(str(field).strip() for field in fields if str(field).strip())
            for case, fields in self.required_fields.items()
        }
        if not frozen or not all(frozen.values()):
            raise ValueError("every executable artifact case requires fields")
        normalized_validators: dict[str, dict[str, Any]] = {}
        for case, fields in self.validators.items():
            normalized_validators[case] = {}
            for field, validator in fields.items():
                if not callable(validator):
                    raise ValueError("executable artifact validators must be callable")
                normalized_validators[case][str(field).strip()] = validator
        if any(field not in normalized_validators.get(case, {}) for case, fields in frozen.items() for field in fields):
            raise ValueError("every required artifact field needs an executable validator")
        object.__setattr__(self, "required_fields", MappingProxyType(frozen))
        object.__setattr__(self, "validators", MappingProxyType({case: MappingProxyType(fields) for case, fields in normalized_validators.items()}))

    def evaluate(self, test: dict[str, Any], attempt: dict[str, Any]) -> dict[str, Any]:
        if test.get("evaluator_type") != self.evaluator_type or test.get("evaluator_id") != self.evaluator_id:
            raise ValueError("test is not bound to this trusted evaluator")
        responses = attempt.get("responses", [])
        if not isinstance(responses, list):
            raise ValueError("attempt responses must be a list")
        by_case = {item.get("case"): item.get("answer") for item in responses if isinstance(item, dict)}
        cases = list(test.get("cases", []))
        if len(by_case) != len(responses) or set(by_case) != set(cases):
            raise ValueError("attempt responses must contain exactly the frozen cases")
        if not cases:
            raise ValueError("frozen cases must be nonempty")
        if len(cases) != len(set(cases)):
            raise ValueError("frozen cases must be unique")
        if not set(cases).issubset(self.required_fields):
            raise ValueError("trusted evaluator lacks required fields for frozen cases")
        results = []
        evidence_refs = []
        for case in cases:
            answer = by_case[case]
            parsed: Any = answer if isinstance(answer, Mapping) else None
            if isinstance(answer, str):
                try:
                    parsed = __import__("json").loads(answer)
                except (TypeError, ValueError):
                    parsed = None
            failed_fields = []
            for field in self.required_fields[case]:
                value = parsed.get(field) if isinstance(parsed, Mapping) else None
                if not _has_content(value):
                    failed_fields.append(field)
                    continue
                try:
                    valid = self.validators[case][field](value)
                except Exception:
                    valid = False
                if valid is not True:
                    failed_fields.append(field)
            results.append({"case": case, "passed": not failed_fields, "failed_fields": failed_fields})
            evidence_refs.append(f"executable-artifact://{self.evaluator_id}/{test['id']}/{attempt['id']}/{case}")
        score = sum(item["passed"] for item in results) / len(results)
        return {
            "schema": "rex-learning-trusted-evaluation-v1",
            "evaluator_id": self.evaluator_id,
            "evaluator_type": self.evaluator_type,
            "test_spec_id": test["id"],
            "attempt_id": attempt["id"],
            "score": score,
            "passed": score >= test["success_threshold"],
            "case_results": results,
            "evidence_refs": evidence_refs,
            "contamination": {"status": test.get("contamination_status", "unknown")},
            "evaluator_independence": test.get("evaluator_independence", "unknown"),
            "provider_provenance": {
                "provider": self.provider,
                "session_id": self.session_id,
                "evaluator_id": self.evaluator_id,
                "required_fields": {case: list(self.required_fields[case]) for case in cases},
                "validator_names": {case: {field: getattr(self.validators[case][field], "__name__", type(self.validators[case][field]).__name__) for field in self.required_fields[case]} for case in cases},
            },
        }


def _has_content(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict)):
        return bool(value)
    return value is not None


@dataclass(frozen=True)
class StableTrustedEvaluator:
    """Reject conflicting substantive judgments from a trusted evaluator.

    The wrapper does not make a nondeterministic evaluator trustworthy. It
    requires repeated agreement before an evaluation can enter learning
    evidence. Human-readable rationale may vary; the judgment may not.
    """

    delegate: TrustedEvaluator
    evaluator_id: str
    evaluator_type: str

    def __post_init__(self) -> None:
        if (self.evaluator_id != self.delegate.evaluator_id
                or self.evaluator_type != self.delegate.evaluator_type):
            raise ValueError("stable evaluator identity must match its delegate")

    @property
    def provider(self) -> str:
        return self.delegate.provider

    @property
    def session_id(self) -> str:
        return self.delegate.session_id

    @staticmethod
    def _substantive(result: Mapping[str, Any]) -> str:
        # Provider explanations and auxiliary metadata are not the judgment.
        # Compare the frozen case verdicts (and machine-derived failure detail)
        # plus the aggregate result used by qualification.
        case_results = []
        for item in result.get("case_results", []):
            if isinstance(item, Mapping):
                case_results.append({
                    "case": item.get("case"),
                    "passed": item.get("passed"),
                    "failed_assertions": item.get("failed_assertions"),
                })
        return json.dumps(
            {"score": result.get("score"), "passed": result.get("passed"), "case_results": case_results},
            sort_keys=True, ensure_ascii=False, separators=(",", ":")
        )

    def evaluate(self, test: dict[str, Any], attempt: dict[str, Any]) -> dict[str, Any]:
        first = self.delegate.evaluate(test, attempt)
        second = self.delegate.evaluate(test, attempt)
        if self._substantive(first) != self._substantive(second):
            raise EvaluatorInstabilityError(
                "trusted evaluator returned conflicting substantive judgments"
            )
        return dict(first)
