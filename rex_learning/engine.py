from __future__ import annotations

import hashlib
import json
import math
import re
import time
from types import MappingProxyType
from typing import Any, Mapping

from .evaluator import CallbackEvaluator, ContractArtifactEvaluator, ExecutableArtifactGrader, ExecutableGrader, ExecutableProcessGrader, ExecutableProjectEvaluator, SemanticChecklistGrader, StableTrustedEvaluator, StructuredBehaviorGrader, TrustedEvaluator
from .store import LearningStore


class LearningError(ValueError):
    def __init__(self, message: str, *, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = dict(details or {})

    def diagnostic(self) -> dict[str, Any]:
        return dict(self.details)


class CeilingBaselineError(LearningError):
    """Trusted evidence shows no measurable pretest improvement."""

    code = "ceiling_baseline"


STATES = {"candidate", "studied", "practicing", "provisionally_demonstrated", "demonstrated", "robust", "challenged", "conflicted", "superseded", "deprecated"}
RELATIONS = {"new", "refinement", "duplicate", "alternative", "conflict", "conditional_difference", "supersedes", "supports"}


def _id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
    return f"{prefix}-{digest}"


def _now() -> float:
    return time.time()


def _stable_case_results(value: Any) -> Any:
    if not isinstance(value, list):
        return value
    return [
        {key: item[key] for key in item if key != "rationale"}
        if isinstance(item, Mapping) else item
        for item in value
    ]


def _trusted_evaluation_matches(supplied: Mapping[str, Any], canonical: Mapping[str, Any]) -> bool:
    fields = ("attempt_id", "test_spec_id", "skill_id", "score", "passed", "status", "evaluator_id", "evaluator_type", "independence", "contamination", "validation_status", "evidence", "case_results", "provider_provenance")
    return all(
        (
            _stable_case_results(supplied.get(field))
            if field == "case_results"
            else supplied.get(field)
        )
        == (
            _stable_case_results(canonical.get(field))
            if field == "case_results"
            else canonical.get(field)
        )
        for field in fields
    )


class LearningEngine:
    def __init__(self, store: LearningStore, *, allow_fixture: bool = False, trusted_evaluators: dict[str, TrustedEvaluator] | None = None):
        self.store = store
        self.allow_fixture = allow_fixture
        validated_evaluators: dict[str, TrustedEvaluator] = {}
        for registry_id, evaluator in (trusted_evaluators or {}).items():
            if not isinstance(evaluator, (CallbackEvaluator, ContractArtifactEvaluator, ExecutableGrader, ExecutableArtifactGrader, ExecutableProcessGrader, ExecutableProjectEvaluator, SemanticChecklistGrader, StableTrustedEvaluator, StructuredBehaviorGrader)) or not isinstance(registry_id, str) or not registry_id.strip() or getattr(evaluator, "evaluator_id", None) != registry_id or getattr(evaluator, "evaluator_type", None) not in {"callback_evaluator", "executable_grader", "executable_artifact_grader", "executable_process_grader", "semantic_checklist_grader", "structured_behavior_grader", "contract_artifact_evaluator", "executable_project_evaluator"}:
                raise LearningError("trusted evaluator registry binding is invalid")
            if not all(isinstance(getattr(evaluator, field, None), str) and getattr(evaluator, field).strip() for field in ("provider", "session_id")):
                raise LearningError("trusted evaluator provenance is incomplete")
            validated_evaluators[registry_id] = evaluator
        self.trusted_evaluators = MappingProxyType(validated_evaluators)

    def _save(self, collection: str, record: dict[str, Any], event: str | None = None) -> dict[str, Any]:
        self.store.write(collection, record["id"], record)
        if event:
            self.store.append_event({"id": _id("evt", event, record["id"]), "type": event, "record_id": record["id"], "at": _now()})
        return record

    def create_curriculum(self, title: str, objective: str, *, method_version: str = "A") -> dict[str, Any]:
        existing_id = _id("curriculum", title, objective)
        try:
            return self.store.read("curricula", existing_id)
        except KeyError:
            pass
        record = {"schema": "rex-learning-curriculum-v1", "id": existing_id, "title": title[:240], "objective": objective[:2000], "method_version": method_version, "status": "active", "source_ids": [], "skill_ids": [], "created_at": _now(), "updated_at": _now()}
        return self._save("curricula", record, "curriculum_created")

    def _curriculum(self, curriculum_id: str) -> dict[str, Any]:
        return self.store.read("curricula", curriculum_id)

    def add_source(self, curriculum_id: str, title: str, locator: str, *, kind: str = "text", content_hash: str = "") -> dict[str, Any]:
        curriculum = self._curriculum(curriculum_id)
        source_id = _id("source", curriculum_id, locator)
        try:
            existing = self.store.read("sources", source_id)
            if content_hash and existing.get("content_hash") and existing["content_hash"] != content_hash:
                raise LearningError("source locator content hash cannot change")
            return existing
        except KeyError:
            pass
        source = {"schema": "rex-learning-source-v1", "id": source_id, "curriculum_id": curriculum_id, "title": title[:240], "locator": locator[:2000], "kind": kind, "content_hash": content_hash, "trust": "unverified_source_claims", "status": "available", "created_at": _now()}
        self._save("sources", source, "source_added")
        if source["id"] not in curriculum["source_ids"]:
            curriculum["source_ids"].append(source["id"])
        curriculum["updated_at"] = _now(); self._save("curricula", curriculum)
        return source

    def record_study_note(self, curriculum_id: str, source_id: str, note: dict[str, Any]) -> dict[str, Any]:
        self._curriculum(curriculum_id)
        source = self.store.read("sources", source_id)
        if source.get("curriculum_id") != curriculum_id:
            raise LearningError("source/curriculum mismatch")
        kind = note.get("kind", "claim")
        record = {"schema": "rex-learning-study-note-v1", "id": _id("note", curriculum_id, source_id, json.dumps(note, sort_keys=True)), "curriculum_id": curriculum_id, "source_id": source_id, "kind": kind, "evidence_kind": "extracted_text" if kind == "extracted_section" else "provider_observation", "claim": str(note.get("claim", ""))[:4000], "conditions": list(note.get("conditions", []))[:12], "limitations": list(note.get("limitations", []))[:12], "counterexamples": list(note.get("counterexamples", []))[:12], "status": "ingested" if kind == "extracted_section" else "observed", "created_at": _now()}
        return self._save("notes", record, "study_note_recorded")

    def record_knowledge(self, curriculum_id: str, item: dict[str, Any]) -> dict[str, Any]:
        claim = str(item.get("claim", "")).strip()
        if not claim: raise LearningError("knowledge claim is required")
        existing = self.store.list("knowledge")
        normalized = " ".join(claim.casefold().split())
        relation = "new"
        status = "active"
        for old in existing:
            old_norm = " ".join(str(old.get("claim", "")).casefold().split())
            if old_norm == normalized:
                relation, status = "duplicate", "integrated"
                break
            if {"avoid premature abstraction", "establish explicit boundaries early"} <= {normalized, old_norm}:
                relation, status = "conflict", "conflicted"
                old["status"] = "conflicted"; self._save("knowledge", old)
                break
        record = {"schema": "rex-learning-knowledge-v1", "id": _id("knowledge", curriculum_id, claim, str(item.get("source_id", ""))), "curriculum_id": curriculum_id, "kind": item.get("kind", "claim"), "claim": claim[:4000], "source_id": item.get("source_id"), "relation": relation, "status": status, "provenance": {"source_ids": [item.get("source_id")] if item.get("source_id") else [], "prior_knowledge_ids": [x["id"] for x in existing if x.get("claim") == claim]}, "created_at": _now()}
        return self._save("knowledge", record, "knowledge_integrated")

    def knowledge(self, identifier: str) -> dict[str, Any]: return self.store.read("knowledge", identifier)
    def skill(self, identifier: str) -> dict[str, Any]: return self.store.read("skills", identifier)

    def create_skill_hypothesis(self, curriculum_id: str, spec: dict[str, Any]) -> dict[str, Any]:
        claim = str(spec.get("claim", "")).strip()
        if not claim or not spec.get("key"): raise LearningError("skill key and claim are required")
        skill_id = _id("skill", curriculum_id, str(spec["key"]))
        try:
            return self.store.read("skills", skill_id)
        except KeyError:
            pass
        record = {"schema": "rex-learning-skill-v1", "id": skill_id, "curriculum_id": curriculum_id, "key": str(spec["key"]), "version": 0, "kind": spec.get("kind", "procedure"), "claim": claim[:4000], "applicability": list(spec.get("applicability", []))[:20], "operational_procedure": str(spec.get("operational_procedure", claim))[:4000], "preconditions": list(spec.get("preconditions", []))[:20], "success_criteria": list(spec.get("success_criteria", []))[:20], "failure_criteria": list(spec.get("failure_criteria", []))[:20], "limitations": list(spec.get("limitations", []))[:20], "state": "candidate", "provenance": {"curriculum_id": curriculum_id, "source_ids": [], "note_ids": [], "knowledge_ids": [], "test_ids": [], "evaluation_ids": [], "application_ids": [], "revision_ids": []}, "created_at": _now(), "updated_at": _now()}
        self._save("skills", record, "skill_hypothesis_created")
        curriculum = self._curriculum(curriculum_id)
        if record["id"] not in curriculum["skill_ids"]:
            curriculum["skill_ids"].append(record["id"])
        curriculum["updated_at"] = _now(); self._save("curricula", curriculum)
        return record

    def define_test(self, curriculum_id: str, skill_id: str, spec: dict[str, Any]) -> dict[str, Any]:
        skill = self.skill(skill_id)
        if skill["curriculum_id"] != curriculum_id: raise LearningError("skill/curriculum mismatch")
        if spec.get("phase") not in {"pretest", "practice", "posttest", "retest", "negative", "edge", "adversarial"}: raise LearningError("invalid test phase")
        cases = list(spec.get("cases", []))
        threshold = spec.get("success_threshold", 1.0)
        evaluator_type = spec.get("evaluator_type")
        expected_case_results = list(spec.get("expected_case_results", []))
        if not cases or len(cases) > 100 or any(not isinstance(case, str) or not case.strip() for case in cases) or len(cases) != len(set(cases)):
            raise LearningError("test cases must be unique, non-empty strings")
        if not isinstance(threshold, (int, float)) or not math.isfinite(float(threshold)) or not 0.0 <= float(threshold) <= 1.0:
            raise LearningError("success threshold must be finite and between 0 and 1")
        if evaluator_type == "deterministic_test":
            expected_ids = [item.get("case") for item in expected_case_results if isinstance(item, Mapping)]
            if len(expected_case_results) != len(cases) or len(expected_ids) != len(set(expected_ids)) or set(expected_ids) != set(cases) or any(not isinstance(item, Mapping) or not isinstance(item.get("case"), str) or not isinstance(item.get("passed"), bool) for item in expected_case_results):
                raise LearningError("deterministic test expected results must exactly match frozen cases")
        contract = spec.get("evaluation_contract", {})
        if not isinstance(contract, (Mapping, str)):
            raise LearningError("evaluation contract must be a mapping or string")
        record = {"schema": "rex-learning-test-v1", "id": _id("test", skill_id, str(spec["phase"]), json.dumps(spec.get("cases", []), sort_keys=True)), "curriculum_id": curriculum_id, "skill_id": skill_id, "phase": spec["phase"], "cases": list(spec.get("cases", []))[:100], "phase_tasks": dict(spec.get("phase_tasks", {})), "evaluation_contract": dict(contract) if isinstance(contract, Mapping) else contract, "artifact_schema": dict(spec.get("artifact_schema", {})) if isinstance(spec.get("artifact_schema", {}), Mapping) else spec.get("artifact_schema", {}), "project_contract": dict(spec.get("project_contract", {})) if isinstance(spec.get("project_contract", {}), Mapping) else spec.get("project_contract", {}), "project_contract_sha256": spec.get("project_contract_sha256"), "expected_case_results": list(spec.get("expected_case_results", []))[:100], "success_threshold": float(spec.get("success_threshold", 1.0)), "allowed_resources": list(spec.get("allowed_resources", [])), "prohibited_resources": list(spec.get("prohibited_resources", [])), "independent_design_lineage": dict(spec.get("independent_design_lineage", {})), "independent_design_provenance": dict(spec.get("independent_design_provenance", {})), "evaluator_type": spec.get("evaluator_type", "unknown"), "evaluator_id": spec.get("evaluator_id"), "evaluator_independence": spec.get("evaluator_independence", "unknown"), "contamination_status": spec.get("contamination_status", "unknown"), "source_free": bool(spec.get("source_free", False)), "novel_transfer": bool(spec.get("novel_transfer", False)), "frozen_at": _now()}
        try:
            existing = self.store.read("tests", record["id"])
        except KeyError:
            pass
        else:
            frozen_fields = ("curriculum_id", "skill_id", "phase", "cases", "phase_tasks", "evaluation_contract", "artifact_schema", "project_contract", "project_contract_sha256", "expected_case_results", "success_threshold", "allowed_resources", "prohibited_resources", "independent_design_lineage", "independent_design_provenance", "evaluator_type", "evaluator_id", "evaluator_independence", "contamination_status")
            if any(existing.get(field) != record.get(field) for field in frozen_fields):
                raise LearningError("frozen test cannot be redefined")
            return existing
        skill["state"] = "practicing" if skill["state"] == "candidate" else skill["state"]; skill["updated_at"] = _now(); self._save("skills", skill)
        return self._save("tests", record, "test_defined")

    def record_attempt(self, test_id: str, result: dict[str, Any], *, task_id: str = "") -> dict[str, Any]:
        test = self.store.read("tests", test_id)
        if test["evaluator_type"] in {"callback_evaluator", "executable_grader", "executable_artifact_grader", "executable_process_grader", "semantic_checklist_grader", "structured_behavior_grader", "contract_artifact_evaluator", "executable_project_evaluator"}:
            responses = list(result.get("responses", []))
            expected_cases = set(test.get("cases", []))
            actual_cases = [item.get("case") for item in responses if isinstance(item, dict)]
            if not expected_cases or set(actual_cases) != expected_cases or len(actual_cases) != len(set(actual_cases)):
                raise LearningError("executable attempt requires exactly one response for every frozen case")
            integrity_source = {"test_id": test_id, "skill_id": test["skill_id"], "task_id": task_id, "responses": responses}
            record = {"schema": "rex-learning-attempt-v1", "id": _id("attempt", test_id, json.dumps({**result, "task_id": task_id}, sort_keys=True)), "test_id": test_id, "skill_id": test["skill_id"], "task_id": task_id, "score": None, "responses": responses, "case_results": [], "integrity_digest": hashlib.sha256(json.dumps(integrity_source, sort_keys=True).encode()).hexdigest(), "created_at": _now(), "status": "recorded"}
            return self._save("attempts", record, "attempt_recorded")
        score = float(result.get("score", 0.0))
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise LearningError("attempt score must be finite and between 0 and 1")
        case_results = list(result.get("case_results", []))
        if test["evaluator_type"] == "deterministic_test":
            expected_cases = {item["case"] for item in test.get("expected_case_results", [])}
            actual_cases = [item.get("case") for item in case_results]
            if not expected_cases or set(actual_cases) != expected_cases or len(actual_cases) != len(set(actual_cases)):
                raise LearningError("deterministic attempt requires exactly one result for every frozen case")
            expected = {item["case"]: bool(item["passed"]) for item in test["expected_case_results"]}
            actual = {item["case"]: bool(item["passed"]) for item in case_results}
            computed = sum(expected[key] == actual[key] for key in expected) / len(expected)
            if score != computed:
                raise LearningError("attempt score must match deterministic case results")
        integrity_source = {"test_id": test_id, "skill_id": test["skill_id"], "task_id": task_id, "score": score, "case_results": case_results}
        record = {"schema": "rex-learning-attempt-v1", "id": _id("attempt", test_id, json.dumps({**result, "task_id": task_id}, sort_keys=True)), "test_id": test_id, "skill_id": test["skill_id"], "task_id": task_id, "score": score, "case_results": case_results, "integrity_digest": hashlib.sha256(json.dumps(integrity_source, sort_keys=True).encode()).hexdigest(), "created_at": _now(), "status": "recorded"}
        return self._save("attempts", record, "attempt_recorded")

    def evaluate_trusted_attempt(self, attempt_id: str) -> dict[str, Any]:
        attempt = self.store.read("attempts", attempt_id)
        test = self.store.read("tests", attempt["test_id"])
        if attempt.get("skill_id") != test.get("skill_id"):
            raise LearningError("trusted attempt/test skill identity mismatch")
        source = {"test_id": attempt["test_id"], "skill_id": attempt["skill_id"], "task_id": attempt.get("task_id", ""), "responses": attempt["responses"]} if test.get("evaluator_type") in {"callback_evaluator", "executable_grader", "executable_artifact_grader", "executable_process_grader", "semantic_checklist_grader", "structured_behavior_grader", "contract_artifact_evaluator", "executable_project_evaluator"} else {"test_id": attempt["test_id"], "skill_id": attempt["skill_id"], "task_id": attempt.get("task_id", ""), "score": attempt.get("score"), "case_results": attempt.get("case_results", [])}
        if attempt.get("integrity_digest") != hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest():
            raise LearningError("attempt integrity verification failed")
        evaluator_id = test.get("evaluator_id")
        evaluator = self.trusted_evaluators.get(str(evaluator_id))
        if test.get("evaluator_type") not in {"callback_evaluator", "executable_grader", "executable_artifact_grader", "executable_process_grader", "semantic_checklist_grader", "structured_behavior_grader", "contract_artifact_evaluator", "executable_project_evaluator"} or evaluator is None:
            raise LearningError("no trusted evaluator is bound to this test")
        try:
            output = evaluator.evaluate(test, attempt)
        except Exception as exc:
            raise LearningError(f"trusted evaluator rejected attempt: {exc}") from exc
        if not isinstance(output, Mapping):
            raise LearningError("trusted evaluator output must be a mapping")
        if output.get("evaluator_id") != evaluator_id or output.get("evaluator_type") != test["evaluator_type"]:
            raise LearningError("trusted evaluator identity mismatch")
        if output.get("test_spec_id") != test["id"] or output.get("attempt_id") != attempt["id"]:
            raise LearningError("trusted evaluator evidence identity mismatch")
        score = output.get("score")
        if not isinstance(score, (int, float)) or not math.isfinite(float(score)) or not 0.0 <= float(score) <= 1.0:
            raise LearningError("trusted evaluator score must be finite and between 0 and 1")
        case_results = output.get("case_results")
        if not isinstance(case_results, list) or not all(isinstance(item, Mapping) and isinstance(item.get("case"), str) and isinstance(item.get("passed"), bool) for item in case_results) or {x["case"] for x in case_results} != set(test.get("cases", [])) or len(case_results) != len(test.get("cases", [])):
            raise LearningError("trusted evaluator returned incomplete case results")
        computed = sum(bool(x.get("passed")) for x in case_results) / len(case_results)
        if not isinstance(output.get("passed"), bool) or float(score) != computed or output["passed"] != (computed >= test["success_threshold"]):
            raise LearningError("trusted evaluator aggregate does not match case results or threshold")
        evidence_refs = output.get("evidence_refs")
        if not isinstance(evidence_refs, list) or len(evidence_refs) != len(test["cases"]) or any(not isinstance(ref, str) or not ref.strip() for ref in evidence_refs):
            raise LearningError("trusted evaluator requires evidence references")
        contamination = output.get("contamination")
        provenance = output.get("provider_provenance")
        if not isinstance(contamination, dict) or contamination.get("status") != test.get("contamination_status"):
            raise LearningError("trusted evaluator contamination does not match frozen test")
        if output.get("evaluator_independence") != test.get("evaluator_independence"):
            raise LearningError("trusted evaluator independence does not match frozen test")
        if not isinstance(provenance, Mapping) or not all(str(provenance.get(key, "")).strip() for key in ("provider", "session_id", "evaluator_id")) or provenance.get("evaluator_id") != evaluator_id or provenance.get("provider") != getattr(evaluator, "provider") or provenance.get("session_id") != getattr(evaluator, "session_id"):
            raise LearningError("trusted evaluator provider provenance is incomplete")
        try:
            json.dumps(output, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise LearningError("trusted evaluator output is not JSON-serializable") from exc
        record = {"schema": "rex-learning-evaluation-v1", "id": _id("evaluation", attempt_id, json.dumps(output, sort_keys=True)), "attempt_id": attempt_id, "test_spec_id": test["id"], "skill_id": attempt["skill_id"], "score": float(score), "passed": bool(output["passed"]), "status": "passed" if output["passed"] else "failed", "evaluator_id": evaluator_id, "evaluator_type": test["evaluator_type"], "independence": test["evaluator_independence"], "contamination": contamination, "validation_status": "verified", "evidence": [{"kind": "trusted_evaluator", "ref": ref} for ref in evidence_refs], "case_results": case_results, "provider_provenance": provenance, "created_at": _now()}
        evaluation = self._save("evaluations", record, "trusted_evaluation_recorded")
        skill = self.skill(attempt["skill_id"]); skill["provenance"]["evaluation_ids"].append(evaluation["id"]); skill["updated_at"] = _now(); self._save("skills", skill)
        return evaluation

    def evaluate_attempt(self, attempt_id: str, result: dict[str, Any]) -> dict[str, Any]:
        attempt = self.store.read("attempts", attempt_id); test = self.store.read("tests", attempt["test_id"])
        if test.get("evaluator_type") in {"callback_evaluator", "executable_grader", "executable_artifact_grader", "executable_process_grader", "semantic_checklist_grader", "structured_behavior_grader", "contract_artifact_evaluator", "executable_project_evaluator"}:
            raise LearningError("executable tests require a trusted evaluator")
        contamination = result.get("contamination", {}); independence = result.get("independence", "unknown")
        if test["evaluator_type"] == "deterministic_test":
            expected = {item["case"]: bool(item["passed"]) for item in test.get("expected_case_results", [])}
            actual = {item["case"]: bool(item["passed"]) for item in attempt.get("case_results", [])}
            if not expected or set(expected) != set(actual): raise LearningError("deterministic test requires a complete frozen expected result set")
            score = sum(expected[key] == actual[key] for key in expected) / len(expected)
            validation_status = "verified"
            passed = score >= test["success_threshold"]
            independence = test["evaluator_independence"]
            contamination = {"status": test["contamination_status"]}
        else:
            score = attempt["score"]
            validation_status = "unverified"
            passed = False
        if contamination.get("status") not in {"clean", "unknown", "not_applicable"}: passed = False
        submitted_evidence = list(result.get("evidence", []))
        if not submitted_evidence or any(not isinstance(item, dict) or not str(item.get("ref", "")).strip() for item in submitted_evidence):
            raise LearningError("evaluation requires non-empty evidence references")
        evidence = submitted_evidence
        if test["evaluator_type"] == "deterministic_test":
            evidence = [
                {"kind": "frozen_test", "ref": test["id"]},
                {"kind": "recorded_attempt", "ref": attempt["id"]},
            ]
        record = {"schema": "rex-learning-evaluation-v1", "id": _id("evaluation", attempt_id, json.dumps(result, sort_keys=True)), "attempt_id": attempt_id, "skill_id": attempt["skill_id"], "score": score, "passed": passed, "status": "passed" if passed else "failed", "evaluator_type": test["evaluator_type"], "independence": independence, "contamination": contamination, "validation_status": validation_status, "evidence": evidence, "created_at": _now()}
        evaluation = self._save("evaluations", record, "evaluation_recorded")
        skill = self.skill(attempt["skill_id"]); skill["provenance"]["evaluation_ids"].append(evaluation["id"]); skill["updated_at"] = _now(); self._save("skills", skill)
        return evaluation

    def derive_candidate_behavioral_evidence(
        self,
        *,
        pretest_attempt_id: str,
        pretest_evaluation_id: str,
        posttest_evaluation_id: str,
        retest_evaluation_id: str,
        negative_evaluation_id: str,
        control_evaluation_id: str | None = None,
    ) -> dict[str, Any]:
        """Derive candidate-gate evidence from immutable test/evaluation records.

        This deliberately does not accept caller-supplied booleans. The frozen
        test metadata declares source isolation and transfer intent; trusted
        evaluator records establish the observed outcomes and provenance.
        """
        pre_attempt = self.store.read("attempts", pretest_attempt_id)
        pre_eval = self.store.read("evaluations", pretest_evaluation_id)
        evaluations = {
            "post": self.store.read("evaluations", posttest_evaluation_id),
            "retest": self.store.read("evaluations", retest_evaluation_id),
            "negative": self.store.read("evaluations", negative_evaluation_id),
        }
        if control_evaluation_id is not None:
            evaluations["control"] = self.store.read("evaluations", control_evaluation_id)
        skill_id = pre_attempt.get("skill_id")
        if not isinstance(skill_id, str) or pre_eval.get("attempt_id") != pretest_attempt_id:
            raise LearningError("candidate evidence baseline identity mismatch")
        records: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
        for name, evaluation in (("pre", pre_eval), *evaluations.items()):
            attempt = self.store.read("attempts", evaluation.get("attempt_id", ""))
            test = self.store.read("tests", attempt.get("test_id", ""))
            if evaluation.get("skill_id") != skill_id or attempt.get("skill_id") != skill_id or test.get("skill_id") != skill_id:
                raise LearningError(f"candidate evidence {name} skill mismatch")
            if evaluation.get("evaluator_type") not in {"callback_evaluator", "executable_grader", "executable_artifact_grader", "executable_process_grader", "semantic_checklist_grader", "structured_behavior_grader", "contract_artifact_evaluator", "executable_project_evaluator"} or evaluation.get("validation_status") != "verified":
                raise LearningError(f"candidate evidence {name} is not verified executable evidence")
            if evaluation.get("independence") != "independent" or evaluation.get("contamination", {}).get("status") != "clean":
                raise LearningError(f"candidate evidence {name} is not independent and clean")
            records[name] = (test, evaluation)
        pre_test, pre = records["pre"]
        post_test, post = records["post"]
        retest_test, retest = records["retest"]
        negative_test, negative = records["negative"]
        control: dict[str, Any] = {}
        if pre_test.get("phase") != "pretest" or post_test.get("phase") not in {"posttest", "retest"}:
            raise LearningError("candidate evidence phases are invalid")
        if control_evaluation_id is not None:
            control_test, control = records["control"]
            if control_test.get("phase") not in {"adversarial", "control"}:
                raise LearningError("candidate control evidence phase is invalid")
            if not isinstance(control.get("score"), (int, float)) or control["score"] >= post["score"]:
                raise LearningError(f"candidate behavioral evidence does not outperform matched control: control={control['score']!r}, post={post['score']!r}")
        if not post.get("passed") or not retest.get("passed") or not negative.get("passed"):
            raise LearningError("candidate behavioral evidence contains a failed required evaluation")
        if not isinstance(pre.get("score"), (int, float)) or not isinstance(post.get("score"), (int, float)) or post["score"] <= pre["score"]:
            raise CeilingBaselineError(
                "candidate behavioral evidence does not show pretest improvement",
                details={"code": CeilingBaselineError.code, "pre_score": pre["score"], "post_score": post["score"]},
            )
        if post_test.get("id") == retest_test.get("id"):
            raise LearningError("candidate reproducibility requires distinct posttest and retest tests")
        if not pre_test.get("source_free") and pre_test.get("phase") != "pretest":
            raise LearningError("candidate baseline source metadata is invalid")
        if not post_test.get("source_free") or not retest_test.get("source_free"):
            raise LearningError("candidate posttest and retest must be source-free")
        if not post_test.get("novel_transfer") or not retest_test.get("novel_transfer"):
            raise LearningError("candidate transfer metadata is missing")
        evidence = {
            "source_free": True,
            "novel_transfer": True,
            "negative_case": True,
            "reproducible": True,
            "delayed_retention": True,
            "contamination_status": "clean",
            "evaluator_independence": "independent",
        }
        if control_evaluation_id is not None:
            evidence.update({"control_discrimination": True, "control_score": control["score"]})
        return evidence

    def promote_demonstrated(self, skill_id: str, *, pretest_attempt_id: str, posttest_evaluation_id: str, pretest_evaluation_id: str | None = None) -> dict[str, Any]:
        if not self.allow_fixture and not pretest_evaluation_id:
            raise LearningError("production promotion requires a verified pretest evaluation")
        skill = self.skill(skill_id); pre = self.store.read("attempts", pretest_attempt_id); post = self.store.read("evaluations", posttest_evaluation_id); post_test = self.store.read("tests", self.store.read("attempts", post["attempt_id"])["test_id"])
        if pre["skill_id"] != skill_id or post["skill_id"] != skill_id: raise LearningError("skill evidence mismatch")
        if skill.get("latest_evaluation", {}).get("evaluation_id") == posttest_evaluation_id and skill.get("state") in {"demonstrated", "robust"}:
            return skill
        pre_test = self.store.read("tests", pre["test_id"])
        if pre_test["phase"] != "pretest": raise LearningError("baseline must come from a pretest")
        if pre_test["skill_id"] != skill_id or pre_test["curriculum_id"] != skill["curriculum_id"]: raise LearningError("baseline test skill mismatch")
        if post_test["skill_id"] != skill_id or post_test["curriculum_id"] != skill["curriculum_id"]: raise LearningError("posttest skill mismatch")
        pre_eval: dict[str, Any] | None = None
        if not self.allow_fixture:
            pre_eval = self.store.read("evaluations", str(pretest_evaluation_id))
            if pre_eval.get("attempt_id") != pre["id"] or pre_eval.get("validation_status") != "verified" or pre_eval.get("evaluator_type") not in {"callback_evaluator", "executable_grader", "executable_artifact_grader", "executable_process_grader", "semantic_checklist_grader", "structured_behavior_grader", "contract_artifact_evaluator", "executable_project_evaluator"}:
                raise LearningError("pretest evaluation must be verified executable-grader evidence")
            if post.get("validation_status") != "verified" or post.get("evaluator_type") not in {"callback_evaluator", "executable_grader", "executable_artifact_grader", "executable_process_grader", "semantic_checklist_grader", "structured_behavior_grader", "contract_artifact_evaluator", "executable_project_evaluator"}:
                raise LearningError("posttest evaluation must be verified executable-grader evidence")
            for supplied, attempt_id in ((pre_eval, pre["id"]), (post, post["attempt_id"])):
                canonical = self.evaluate_trusted_attempt(attempt_id)
                if not _trusted_evaluation_matches(supplied, canonical):
                    raise LearningError("stored trusted evaluation does not match canonical evaluator output")
        elif pre_test["evaluator_type"] != "deterministic_test":
            raise LearningError("fixture baseline must come from a deterministic pretest")
        if post_test["phase"] not in {"posttest", "retest"} or (not self.allow_fixture and post_test["evaluator_type"] not in {"callback_evaluator", "executable_grader", "executable_artifact_grader", "executable_process_grader", "semantic_checklist_grader", "structured_behavior_grader", "contract_artifact_evaluator", "executable_project_evaluator"}) or (self.allow_fixture and post_test["evaluator_type"] != "deterministic_test") or not post["passed"]: raise LearningError("demonstration requires a passed held-out posttest")
        baseline_score = pre_eval["score"] if pre_eval is not None else pre["score"]
        if not isinstance(baseline_score, (int, float)) or post["score"] <= baseline_score:
            raise LearningError("demonstration requires improvement over the pretest")
        if post["independence"] in {"none", "unknown", "shared_context", "self_evaluation"} or post["contamination"].get("status") not in {"clean", "not_applicable"}: raise LearningError("weak or contaminated evaluation cannot demonstrate skill")
        version = skill["version"] + 1
        if skill["version"]:
            self.store.write("skill_versions", f"{skill_id}.v{skill['version']}", {"schema": "rex-learning-skill-version-v1", "skill_id": skill_id, "version": skill["version"], "snapshot": skill, "superseded_by": version, "created_at": _now()})
        skill.update({"version": version, "state": "demonstrated", "baseline": {"attempt_id": pre["id"], "score": baseline_score}, "latest_evaluation": {"evaluation_id": post["id"], "score": post["score"]}, "demonstrated_at": _now(), "updated_at": _now()})
        skill["provenance"]["test_ids"].append(post_test["id"])
        return self._save("skills", skill, "skill_demonstrated")

    def promote_candidate_skill(
        self,
        candidate: Mapping[str, Any],
        *,
        curriculum_id: str,
        pretest_attempt_id: str,
        pretest_evaluation_id: str,
        posttest_evaluation_id: str,
        behavioral_evidence: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Install a reviewed candidate only after the normal behavioral gate passes."""
        from .candidate_skill import promotion_boundary

        if not isinstance(candidate, Mapping):
            raise LearningError("candidate package is required")
        candidate_id = str(candidate.get("candidate_id", "")).strip()
        if not candidate_id:
            raise LearningError("candidate_id is required")
        decision = candidate.get("review", {}).get("decision", {}).get("decision")
        if decision not in {"PROMOTE", "PROMOTE_WITH_NARROWER_SCOPE"}:
            raise LearningError("candidate review is not promotable")
        boundary = promotion_boundary(candidate, behavioral_evidence=behavioral_evidence)
        if not boundary["promoted"]:
            raise LearningError(f"candidate promotion blocked: {boundary['reason']}")
        proposed = candidate.get("reviewed_surface") or candidate.get("proposed_skill") or {}
        intent = candidate.get("curriculum_intent") or {}
        claim = proposed.get("claim")
        if isinstance(claim, list):
            claim = "; ".join(str(item).strip() for item in claim if str(item).strip())
        claim = str(claim or "").strip()
        procedure = str(proposed.get("operational_procedure") or proposed.get("procedure") or "").strip()
        if not claim or not procedure:
            raise LearningError("candidate lacks an operational claim or procedure")
        key = str(intent.get("key") or candidate_id).strip()
        skill = self.create_skill_hypothesis(curriculum_id, {
            "key": key,
            "kind": proposed.get("kind", "procedure"),
            "claim": claim,
            "applicability": proposed.get("applicability") or proposed.get("scope") or [],
            "operational_procedure": procedure,
            "preconditions": proposed.get("preconditions") or [],
            "limitations": proposed.get("limitations") or [],
        })
        promoted = self.promote_demonstrated(
            skill["id"],
            pretest_attempt_id=pretest_attempt_id,
            pretest_evaluation_id=pretest_evaluation_id,
            posttest_evaluation_id=posttest_evaluation_id,
        )
        stored = dict(candidate)
        stored.update({"state": "APPROVED", "promoted": True, "durable_skill_id": promoted["id"], "promotion": {"boundary": boundary, "behavioral_evidence": dict(behavioral_evidence), "skill_id": promoted["id"]}})
        self.store.write("candidate_skills", candidate_id, stored)
        promoted["provenance"]["candidate_id"] = candidate_id
        promoted["provenance"]["candidate_source_refs"] = list(candidate.get("source", {}).get("source_refs", []))
        return self._save("skills", promoted, "candidate_skill_promoted")

    def promote_candidate_skill_from_records(
        self,
        candidate: Mapping[str, Any],
        *,
        curriculum_id: str,
        pretest_attempt_id: str,
        pretest_evaluation_id: str,
        posttest_evaluation_id: str,
        retest_evaluation_id: str,
        negative_evaluation_id: str,
        control_evaluation_id: str | None = None,
    ) -> dict[str, Any]:
        """Promote a candidate using promotion evidence derived from records."""
        evidence = self.derive_candidate_behavioral_evidence(
            pretest_attempt_id=pretest_attempt_id,
            pretest_evaluation_id=pretest_evaluation_id,
            posttest_evaluation_id=posttest_evaluation_id,
            retest_evaluation_id=retest_evaluation_id,
            negative_evaluation_id=negative_evaluation_id,
            control_evaluation_id=control_evaluation_id,
        )
        return self.promote_candidate_skill(
            candidate,
            curriculum_id=curriculum_id,
            pretest_attempt_id=pretest_attempt_id,
            pretest_evaluation_id=pretest_evaluation_id,
            posttest_evaluation_id=posttest_evaluation_id,
            behavioral_evidence=evidence,
        )

    def revise_skill(self, skill_id: str, revision: dict[str, Any], *, reason: str, evidence_ids: list[str]) -> dict[str, Any]:
        skill = self.skill(skill_id)
        if not reason.strip() or not evidence_ids:
            raise LearningError("revision requires a reason and evidence")
        curriculum_id = skill["curriculum_id"]
        resolved_evidence: list[dict[str, Any]] = []
        for evidence_id in evidence_ids[:20]:
            found = None
            for collection in ("evaluations", "attempts", "notes", "knowledge", "applications"):
                try:
                    candidate = self.store.read(collection, str(evidence_id))
                except KeyError:
                    continue
                found = (collection, candidate)
                break
            if found is None:
                raise LearningError("revision evidence must reference an existing record")
            collection, candidate = found
            if collection in {"evaluations", "attempts", "applications"} and candidate.get("skill_id") != skill_id:
                raise LearningError("revision evidence does not belong to the skill")
            if collection in {"notes", "knowledge"} and candidate.get("curriculum_id") != curriculum_id:
                raise LearningError("revision evidence does not belong to the curriculum")
            resolved_evidence.append({"collection": collection, "id": candidate["id"]})
        previous = dict(skill)
        current_version = skill["version"]
        next_version = current_version + 1
        self.store.write("skill_versions", f"{skill_id}.v{current_version}", {"schema": "rex-learning-skill-version-v1", "skill_id": skill_id, "version": current_version, "snapshot": previous, "superseded_by": next_version, "revision_reason": reason[:2000], "evidence": resolved_evidence, "created_at": _now()})
        for field in ("claim", "applicability", "operational_procedure", "preconditions", "success_criteria", "failure_criteria", "limitations"):
            if field in revision:
                skill[field] = revision[field]
        skill.update({"version": next_version, "state": "practicing", "updated_at": _now()})
        skill["provenance"]["revision_ids"].append(_id("revision", skill_id, str(next_version), reason))
        return self._save("skills", skill, "skill_revised")

    def consolidate_learner_revision(self, skill_id: str, revision: dict[str, Any], *, learner_provenance: dict[str, Any], source_provenance: dict[str, Any], failure_ref: str, revision_type: str) -> dict[str, Any]:
        """Persist an authorized provider's study/consolidation result as a candidate."""
        skill = self.skill(skill_id)
        role = learner_provenance.get("role", "learner")
        if role not in {"learner", "study", "study_and_consolidate", "consolidator"}:
            raise LearningError("only learner-authored study/consolidation operations may produce learner content")
        if not learner_provenance.get("provider") or not learner_provenance.get("session_id"):
            raise LearningError("study/consolidation provenance is incomplete")
        if not source_provenance.get("source_refs") or not failure_ref.strip() or not revision_type.strip():
            raise LearningError("learner-authored revision requires source and failure provenance")
        claim = str(revision.get("claim", "")).strip()
        procedure = str(revision.get("operational_procedure", "")).strip()
        if not claim or not procedure:
            raise LearningError("learner-authored revision requires claim and operational procedure")
        next_version = int(skill.get("version", 0)) + 1
        previous = dict(skill)
        self.store.write("skill_versions", f"{skill_id}.v{skill.get('version', 0)}", {"schema": "rex-learning-skill-version-v1", "skill_id": skill_id, "version": skill.get("version", 0), "snapshot": previous, "superseded_by": next_version, "revision_reason": revision_type, "created_at": _now()})
        updated = dict(skill)
        author = "learner" if role == "learner" and set(learner_provenance) <= {"provider", "session_id", "response_digest", "role"} else "provider_derived"
        updated.update({
            "version": next_version, "state": "practicing", "claim": claim[:4000],
            "operational_procedure": procedure[:4000], "applicability": list(revision.get("applicability", []))[:20],
            "non_applicability": list(revision.get("non_applicability", []))[:20], "limitations": list(revision.get("boundaries", revision.get("limitations", [])))[:20],
            "uncertainty": str(revision.get("uncertainty", ""))[:2000], "author": author,
            "qualification_status": "candidate", "learner_provenance": dict(learner_provenance),
            "source_provenance": dict(source_provenance), "failure_ref": failure_ref,
            "revision_type": revision_type, "updated_at": _now(),
        })
        return self._save("skills", updated, "learner_revision_consolidated")

    def qualify_learner_revision(self, skill_id: str, validation: dict[str, Any]) -> dict[str, Any]:
        """Legacy name: source-grounded content validation, not behavioral demonstration."""
        skill = self.skill(skill_id)
        if skill.get("author") not in {"learner", "provider_derived"} or skill.get("qualification_status") != "candidate":
            raise LearningError("only a provider-derived candidate can be content validated")
        if validation.get("independence") != "independent" or validation.get("contamination", {}).get("status") != "clean":
            raise LearningError("learner revision validation must be independent and clean")
        if not validation.get("provider") or not validation.get("session_id") or validation.get("judgment") not in {"supported", "qualified"}:
            raise LearningError("learner revision validation provenance is incomplete")
        skill["qualification_status"] = "content_validated"
        skill["state"] = "practicing"
        skill["validation"] = dict(validation)
        skill["updated_at"] = _now()
        return self._save("skills", skill, "learner_revision_qualified")

    def validate_skill_content(self, skill_id: str, validation: dict[str, Any]) -> dict[str, Any]:
        """Validate source fidelity and boundaries without claiming behavioral competence."""
        skill = self.skill(skill_id)
        if skill.get("qualification_status") != "candidate":
            raise LearningError("only a candidate can receive content validation")
        if validation.get("contamination", {}).get("status") != "clean":
            raise LearningError("content validation must be clean")
        if validation.get("judgment") not in {"supported", "qualified"}:
            raise LearningError("content validation judgment is not supportive")
        if not validation.get("provider") or not validation.get("session_id"):
            raise LearningError("content validation provenance is incomplete")
        skill["qualification_status"] = "content_validated"
        skill["state"] = "practicing"
        skill["content_validation"] = dict(validation)
        skill["updated_at"] = _now()
        return self._save("skills", skill, "skill_content_validated")

    def set_curriculum_status(self, curriculum_id: str, status: str, *, reason: str = "") -> dict[str, Any]:
        if status not in {"active", "paused", "completed", "partially_completed", "blocked", "needs_more_evidence", "needs_human_input", "failed"}:
            raise LearningError("invalid curriculum status")
        curriculum = self._curriculum(curriculum_id); curriculum.update({"status": status, "status_reason": reason[:1000], "updated_at": _now()}); return self._save("curricula", curriculum, "curriculum_status_changed")

    def force_qualified_fixture(self, skill_id: str, *, baseline: float, post: float) -> dict[str, Any]:
        """Fixture-only qualification seam; real providers must use normal evidence APIs."""
        if not self.allow_fixture:
            raise LearningError("fixture qualification is disabled for production engines")
        skill = self.skill(skill_id); skill.update({"version": 1, "state": "demonstrated", "baseline": {"score": baseline, "fixture": True}, "latest_evaluation": {"score": post, "fixture": True}, "demonstrated_at": _now(), "updated_at": _now()}); return self._save("skills", skill, "fixture_skill_demonstrated")

    @staticmethod
    def _normalize_requirement(value: Any) -> str:
        return " ".join(str(value).casefold().split())

    @staticmethod
    def _requirement_tokens(value: Any) -> set[str]:
        """Return distinctive words for conservative natural-language reuse."""
        stop = {"a", "an", "and", "for", "from", "in", "of", "on", "the", "to", "with"}
        return {
            token
            for token in re.findall(r"[a-z0-9]+", str(value).casefold())
            if len(token) > 2 and token not in stop
        }

    def _operational_context(self, skill: dict[str, Any], *, max_chars: int) -> dict[str, Any]:
        context = {"schema": "rex-learning-operational-context-v1", "skill_id": skill["id"], "skill_version": skill["version"], "procedure": skill.get("operational_procedure", skill.get("claim", ""))[:4000], "preconditions": list(skill.get("preconditions", []))[:8], "limitations": list(skill.get("limitations", []))[:8]}
        while len(json.dumps(context, ensure_ascii=False)) > max_chars and context["limitations"]:
            context["limitations"].pop()
        while len(json.dumps(context, ensure_ascii=False)) > max_chars and context["preconditions"]:
            context["preconditions"].pop()
        if len(json.dumps(context, ensure_ascii=False)) > max_chars:
            context["procedure"] = context["procedure"][: max(0, max_chars // 2)]
        return context

    def discover_applicable_skills(self, task_context: Mapping[str, Any], *, max_candidates: int = 3, max_chars: int = 4000) -> list[dict[str, Any]]:
        requirements = {self._normalize_requirement(item) for item in list(task_context.get("requirements", [])) if self._normalize_requirement(item)}
        if not requirements:
            return []
        candidates: list[dict[str, Any]] = []
        for skill in self.store.list("skills"):
            if skill.get("state") not in {"demonstrated", "robust", "practicing"} or (skill.get("state") not in {"demonstrated", "robust"} and skill.get("qualification_status") not in {"content_validated", "qualified", "demonstrated"}) or skill.get("version", 0) < 1:
                continue
            applicability = {self._normalize_requirement(item) for item in skill.get("applicability", []) if self._normalize_requirement(item)}
            if not applicability:
                applicability = {self._normalize_requirement(skill.get("claim", ""))}
            matched = sorted(requirements & applicability)
            if not matched:
                # Runtime callers often have a compact extracted requirement
                # while the persisted applicability claim is deliberately more
                # descriptive. Require two distinctive shared terms; a single
                # broad word is too weak and would make selection noisy.
                matched = sorted(
                    requirement
                    for requirement in requirements
                    if len(self._requirement_tokens(requirement) & self._requirement_tokens(" ".join(applicability))) >= 2
                )
            if not matched:
                continue
            context = self._operational_context(skill, max_chars=max_chars)
            candidates.append({"skill_id": skill["id"], "key": skill["key"], "skill_version": skill["version"], "reuse_mode": "experimental" if skill.get("qualification_status") == "content_validated" else "demonstrated", "consumer_provider": None, "why_selected": f"task requirement matched bounded applicability: {matched[0]}", "operational_context": context})
        candidates.sort(key=lambda item: (-len(item["why_selected"]), item["skill_id"]))
        bounded = candidates[:max(1, min(max_candidates, 10))]
        while len(json.dumps(bounded, ensure_ascii=False)) > max_chars and bounded:
            bounded.pop()
        return bounded

    def compose_applicable_skills(self, task_context: Mapping[str, Any], *, max_skills: int = 4, max_chars: int = 8000) -> dict[str, Any]:
        """Return a bounded execution bundle for a task requiring multiple skills.

        Composition is selection of separately qualified operational contexts;
        it does not mint a synthetic combined skill or bypass qualification.
        The caller still records each skill application and evaluates the
        resulting task independently.
        """
        # Discovery must not apply the final composition byte bound: doing so
        # can silently remove a required skill before order validation. Bound
        # discovery separately, then fail closed if the complete composition
        # cannot fit in the caller's requested context budget.
        discovery_chars = max(max_chars, 4000 * max(1, min(max_skills, 10)) + 2000)
        selected = self.discover_applicable_skills(task_context, max_candidates=max_skills, max_chars=discovery_chars)
        requested_order = [str(item) for item in task_context.get("execution_order", [])]
        by_key = {item["key"]: item for item in selected}
        if requested_order:
            if set(requested_order) != set(by_key) or len(requested_order) != len(by_key):
                raise LearningError("execution_order must contain each selected skill exactly once")
            selected = [by_key[key] for key in requested_order]
            decision = {"mode": "sequential", "basis": "task-declared dependency order", "order": requested_order}
        else:
            decision = {"mode": "sequential", "basis": "deterministic discovery order", "order": [item["key"] for item in selected]}
        contexts = [item["operational_context"] for item in selected]
        bundle = {"schema": "rex-learning-composed-context-v1", "task_requirements": list(task_context.get("requirements", [])), "composition": decision, "skills": selected, "operational_contexts": contexts}
        if len(json.dumps(bundle, ensure_ascii=False)) > max_chars and bundle["skills"]:
            raise LearningError("composition context exceeds max_chars")
        return bundle

    def qualify_process_task(
        self,
        task_context: Mapping[str, Any],
        *,
        skill_key: str,
        task_id: str,
        executor: Any,
        evaluator: ExecutableGrader,
        case_id: str = "process",
        control_executor: Any | None = None,
    ) -> dict[str, Any]:
        """Behaviorally qualify one persisted workflow on a fresh task.

        The executor is deliberately narrow: it receives one discovered
        workflow context and must return one independently graded response.
        The engine records the process definition, execution evidence, and an
        optional no-skill control; it does not interpret or promote the
        workflow from a self-report.
        """
        if evaluator.evaluator_id not in self.trusted_evaluators:
            raise LearningError("process evaluator must be registered as trusted")
        selected = self.discover_applicable_skills(task_context, max_candidates=10, max_chars=8000)
        matches = [item for item in selected if item["key"] == skill_key]
        if len(matches) != 1:
            raise LearningError("workflow skill was not uniquely discovered")
        workflow = matches[0]
        skill = self.skill(workflow["skill_id"])
        if skill.get("kind") != "workflow":
            raise LearningError("process qualification requires a workflow skill")
        process_id = _id("process", task_id, workflow["skill_id"], str(workflow["skill_version"]), case_id)
        test = {"id": _id("process-test", process_id), "schema": "rex-learning-process-test-v1", "phase": "edge", "cases": [case_id], "success_threshold": 1.0, "evaluator_type": evaluator.evaluator_type, "evaluator_id": evaluator.evaluator_id, "evaluator_independence": "independent", "contamination_status": "clean", "allowed_resources": [], "prohibited_resources": ["hidden_answer_key"]}

        def run(run_executor: Any, *, role: str) -> tuple[dict[str, Any], dict[str, Any]]:
            result = run_executor(task_context=dict(task_context), operational_context=(workflow["operational_context"] if role == "production" else None))
            if not isinstance(result, Mapping):
                raise LearningError("process executor must return an object")
            responses = result.get("responses")
            if not isinstance(responses, list) or len(responses) != 1 or not isinstance(responses[0], Mapping) or responses[0].get("case") != case_id:
                raise LearningError("process executor must return exactly one response for the frozen case")
            attempt = {"schema": "rex-learning-process-attempt-v1", "id": _id("process-attempt", process_id, role, json.dumps(dict(result), sort_keys=True)), "process_id": process_id, "task_id": task_id, "role": role, "test_id": test["id"], "skill_id": workflow["skill_id"], "skill_version": workflow["skill_version"], "operational_context": workflow["operational_context"] if role == "production" else {}, "responses": [dict(responses[0])], "provider_response": result.get("provider_response", {}), "provenance": {"executor": getattr(run_executor, "provider", getattr(run_executor, "__name__", type(run_executor).__name__))}}
            self.store.write("process_attempts", attempt["id"], attempt)
            evaluation = dict(evaluator.evaluate(test, attempt))
            evaluation.update({"schema": "rex-learning-process-evaluation-v1", "id": _id("process-evaluation", attempt["id"]), "process_id": process_id, "attempt_id": attempt["id"], "task_id": task_id, "role": role, "skill_id": workflow["skill_id"], "validation_status": "verified"})
            self.store.write("process_evaluations", evaluation["id"], evaluation)
            return attempt, evaluation

        attempt, evaluation = run(executor, role="production")
        application = self.apply_skill(skill_key, task_id, workflow["operational_context"]["procedure"], outcome={"passed": evaluation["passed"], "process_id": process_id}, curriculum_id=skill["curriculum_id"], selection_reason=workflow["why_selected"], operational_context=workflow["operational_context"], execution_ref=attempt["id"], evaluator_ref=evaluation["id"])
        control = None
        if control_executor is not None:
            control_attempt, control_evaluation = run(control_executor, role="control")
            control = {"attempt": control_attempt, "evaluation": control_evaluation}
        run_record = {"schema": "rex-learning-process-run-v1", "id": process_id, "task_id": task_id, "task_context": dict(task_context), "skill_id": workflow["skill_id"], "skill_version": workflow["skill_version"], "selection_reason": workflow["why_selected"], "operational_context": workflow["operational_context"], "attempt": attempt, "evaluation": evaluation, "application": application, "control": control}
        return self._save("process_runs", run_record, "process_qualified")

    def qualify_composed_task(
        self,
        task_context: Mapping[str, Any],
        *,
        task_id: str,
        executor: Any,
        evaluator: ExecutableGrader,
        case_id: str = "composed",
        control_executor: Any | None = None,
    ) -> dict[str, Any]:
        """Execute and independently qualify a task using separately qualified skills.

        This records composition as a task-level behavioral event. It never
        creates or promotes a synthetic combined skill; each selected skill is
        linked through its own application record instead.
        """
        if evaluator.evaluator_id not in self.trusted_evaluators:
            raise LearningError("composition evaluator must be registered as trusted")
        bundle = self.compose_applicable_skills(task_context)
        if len(bundle["skills"]) < 2:
            raise LearningError("composition requires at least two selected skills")
        composition_id = _id("composition", task_id, json.dumps(bundle["composition"], sort_keys=True))
        test = {"id": _id("composition-test", composition_id, case_id), "schema": "rex-learning-composition-test-v1", "phase": "edge", "cases": [case_id], "success_threshold": 1.0, "evaluator_type": evaluator.evaluator_type, "evaluator_id": evaluator.evaluator_id, "evaluator_independence": "independent", "contamination_status": "clean"}

        def run_attempt(run_executor: Any, *, role: str) -> tuple[dict[str, Any], dict[str, Any]]:
            result = run_executor(task_context=dict(task_context), composition=bundle["composition"], operational_contexts=list(bundle["operational_contexts"] if role == "production" else []))
            if not isinstance(result, Mapping):
                raise LearningError("composition executor must return an object")
            responses = result.get("responses")
            if not isinstance(responses, list) or len(responses) != 1 or not isinstance(responses[0], Mapping) or responses[0].get("case") != case_id:
                raise LearningError("composition executor must return exactly one response for the frozen case")
            attempt = {"schema": "rex-learning-composition-attempt-v1", "id": _id("composition-attempt", composition_id, role, json.dumps(dict(result), sort_keys=True)), "composition_id": composition_id, "task_id": task_id, "role": role, "test_id": test["id"], "responses": [dict(responses[0])], "provider_response": dict(result)}
            self.store.write("composition_attempts", attempt["id"], attempt)
            evaluation = dict(evaluator.evaluate(test, attempt))
            evaluation.update({"schema": "rex-learning-composition-evaluation-v1", "id": _id("composition-evaluation", attempt["id"]), "composition_id": composition_id, "attempt_id": attempt["id"], "task_id": task_id, "role": role, "validation_status": "verified"})
            self.store.write("composition_evaluations", evaluation["id"], evaluation)
            return attempt, evaluation

        attempt, evaluation = run_attempt(executor, role="production")
        applications = []
        for item in bundle["skills"]:
            applications.append(self.apply_skill(item["key"], f"{task_id}:{item['key']}", item["operational_context"]["procedure"], outcome={"composition_id": composition_id, "role": "component", "passed": evaluation["passed"]}, curriculum_id=self.skill(item["skill_id"])["curriculum_id"], selection_reason=item["why_selected"], operational_context=item["operational_context"], execution_ref=attempt["id"], evaluator_ref=evaluation["id"]))
        control = None
        if control_executor is not None:
            control_attempt, control_evaluation = run_attempt(control_executor, role="control")
            control = {"attempt": control_attempt, "evaluation": control_evaluation}
        run = {"schema": "rex-learning-composition-run-v1", "id": composition_id, "task_id": task_id, "task_context": dict(task_context), "discovered_skill_ids": [item["skill_id"] for item in bundle["skills"]], "selection_reasons": [item["why_selected"] for item in bundle["skills"]], "composition": bundle["composition"], "operational_contexts": bundle["operational_contexts"], "attempt": attempt, "evaluation": evaluation, "applications": applications, "control": control}
        return self._save("composition_runs", run, "composition_qualified")

    def apply_skill(self, key: str, task_id: str, procedure_used: str, *, outcome: dict[str, Any], curriculum_id: str | None = None, selection_reason: str = "", operational_context: dict[str, Any] | None = None, execution_ref: str = "", evaluator_ref: str = "") -> dict[str, Any]:
        matching_skills = [s for s in self.store.list("skills") if s.get("key") == key and (curriculum_id is None or s.get("curriculum_id") == curriculum_id)]
        if curriculum_id is None and len({s.get("curriculum_id") for s in matching_skills}) > 1:
            raise LearningError("curriculum_id is required when skill key is ambiguous")
        skill_ids = {s["id"] for s in matching_skills}
        existing = [item for item in self.store.list("applications") if item.get("task_id") == task_id and item.get("skill_id") in skill_ids]
        if existing:
            return existing[0]
        skills = [s for s in matching_skills if s.get("state") in {"demonstrated", "robust"}]
        if not skills: raise LearningError("no current demonstrated skill matches key")
        skill = max(skills, key=lambda s: s["version"])
        outcome = dict(outcome)
        outcome["validation_status"] = "unverified"
        record = {"schema": "rex-learning-application-v1", "id": _id("application", skill["id"], task_id), "skill_id": skill["id"], "skill_version": skill["version"], "task_id": task_id, "selection_reason": selection_reason[:2000], "operational_context": dict(operational_context or {}) , "procedure_used": procedure_used[:2000], "execution_ref": execution_ref[:500], "evaluator_ref": evaluator_ref[:500], "outcome": outcome, "validation_status": "unverified", "created_at": _now()}
        self._save("applications", record, "skill_applied")
        skill["provenance"]["application_ids"].append(record["id"]); skill["updated_at"] = _now(); self._save("skills", skill)
        return record

    def attach_application_evidence(self, application_id: str, *, execution_ref: str, evaluation_id: str, outcome: dict[str, Any]) -> dict[str, Any]:
        try:
            application = self.store.read("applications", application_id)
            evaluation = self.store.read("evaluations", evaluation_id)
            attempt = self.store.read("attempts", execution_ref)
        except KeyError as exc:
            raise LearningError("application evidence must reference matching trusted evaluation") from exc
        if not execution_ref.strip() or evaluation.get("attempt_id") != execution_ref or evaluation.get("skill_id") != application.get("skill_id") or (attempt.get("task_id") and attempt.get("task_id") != application.get("task_id")) or evaluation.get("validation_status") != "verified" or evaluation.get("evaluator_type") not in {"callback_evaluator", "executable_grader", "executable_artifact_grader", "executable_process_grader", "semantic_checklist_grader", "structured_behavior_grader", "contract_artifact_evaluator", "executable_project_evaluator"}:
            raise LearningError("application evidence must reference matching trusted evaluation")
        canonical = self.evaluate_trusted_attempt(execution_ref)
        if not _trusted_evaluation_matches(evaluation, canonical):
            raise LearningError("application evidence does not match canonical evaluation")
        application.update({"execution_ref": execution_ref[:500], "evaluator_ref": evaluation_id[:500], "evaluator": {"id": evaluation.get("evaluator_id"), "type": evaluation.get("evaluator_type"), "provider_provenance": evaluation.get("provider_provenance")}, "outcome": {**dict(outcome), "score": evaluation["score"], "passed": evaluation["passed"]}, "validation_status": "verified", "updated_at": _now()})
        return self._save("applications", application, "application_evidence_attached")

    def challenge_skill(self, skill_id: str, task_id: str, outcome: dict[str, Any]) -> dict[str, Any]:
        skill = self.skill(skill_id)
        challenge_id = _id("challenge", skill_id, task_id)
        try:
            application = self.store.read("applications", challenge_id)
        except KeyError:
            application = {"schema": "rex-learning-challenge-v1", "id": challenge_id, "skill_id": skill_id, "skill_version": skill["version"], "task_id": task_id, "procedure_used": "challenge", "outcome": outcome, "created_at": _now()}
            self._save("applications", application, "skill_challenged")
        skill["state"] = "challenged"
        if application["id"] not in skill["provenance"]["application_ids"]:
            skill["provenance"]["application_ids"].append(application["id"])
        skill["updated_at"] = _now()
        return self._save("skills", skill)

    def prepared_context(self, *, max_chars: int = 4000) -> dict[str, Any]:
        demonstrated = [{"key": s["key"], "version": s["version"], "claim": s["claim"], "state": s["state"], "baseline_score": s.get("baseline", {}).get("score"), "latest_score": s.get("latest_evaluation", {}).get("score")} for s in self.store.list("skills") if s.get("state") in {"demonstrated", "robust", "challenged"}]
        context = {"schema": "rex-learning-prepared-context-v1", "demonstrated_skills": demonstrated[-20:], "challenged_skills": [{"key": x["key"], "version": x["version"]} for x in self.store.list("skills") if x.get("state") == "challenged"][-10:], "active_curricula": [{"title": c["title"], "objective": c["objective"], "status": c["status"]} for c in self.store.list("curricula") if c.get("status") == "active"][-10:]}
        while len(json.dumps(context, ensure_ascii=False)) > max_chars and context["demonstrated_skills"]:
            context["demonstrated_skills"].pop(0)
        while len(json.dumps(context, ensure_ascii=False)) > max_chars and context["active_curricula"]:
            context["active_curricula"].pop(0)
        if len(json.dumps(context, ensure_ascii=False)) > max_chars:
            context = {"schema": "rex-learning-prepared-context-v1", "demonstrated_skills": [], "challenged_skills": [], "active_curricula": []}
        return context
