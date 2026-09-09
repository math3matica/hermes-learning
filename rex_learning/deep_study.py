from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from typing import Any, Mapping

from .engine import LearningError
from .semantic_evaluator import provenance_record, validate_judgment
from .store import LearningStore
from .instructional_abstraction import CurriculumIntent, InstructionalAbstraction


STUDY_STATES = (
    "discovered", "ingested", "inspected", "queued", "close_reading", "retrieval_due",
    "provisionally_understood", "needs_reread", "integrated", "practice_designed",
    "practiced", "tested", "complete",
)


def _now() -> float:
    return time.time()


def _id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()[:20]
    return f"{prefix}-{digest}"


class DeepStudyEngine:
    """Durable, evidence-gated study lifecycle layered onto LearningStore.

    This is intentionally separate from LearningEngine V1. Existing records are
    historical evidence and are never reinterpreted or mutated by this class.
    """

    def __init__(self, store: LearningStore):
        self.store = store

    def create_curriculum_intent(self, intent: dict[str, Any], *, provenance: dict[str, Any]) -> dict[str, Any]:
        """Persist an immutable, versioned learning objective independently of source units."""
        parsed = CurriculumIntent.from_dict(intent)
        if not provenance or not any(provenance.get(key) for key in ("source", "evidence_refs", "operation_id")):
            raise LearningError("curriculum intent provenance is required")
        intent_id = _id("curriculum-intent", parsed.key, json.dumps(parsed.to_dict(), sort_keys=True))
        record = {"schema": "rex-learning-curriculum-intent-v1", "id": intent_id, "key": parsed.key, "version": 1, "intent": parsed.to_dict(), "provenance": dict(provenance), "created_at": _now()}
        try:
            existing = self.store.read("curriculum_intents", intent_id)
        except KeyError:
            return self._save("curriculum_intents", record, "curriculum_intent_created")
        if existing.get("intent") != record["intent"] or existing.get("provenance") != record["provenance"]:
            raise LearningError("curriculum intent version is immutable")
        return existing

    def record_instructional_abstraction(self, *, intent_id: str, source_id: str, unit_id: str, source_hash: str, abstraction: InstructionalAbstraction | dict[str, Any], provenance: dict[str, Any], prior_abstraction_id: str | None = None) -> dict[str, Any]:
        if not provenance or not all(provenance.get(key) for key in ("operation_id", "input_hash", "output_hash")):
            raise LearningError("abstraction provenance is required")
        parsed = abstraction if isinstance(abstraction, InstructionalAbstraction) else InstructionalAbstraction.from_dict(abstraction)
        record_id = _id("instructional-abstraction", intent_id, source_id, unit_id, source_hash, provenance["output_hash"])
        record = {"schema": "rex-learning-instructional-abstraction-v1", "id": record_id, "curriculum_intent_id": intent_id, "source_id": source_id, "unit_id": unit_id, "source_hash": source_hash, "abstraction": parsed.to_dict(), "provenance": dict(provenance), "prior_abstraction_id": prior_abstraction_id, "created_at": _now()}
        try:
            existing = self.store.read("instructional_abstractions", record_id)
        except KeyError:
            return self._save("instructional_abstractions", record, "instructional_abstraction_recorded")
        if {key: value for key, value in existing.items() if key != "created_at"} != {key: value for key, value in record.items() if key != "created_at"}:
            raise LearningError("abstraction record is immutable")
        return existing

    def record_provider_instructional_abstraction(self, *, intent_id: str, source_id: str, unit_id: str, source_hash: str, raw_provider_answer: dict[str, Any], normalized_provider_answer: dict[str, Any], canonical_abstraction: dict[str, Any], provenance: dict[str, Any], prior_abstraction_id: str | None = None) -> dict[str, Any]:
        """Persist the complete provider-boundary evidence without rewriting V1 records."""
        required_hashes = ("operation_id", "raw_hash", "normalized_hash", "canonical_hash")
        if not provenance or not all(provenance.get(key) for key in required_hashes):
            raise LearningError("provider abstraction provenance is incomplete")
        if canonical_abstraction.get("schema") != "rex-learning-instructional-abstraction-v2":
            raise LearningError("provider abstraction must use the v2 schema")
        record_id = _id("instructional-abstraction-v2", intent_id, source_id, unit_id, source_hash, provenance["canonical_hash"])
        record = {
            "schema": "rex-learning-instructional-abstraction-v2-record",
            "id": record_id,
            "curriculum_intent_id": intent_id,
            "source_id": source_id,
            "unit_id": unit_id,
            "source_hash": source_hash,
            "raw_provider_answer": json.loads(json.dumps(raw_provider_answer, ensure_ascii=False)),
            "normalized_provider_answer": json.loads(json.dumps(normalized_provider_answer, ensure_ascii=False)),
            "canonical_abstraction": json.loads(json.dumps(canonical_abstraction, ensure_ascii=False)),
            "provenance": dict(provenance),
            "prior_abstraction_id": prior_abstraction_id,
            "created_at": _now(),
        }
        try:
            existing = self.store.read("instructional_abstractions", record_id)
        except KeyError:
            return self._save("instructional_abstractions", record, "provider_instructional_abstraction_recorded")
        comparable = {key: value for key, value in record.items() if key != "created_at"}
        existing_comparable = {key: value for key, value in existing.items() if key != "created_at"}
        if existing_comparable != comparable:
            raise LearningError("provider abstraction record is immutable")
        return existing

    def _save(self, collection: str, record: dict[str, Any], event_type: str | None = None, *, run_id: str | None = None) -> dict[str, Any]:
        self.store.write(collection, record["id"], record)
        if event_type:
            event = {
                "schema": "rex-learning-study-event-v1",
                "id": _id("study-event", event_type, record["id"], str(record.get("version", 0))),
                "type": event_type,
                "record_id": record["id"],
                "run_id": run_id or record.get("run_id"),
                "at": _now(),
            }
            self.store.append_event(event)
        return record

    def create_run(self, *, title: str, source_id: str, source_hash: str, method_version: str, provider_policy: dict[str, Any], run_version: str = "v1") -> dict[str, Any]:
        run_id = _id("study-run", title, source_id, source_hash, method_version, run_version)
        try:
            return self.store.read("study_runs", run_id)
        except KeyError:
            pass
        record = {
            "schema": "rex-learning-deep-study-run-v1", "id": run_id,
            "title": title[:240], "source_id": source_id, "source_hash": source_hash,
            "method_version": method_version, "run_version": run_version,
            "provider_policy": dict(provider_policy), "status": "active",
            "checkpoint": {"last_unit_id": None, "last_phase": None},
            "created_at": _now(), "updated_at": _now(),
        }
        return self._save("study_runs", record, "study_run_created", run_id=run_id)

    def run(self, run_id: str) -> dict[str, Any]:
        return self.store.read("study_runs", run_id)

    def checkpoint(self, run_id: str, *, phase: str, unit_id: str | None = None, artifacts: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Persist the last durable work boundary for safe process resumption."""
        if not phase.strip():
            raise LearningError("checkpoint phase is required")
        run = self.run(run_id)
        checkpoint = {"last_unit_id": unit_id, "last_phase": phase, "at": _now()}
        run["checkpoint"] = checkpoint
        if artifacts is not None:
            run["artifacts"] = dict(artifacts)
        run["updated_at"] = _now()
        return self._save("study_runs", run, "study_run_checkpointed", run_id=run_id)

    def create_method_version(self, *, method_id: str, version: str, parent_version: str | None, policy: dict[str, Any], provenance: dict[str, Any]) -> dict[str, Any]:
        record = {
            "schema": "rex-learning-study-method-v1", "id": _id("study-method", method_id, version),
            "method_id": method_id, "version": version, "parent_version": parent_version,
            "policy": dict(policy), "provenance": dict(provenance), "created_at": _now(),
        }
        try:
            existing = self.store.read("study_methods", record["id"])
        except KeyError:
            return self._save("study_methods", record, "study_method_version_created")
        if {key: value for key, value in existing.items() if key != "created_at"} != {key: value for key, value in record.items() if key != "created_at"}:
            raise LearningError("study method version is immutable")
        return existing

    def create_unit(self, run_id: str, *, key: str, kind: str, title: str, parent_key: str | None, source_refs: list[str], source_hash: str) -> dict[str, Any]:
        self.run(run_id)
        if not key.strip() or kind not in {"book", "part", "chapter", "section", "chunk"}:
            raise LearningError("invalid study unit")
        unit_id = _id("study-unit", run_id, key)
        record = {
            "schema": "rex-learning-study-unit-v1", "id": unit_id, "run_id": run_id,
            "key": key, "kind": kind, "title": title[:400], "parent_key": parent_key,
            "source_refs": list(dict.fromkeys(source_refs)), "source_hash": source_hash,
            "state": "discovered", "state_history": [{"state": "discovered", "at": _now(), "reason": "unit_created", "evidence_ids": []}],
            "version": 0, "capability_evidence": [], "uncertainty": [], "open_question_ids": [],
            "created_at": _now(), "updated_at": _now(),
        }
        try:
            existing = self.store.read("study_units", unit_id)
        except KeyError:
            return self._save("study_units", record, "study_unit_created", run_id=run_id)
        if existing.get("source_hash") != source_hash:
            raise LearningError("study unit source identity cannot change")
        if existing.get("source_refs") != record["source_refs"]:
            if record["source_refs"] and not existing.get("source_refs"):
                existing["source_refs"] = record["source_refs"]
                existing["updated_at"] = _now()
                self.store.write("study_units", existing["id"], existing)
            elif record["source_refs"]:
                raise LearningError("study unit source references cannot change")
        return existing

    def unit(self, unit_id: str) -> dict[str, Any]:
        return self.store.read("study_units", unit_id)

    def mark_ingested(self, unit_id: str, *, evidence: dict[str, Any]) -> dict[str, Any]:
        unit = self.unit(unit_id)
        if unit["state"] not in {"discovered", "ingested"}:
            return unit
        if not evidence.get("source_refs"):
            raise LearningError("ingestion requires source references")
        if unit["state"] == "ingested":
            return unit
        return self._transition(unit, "ingested", "deterministic_ingestion", [self._evidence_id(evidence)])

    @staticmethod
    def _evidence_id(evidence: dict[str, Any]) -> str:
        return _id("evidence", json.dumps(evidence, sort_keys=True, ensure_ascii=False))

    def _transition(self, unit: dict[str, Any], state: str, reason: str, evidence_ids: list[str]) -> dict[str, Any]:
        if state not in STUDY_STATES:
            raise LearningError("invalid study state")
        if not evidence_ids and state not in {"discovered", "queued"}:
            raise LearningError(f"{state} requires evidence")
        updated = dict(unit)
        updated["state"] = state
        updated["version"] = int(unit.get("version", 0)) + 1
        updated["state_history"] = list(unit.get("state_history", [])) + [{"state": state, "at": _now(), "reason": reason, "evidence_ids": list(evidence_ids)}]
        updated["updated_at"] = _now()
        return self._save("study_units", updated, "study_unit_state_changed", run_id=unit["run_id"])

    def transition(self, unit_id: str, state: str, *, reason: str = "", evidence_ids: list[str] | None = None) -> dict[str, Any]:
        unit = self.unit(unit_id)
        evidence_ids = list(evidence_ids or [])
        if state == "close_reading":
            if not self._has_exposure(unit_id, "close_reading"):
                raise LearningError("close_reading requires provider exposure")
        if state == "retrieval_due":
            if not self._has_exposure(unit_id, "close_reading"):
                raise LearningError("retrieval_due requires close-reading exposure")
        if state == "complete" and unit["state"] not in {"tested", "complete"}:
            raise LearningError("complete requires tested state")
        if state == unit["state"]:
            return unit
        return self._transition(unit, state, reason or "explicit_transition", evidence_ids)

    def _has_exposure(self, unit_id: str, phase: str) -> bool:
        return any(x.get("unit_id") == unit_id and x.get("phase") == phase for x in self.store.list("study_exposures"))

    def record_exposure(self, run_id: str, unit_id: str, *, phase: str, provider: str, session_id: str, source_refs: list[str], input_hash: str, material_kind: str, output_ref: str, prompt_tokens: int | None = None, output_tokens: int | None = None, operation_provenance: dict[str, Any] | None = None) -> dict[str, Any]:
        unit = self.unit(unit_id)
        if unit["run_id"] != run_id:
            raise LearningError("unit/run mismatch")
        if phase not in {"inspection", "close_reading", "chapter_integration", "part_integration", "book_integration", "reread", "practice", "test"}:
            raise LearningError("invalid study exposure phase")
        exposure_id = _id("study-exposure", run_id, unit_id, phase, input_hash)
        record = {
            "schema": "rex-learning-study-exposure-v1", "id": exposure_id, "run_id": run_id, "unit_id": unit_id,
            "phase": phase, "provider": provider, "session_id": session_id, "source_refs": list(dict.fromkeys(source_refs)),
            "input_hash": input_hash, "material_kind": material_kind, "output_ref": output_ref,
            "prompt_tokens": prompt_tokens, "output_tokens": output_tokens, "operation_provenance": dict(operation_provenance or {}), "created_at": _now(),
        }
        try:
            existing = self.store.read("study_exposures", exposure_id)
        except KeyError:
            return self._save("study_exposures", record, "study_exposure_recorded", run_id=run_id)
        if {key: value for key, value in existing.items() if key != "created_at"} != {key: value for key, value in record.items() if key != "created_at"}:
            raise LearningError("idempotency key reused with different exposure")
        return existing

    def record_retrieval_attempt(self, run_id: str, unit_id: str, *, provider: str, session_id: str, source_free: bool, input_hash: str, score: float, passed: bool, failed_question_ids: list[str], evidence_ref: str) -> dict[str, Any]:
        unit = self.unit(unit_id)
        if unit["run_id"] != run_id:
            raise LearningError("unit/run mismatch")
        if not source_free:
            raise LearningError("retrieval evidence must be source-free")
        if not 0.0 <= float(score) <= 1.0:
            raise LearningError("retrieval score must be between 0 and 1")
        attempt_id = _id("study-retrieval", run_id, unit_id, input_hash)
        record = {
            "schema": "rex-learning-study-retrieval-v1", "id": attempt_id, "run_id": run_id, "unit_id": unit_id,
            "provider": provider, "session_id": session_id, "source_free": True, "input_hash": input_hash,
            "score": float(score), "passed": bool(passed), "failed_question_ids": list(failed_question_ids),
            "evidence_ref": evidence_ref, "created_at": _now(),
        }
        try:
            existing = self.store.read("study_retrievals", attempt_id)
        except KeyError:
            self._save("study_retrievals", record, "study_retrieval_recorded", run_id=run_id)
        else:
            comparable = {key: value for key, value in record.items() if key != "created_at"}
            existing_comparable = {key: value for key, value in existing.items() if key != "created_at"}
            if existing_comparable != comparable:
                raise LearningError("idempotency key reused with different retrieval")
            current = self.unit(unit_id)
            if existing["passed"] and current["state"] == "retrieval_due":
                self._transition(current, "provisionally_understood", "replayed_successful_source_free_retrieval", [attempt_id])
            elif not existing["passed"] and current["state"] in {"retrieval_due", "provisionally_understood"}:
                self._transition(current, "needs_reread", "replayed_failed_source_free_retrieval", [attempt_id])
                self.schedule_reread(run_id, unit_id, reason="retrieval_failure", trigger_id=attempt_id, source_refs=unit["source_refs"])
            return existing
        if passed:
            self._transition(self.unit(unit_id), "provisionally_understood", "successful_source_free_retrieval", [attempt_id])
        else:
            self._transition(self.unit(unit_id), "needs_reread", "retrieval_failure", [attempt_id])
            self.schedule_reread(run_id, unit_id, reason="retrieval_failure", trigger_id=attempt_id, source_refs=unit["source_refs"])
        return record

    def schedule_reread(self, run_id: str, unit_id: str, *, reason: str, trigger_id: str, source_refs: list[str]) -> dict[str, Any]:
        reread_id = _id("study-reread", run_id, unit_id, reason, trigger_id)
        record = {"schema": "rex-learning-study-reread-v1", "id": reread_id, "run_id": run_id, "unit_id": unit_id, "reason": reason, "trigger_id": trigger_id, "source_refs": list(source_refs), "status": "scheduled", "created_at": _now()}
        try:
            existing = self.store.read("study_rereads", reread_id)
        except KeyError:
            return self._save("study_rereads", record, "study_reread_scheduled", run_id=run_id)
        return existing

    def record_relation(self, run_id: str, *, source_unit_id: str, target_unit_id: str, relation: str, evidence_ref: str, provider: str, session_id: str) -> dict[str, Any]:
        if relation not in {"supports", "refines", "contradicts", "depends_on", "exemplifies", "revises"}:
            raise LearningError("invalid study relation")
        source, target = self.unit(source_unit_id), self.unit(target_unit_id)
        if source["run_id"] != run_id or target["run_id"] != run_id:
            raise LearningError("relation/run mismatch")
        relation_id = _id("study-relation", run_id, source_unit_id, target_unit_id, relation, evidence_ref)
        record = {"schema": "rex-learning-study-relation-v1", "id": relation_id, "run_id": run_id, "source_unit_id": source_unit_id, "target_unit_id": target_unit_id, "relation": relation, "evidence_ref": evidence_ref, "provider": provider, "session_id": session_id, "created_at": _now()}
        try:
            existing = self.store.read("study_relations", relation_id)
        except KeyError:
            self._save("study_relations", record, "study_relation_recorded", run_id=run_id)
        else:
            return existing
        if relation in {"refines", "contradicts", "revises"} and target["state"] in {"provisionally_understood", "integrated", "practice_designed", "practiced", "tested"}:
            self._transition(self.unit(target_unit_id), "needs_reread", f"later_{relation}", [relation_id])
            self.schedule_reread(run_id, target_unit_id, reason=f"later_{relation}", trigger_id=relation_id, source_refs=target["source_refs"])
        return record

    def record_conflict(self, run_id: str, unit_id: str, *, claim_ref: str, prior_ref: str, evidence_ref: str, provider: str, session_id: str) -> dict[str, Any]:
        unit = self.unit(unit_id)
        if unit["run_id"] != run_id:
            raise LearningError("unit/run mismatch")
        record = {"schema": "rex-learning-study-conflict-v1", "id": _id("study-conflict", run_id, unit_id, claim_ref, prior_ref), "run_id": run_id, "unit_id": unit_id, "claim_ref": claim_ref, "prior_ref": prior_ref, "evidence_ref": evidence_ref, "provider": provider, "session_id": session_id, "resolution_state": "open", "created_at": _now()}
        try:
            return self.store.read("study_conflicts", record["id"])
        except KeyError:
            return self._save("study_conflicts", record, "study_conflict_recorded", run_id=run_id)

    def create_practice_task(self, run_id: str, unit_id: str, *, claim_ref: str, task_ref: str, source_refs: list[str]) -> dict[str, Any]:
        unit = self.unit(unit_id)
        if unit["run_id"] != run_id:
            raise LearningError("unit/run mismatch")
        task = {"schema": "rex-learning-study-practice-task-v1", "id": _id("study-practice", run_id, unit_id, task_ref), "run_id": run_id, "unit_id": unit_id, "claim_ref": claim_ref, "task_ref": task_ref, "source_refs": list(source_refs), "status": "designed", "created_at": _now()}
        try:
            return self.store.read("study_practice_tasks", task["id"])
        except KeyError:
            self._save("study_practice_tasks", task, "study_practice_task_created", run_id=run_id)
            if unit["state"] in {"provisionally_understood", "integrated"}:
                self._transition(self.unit(unit_id), "practice_designed", "operational_claim_practice_designed", [task["id"]])
            return task

    def record_practice_result(self, run_id: str, task_id: str, *, passed: bool, score: float, provider: str, session_id: str, failure_reason: str = "") -> dict[str, Any]:
        task = self.store.read("study_practice_tasks", task_id)
        if task["run_id"] != run_id or not 0.0 <= float(score) <= 1.0:
            raise LearningError("invalid practice result")
        result = {"schema": "rex-learning-study-practice-result-v1", "id": _id("study-practice-result", task_id, provider, session_id), "run_id": run_id, "task_id": task_id, "unit_id": task["unit_id"], "passed": bool(passed), "score": float(score), "provider": provider, "session_id": session_id, "failure_reason": failure_reason, "created_at": _now()}
        try:
            existing = self.store.read("study_practice_results", result["id"])
        except KeyError:
            self._save("study_practice_results", result, "study_practice_result_recorded", run_id=run_id)
        else:
            return existing
        if passed:
            self._transition(self.unit(task["unit_id"]), "practiced", "practice_passed", [result["id"]])
        else:
            self._transition(self.unit(task["unit_id"]), "needs_reread", "practice_failure", [result["id"]])
            self.schedule_reread(run_id, task["unit_id"], reason="practice_failure", trigger_id=result["id"], source_refs=task["source_refs"])
        return result

    def record_test_result(self, run_id: str, unit_id: str, *, test_kind: str, unseen: bool, passed: bool, score: float, provider: str, session_id: str, evidence_ref: str, evaluator_class: str = "provider_structural", failure_reason: str = "") -> dict[str, Any]:
        unit = self.unit(unit_id)
        if unit["run_id"] != run_id or test_kind not in {"retest", "transfer", "integration_check"} or not unseen:
            raise LearningError("invalid unseen study test")
        if not 0.0 <= float(score) <= 1.0:
            raise LearningError("test score must be between 0 and 1")
        record = {"schema": "rex-learning-study-test-v1", "id": _id("study-test", run_id, unit_id, test_kind, evidence_ref), "run_id": run_id, "unit_id": unit_id, "test_kind": test_kind, "unseen": True, "passed": bool(passed), "score": float(score), "provider": provider, "session_id": session_id, "evidence_ref": evidence_ref, "evaluator_class": evaluator_class, "failure_reason": failure_reason, "capability_evidence": evaluator_class == "independent_external", "created_at": _now()}
        try:
            existing = self.store.read("study_tests", record["id"])
        except KeyError:
            self._save("study_tests", record, "study_test_recorded", run_id=run_id)
        else:
            return existing
        if passed:
            self._transition(self.unit(unit_id), "tested", f"unseen_{test_kind}_passed", [record["id"]])
        else:
            self._transition(self.unit(unit_id), "needs_reread", f"unseen_{test_kind}_failed", [record["id"]])
            self.schedule_reread(run_id, unit_id, reason=f"{test_kind}_failure", trigger_id=record["id"], source_refs=unit["source_refs"])
        return record

    def record_semantic_evaluation(self, run_id: str, unit_id: str, *, test_ref: str, attempt_id: str, learner: dict[str, Any], evaluator: dict[str, Any], judgment: dict[str, Any], contamination: dict[str, Any], protocol_ref: str, remediation: dict[str, Any] | None = None) -> dict[str, Any]:
        """Persist an independent semantic judgment without changing study state."""
        unit = self.unit(unit_id)
        if unit["run_id"] != run_id or not test_ref.strip() or not attempt_id.strip() or not protocol_ref.strip():
            raise LearningError("invalid semantic evaluation identity")
        if not isinstance(learner, dict) or not isinstance(evaluator, dict):
            raise LearningError("semantic evaluation provenance is required")
        for record, label in ((learner, "learner"), (evaluator, "evaluator")):
            if not record.get("provider") or not record.get("session_id"):
                raise LearningError(f"{label} provenance requires provider and session_id")
        if contamination.get("status") not in {"clean", "contaminated", "unknown", "not_applicable"}:
            raise LearningError("invalid semantic contamination status")
        normalized = validate_judgment(judgment)
        evaluation_id = _id("study-semantic", run_id, unit_id, test_ref, attempt_id)
        record = provenance_record(
            learner=learner, evaluator=evaluator, judgment=normalized,
            contamination=contamination, remediation=remediation,
        )
        record.update({"id": evaluation_id, "run_id": run_id, "unit_id": unit_id,
                       "test_ref": test_ref, "attempt_id": attempt_id,
                       "protocol_ref": protocol_ref, "created_at": _now()})
        try:
            existing = self.store.read("study_semantic_evaluations", evaluation_id)
        except KeyError:
            return self._save("study_semantic_evaluations", record, "study_semantic_evaluation_recorded", run_id=run_id)
        comparable = {key: value for key, value in record.items() if key != "created_at"}
        existing_comparable = {key: value for key, value in existing.items() if key != "created_at"}
        if existing_comparable != comparable:
            raise LearningError("semantic evaluation record is immutable")
        return existing

    def progress(self, run_id: str) -> dict[str, Any]:
        units = [u for u in self.store.list("study_units") if u.get("run_id") == run_id]
        counts = Counter(u["state"] for u in units)
        return {"run_id": run_id, "total_units": len(units), "counts": {state: counts.get(state, 0) for state in STUDY_STATES}, "next_units": [u["id"] for u in units if u["state"] in {"queued", "close_reading", "retrieval_due", "needs_reread"}][:10]}

    def human_progress(self, run_id: str) -> str:
        run = self.run(run_id)
        progress = self.progress(run_id)
        counts = progress["counts"]
        providers = Counter(x.get("provider") for x in self.store.list("study_exposures") if x.get("run_id") == run_id)
        provider_text = ",".join(f"{key}={value}" for key, value in sorted(providers.items()) if key)
        return (f"[Deep Study] {run['title']}\n"
                f"units={progress['total_units']} state={run['status']} "
                f"ingested={counts['ingested']} inspected={counts['inspected']} queued={counts['queued']} "
                f"close_reading={counts['close_reading']} "
                f"understood={counts['provisionally_understood']} reread_due={counts['needs_reread']} "
                f"integrated={counts['integrated']} practiced={counts['practiced']} tested={counts['tested']} "
                f"complete={counts['complete']}\nprovider={provider_text or 'none'}")

    def complete_unit(self, unit_id: str, *, evidence_ids: list[str]) -> dict[str, Any]:
        unit = self.unit(unit_id)
        if unit["state"] != "tested":
            raise LearningError("only tested units can complete")
        return self._transition(unit, "complete", "evidence_backed_unit_completion", evidence_ids)
