from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

from .engine import LearningError
from .store import LearningStore


INDEPENDENCE_LEVELS = {
    "deterministic",
    "same_provider_fresh_context",
    "same_model_isolated_context",
    "different_model_same_provider",
    "different_provider",
    "externally_executable",
    "human_reviewed",
}


@dataclass(frozen=True)
class CognitiveContextPolicy:
    isolation: str = "fresh"
    source_visible: bool = False
    hidden_reference_visible: bool = False
    learner_answer_visible: bool = False
    prior_answer_visible: bool = False
    persisted_skill_visible: bool = False
    learner_context_visible: bool = False
    remediation_visible: bool = False
    curriculum_intent_visible: bool = False

    @classmethod
    def for_operation(cls, operation_type: str) -> "CognitiveContextPolicy":
        if operation_type in {"close_read", "inspect_source", "classify_source", "structural_map", "instructional_abstraction", "integrate", "corrective_study", "consolidate", "study_and_consolidate"}:
            return cls(source_visible=True, learner_context_visible=True, persisted_skill_visible=True, curriculum_intent_visible=operation_type == "instructional_abstraction")
        if operation_type in {"retrieve", "transfer_test", "applicability_test", "retention_test", "attempt_task", "fresh_application"}:
            return cls(persisted_skill_visible=True)
        if operation_type == "semantic_evaluation":
            return cls(hidden_reference_visible=True, learner_answer_visible=True)
        return cls()

    def as_dict(self) -> dict[str, Any]:
        return {
            "isolation": self.isolation,
            "source_visible": self.source_visible,
            "hidden_reference_visible": self.hidden_reference_visible,
            "learner_answer_visible": self.learner_answer_visible,
            "prior_answer_visible": self.prior_answer_visible,
            "persisted_skill_visible": self.persisted_skill_visible,
            "learner_context_visible": self.learner_context_visible,
            "remediation_visible": self.remediation_visible,
            "curriculum_intent_visible": self.curriculum_intent_visible,
        }


@dataclass(frozen=True)
class ProviderPolicy:
    routes: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def resolve(self, operation_type: str) -> Mapping[str, Any]:
        route = self.routes.get(operation_type, self.routes.get("default"))
        if not route or not route.get("provider"):
            raise LearningError(f"no cognitive provider policy for {operation_type}")
        return route


@dataclass(frozen=True)
class CognitiveRequest:
    operation_id: str
    operation_type: str
    provider: str
    model: str
    inputs: Mapping[str, Any]
    context_policy: CognitiveContextPolicy
    parent_operation_ids: tuple[str, ...] = ()


class CognitiveProvider(Protocol):
    provider_id: str
    model_id: str

    def invoke(self, request: CognitiveRequest) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class CognitiveResult:
    output: dict[str, Any]
    provenance: dict[str, Any]


class CognitiveRouter:
    """Provider-neutral cognitive operation dispatcher owned by Hermes."""

    def __init__(self, store: LearningStore, *, providers: Mapping[str, CognitiveProvider], policy: ProviderPolicy):
        self.store = store
        self.providers = dict(providers)
        self.policy = policy

    @staticmethod
    def _hash(value: Any) -> str:
        return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()

    @staticmethod
    def _validate_context(operation_type: str, inputs: Mapping[str, Any], context: CognitiveContextPolicy) -> None:
        if operation_type in {"retrieve", "transfer_test", "applicability_test", "retention_test", "attempt_task", "fresh_application"}:
            if context.source_visible or context.hidden_reference_visible or context.prior_answer_visible:
                raise LearningError(f"{operation_type} must be source-free and reference-free")
        if operation_type == "semantic_evaluation":
            if not context.learner_answer_visible or not context.hidden_reference_visible or context.prior_answer_visible:
                raise LearningError("semantic evaluation requires learner answer and hidden reference without prior answer")
        if not context.source_visible and any(key in inputs for key in ("source", "material", "source_text")):
            raise LearningError("context policy forbids source material for this operation")
        if not context.hidden_reference_visible and any(key in inputs for key in ("hidden_reference", "answer_key", "rubric")):
            raise LearningError("context policy forbids hidden evaluation reference for this operation")
        if not context.persisted_skill_visible and "persisted_skill" in inputs:
            raise LearningError("context policy forbids persisted skill for this operation")

    def run(
        self,
        operation_type: str,
        inputs: Mapping[str, Any],
        *,
        context: CognitiveContextPolicy | None = None,
        evidence_class: str = "same_provider_fresh_context",
        parent_operation_ids: tuple[str, ...] = (),
        _route_override: Mapping[str, Any] | None = None,
        _escalated_from: str | None = None,
    ) -> CognitiveResult:
        context = context or CognitiveContextPolicy.for_operation(operation_type)
        self._validate_context(operation_type, inputs, context)
        if evidence_class not in INDEPENDENCE_LEVELS:
            raise LearningError("invalid cognitive evidence class")
        route = _route_override or self.policy.resolve(operation_type)
        provider_id = str(route["provider"])
        provider = self.providers.get(provider_id)
        if provider is None:
            raise LearningError(f"cognitive provider is not registered: {provider_id}")
        operation_id = f"cognitive-{uuid.uuid4().hex}"
        request = CognitiveRequest(
            operation_id=operation_id,
            operation_type=operation_type,
            provider=provider_id,
            model=str(route.get("model", getattr(provider, "model_id", ""))),
            inputs=dict(inputs),
            context_policy=context,
            parent_operation_ids=parent_operation_ids,
        )
        started = time.time()
        try:
            output = dict(provider.invoke(request))
        except Exception as exc:
            record = {
                "schema": "rex-learning-cognitive-operation-v1", "id": operation_id,
                "operation_type": operation_type, "provider": provider_id, "model": request.model,
                "status": "failed", "error": str(exc), "input_hash": self._hash(inputs),
                "context_policy": context.as_dict(), "evidence_class": evidence_class,
                "parent_operation_ids": list(parent_operation_ids), "created_at": started,
            }
            self.store.write("cognitive_operations", operation_id, record)
            fallback = self.policy.routes.get("fallback")
            if fallback and fallback.get("provider") and fallback.get("provider") != provider_id:
                result = self.run(
                    operation_type, inputs, context=context, evidence_class=evidence_class,
                    parent_operation_ids=parent_operation_ids + (operation_id,),
                    _route_override=fallback, _escalated_from=provider_id,
                )
                result.provenance["escalated_from"] = provider_id
                fallback_record = self.store.read("cognitive_operations", result.provenance["operation_id"])
                fallback_record["escalated_from"] = provider_id
                self.store.write("cognitive_operations", result.provenance["operation_id"], fallback_record)
                return result
            raise LearningError(f"cognitive provider operation failed: {exc}") from exc
        output_hash = self._hash(output)
        provenance = {
            "operation_id": operation_id, "operation_type": operation_type,
            "provider": provider_id, "model": request.model,
            "request_id": operation_id, "session_id": str(route.get("session_id", "")),
            "input_hash": self._hash(inputs), "output_hash": output_hash,
            "context_policy": context.as_dict(), "evidence_class": evidence_class,
            "parent_operation_ids": list(parent_operation_ids), "created_at": started,
        }
        if _escalated_from:
            provenance["escalated_from"] = _escalated_from
        record = {"schema": "rex-learning-cognitive-operation-v1", **provenance, "status": "completed", "output": output}
        self.store.write("cognitive_operations", operation_id, record)
        return CognitiveResult(output=output, provenance=provenance)
