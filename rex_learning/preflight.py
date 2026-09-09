from __future__ import annotations

import hashlib
import json
import re
import shlex
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .capability_discovery import (
    bind_independent_competence_test_design,
    build_competence_test_requests,
    normalize_independent_test_design,
    select_learning_treatments,
)
from .engine import LearningError


def _validate_executable_mutation_fixture(task: Any) -> None:
    """Reject mutation tasks that omit the executable implementation pair."""
    if not isinstance(task, str):
        return
    lowered = task.casefold()
    marker = next(
        (candidate for candidate in ("learner-visible fixture:", "learner-visible starter:") if candidate in lowered),
        None,
    )
    if marker is None:
        return
    body = lowered.split(marker, 1)[1]
    if "mutant" not in body and "mutation" not in body:
        return
    fenced_blocks = len(re.findall(r"```(?:java|kotlin|javascript|typescript|python)\s*.*?```", body, flags=re.S))
    class_labels = re.findall(
        r"(?:original production class|(?:complete\s+)?mutant(?: production)? class(?:\s+[A-Za-z0-9_-]+)?)\s*:",
        body,
    )
    if fenced_blocks < 2 and not class_labels:
        return
    if fenced_blocks >= 2:
        return
    labeled_classes = re.findall(
        r"(?:original production class|(?:complete\s+)?mutant(?: production)? class(?:\s+[A-Za-z0-9_-]+)?)\s*:\s*.*?\b(?:public\s+)?(?:final\s+)?class\s+[a-z_$][\w$]*",
        body,
        flags=re.S,
    )
    if len(labeled_classes) >= 2:
        return
    raise LearningError("executable mutation task lacks a complete original and mutant fixture")


def _materialize_learner_contract_declarations(task: str, contract: Any) -> str:
    """Append Hermes-owned field/constraint declarations when omitted.

    The independent designer may provide a valid machine-readable contract
    without repeating every constraint in prose.  The learner must still see
    those constraints, so preflight derives a neutral declaration from the
    already-bound schema.  This adds no answer, expected result, or grader.
    """
    if not isinstance(task, str) or not isinstance(contract, str):
        return task
    try:
        parsed = json.loads(contract)
    except json.JSONDecodeError:
        return task
    if not isinstance(parsed, Mapping):
        return task
    required = parsed.get("required_fields")
    field_types = parsed.get("field_types")
    assertions = parsed.get("assertions")
    if not isinstance(required, list) or not isinstance(field_types, Mapping) or not isinstance(assertions, list):
        return task
    lowered = task.casefold()
    additions: list[str] = []
    if "required output fields:" not in lowered:
        fields = ", ".join(f"{field} ({field_types.get(field, 'unspecified')})" for field in required)
        additions.append(f"REQUIRED OUTPUT FIELDS: {fields}.")
    if "constraints:" not in lowered:
        constraints: list[str] = []
        for assertion in assertions:
            if not isinstance(assertion, Mapping):
                continue
            path = assertion.get("path")
            operator = assertion.get("operator")
            if not isinstance(path, str) or not isinstance(operator, str):
                continue
            # Expected values are evaluator-only material.  Field types and
            # structural/cardinality constraints are safe to expose; a
            # value_is assertion would hand the learner the answer.
            if operator == "value_is":
                continue
            if operator == "all_unique":
                constraints.append(f"{path} must satisfy all_unique.")
            else:
                constraints.append(f"{path} must satisfy {operator} with value {json.dumps(assertion.get('value'), ensure_ascii=False)}.")
        if constraints:
            additions.append("CONSTRAINTS: " + " ".join(constraints))
    return task if not additions else task.rstrip() + "\n" + "\n".join(additions)


def prepare_independent_designs(
    *,
    discovery: Mapping[str, Any],
    design_provider: Any,
    limit: int | None = 2,
    baseline_scores: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a frozen, provider-neutral independent-design preflight package.

    The discovery packet is the only source of capability selection. The
    provider may author designs, but Hermes validates their shape and lineage;
    it cannot supply answer keys, evaluators, or qualification outcomes.
    """
    if discovery.get("schema") != "rex-learning-capability-discovery-v1":
        raise LearningError("unsupported capability discovery result")
    if not callable(design_provider):
        raise LearningError("design_provider must be callable")

    treatments = select_learning_treatments(
        discovery,
        limit=limit,
        baseline_scores=baseline_scores,
    )
    requests = build_competence_test_requests(
        discovery,
        selected_keys=[str(item["intent_key"]) for item in treatments],
    )
    requests_by_id = {str(item["proposal_id"]): item for item in requests}
    bound_designs: list[dict[str, Any]] = []
    for treatment in treatments:
        proposal_id = str(treatment["proposal_id"])
        request = requests_by_id.get(proposal_id)
        if request is None:
            raise LearningError(f"selected treatment has no independent design request: {proposal_id}")
        supplied = design_provider(dict(request))
        if not isinstance(supplied, Mapping):
            raise LearningError(f"independent design provider returned a non-object for {proposal_id}")
        design = supplied.get("design")
        provenance = supplied.get("designer_provenance")
        if not isinstance(design, Mapping) or not isinstance(provenance, Mapping):
            raise LearningError(f"independent design provider response is incomplete for {proposal_id}")
        normalized_design = normalize_independent_test_design(design)
        # Lineage is Hermes-owned.  A designer need not echo request metadata,
        # but any metadata it does return must still match exactly; bind() is
        # the final fail-closed check for that distinction.
        normalized_design = dict(normalized_design)
        for field in (
            "proposal_id",
            "curriculum_id",
            "curriculum_intent_id",
            "source_id",
            "source_hash",
            "target_capability",
            "skill_kind",
            "source_refs",
        ):
            if field not in normalized_design:
                normalized_design[field] = request[field]
        if normalized_design.get("self_contained_tasks") is True and isinstance(normalized_design.get("phase_tasks"), Mapping):
            contracts = normalized_design.get("evaluation_contracts")
            if isinstance(contracts, Mapping):
                normalized_design["phase_tasks"] = {
                    phase: _materialize_learner_contract_declarations(task, contracts.get(phase))
                    for phase, task in normalized_design["phase_tasks"].items()
                }
        bound_designs.append(
            bind_independent_competence_test_design(
                request=request,
                design=normalized_design,
                designer_provenance=provenance,
            )
        )

    packet = {
        "schema": "rex-learning-independent-design-preflight-v1",
        "discovery_schema": discovery["schema"],
        "discovery_hash": hashlib.sha256(
            json.dumps(discovery, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "treatments": treatments,
        "requests": requests,
        "bound_designs": bound_designs,
        "status": "passed",
    }
    return packet


def validate_pilot_preflight(
    packet: Mapping[str, Any],
    *,
    trusted_evaluators: Mapping[str, Any],
    command_cwd: str | Path = ".",
) -> None:
    """Fail closed when a passed design is not executable by its pilot.

    Design preflight validates provider-authored structure.  Pilot readiness
    additionally requires the declared evaluator to be present in Hermes's
    trusted registry and its configured process command to resolve from the
    pilot working directory.  This prevents a structurally passed package
    from becoming a misleading setup failure at learner exposure time.
    """
    if packet.get("schema") not in {
        "rex-learning-independent-design-preflight-v1",
        "softengbook-autonomous-discovery-design-preflight-v1",
    } or packet.get("status") != "passed":
        raise LearningError("pilot requires a passed independent-design preflight")
    designs = packet.get("bound_designs")
    if not isinstance(designs, list) or not designs:
        raise LearningError("pilot preflight contains no bound designs")
    root = Path(command_cwd)

    def validate_command_support(command: str | tuple[str, ...], *, design_owned: bool) -> None:
        parts = tuple(command) if not isinstance(command, str) else tuple(shlex.split(command))
        for argument in parts[1:]:
            candidate = Path(argument)
            if candidate.suffix in {".py", ".sh", ".bash"}:
                if not candidate.is_absolute():
                    candidate = root / candidate
                if not candidate.exists():
                    suffix = "design execution support file" if design_owned else "pilot preflight evaluator support file"
                    raise LearningError(f"{suffix} is missing")

    for design in designs:
        if not isinstance(design, Mapping):
            raise LearningError("pilot preflight contains an invalid bound design")
        evaluator_id = design.get("evaluator_id")
        if not isinstance(evaluator_id, str) or not evaluator_id.strip():
            raise LearningError("pilot preflight evaluator binding is incomplete")
        evaluator = trusted_evaluators.get(evaluator_id)
        if evaluator is None:
            raise LearningError("pilot preflight evaluator is not registered in Hermes")
        if getattr(evaluator, "evaluator_type", None) != design.get("evaluator_type"):
            raise LearningError("pilot preflight evaluator type does not match Hermes registry")
        if design.get("evaluator_type") == "executable_project_evaluator":
            if design.get("project_contract") != getattr(getattr(evaluator, "contract", None), "to_dict", lambda: None)():
                raise LearningError("pilot preflight project contract does not match Hermes evaluator")
            if design.get("project_contract_sha256") != getattr(getattr(evaluator, "contract", None), "sha256", None):
                raise LearningError("pilot preflight project contract hash does not match Hermes evaluator")
            continue
        command = tuple(str(item) for item in getattr(evaluator, "command", ()) if str(item))
        if not command:
            raise LearningError("pilot preflight evaluator command is empty")
        executable = Path(command[0])
        if not executable.is_absolute():
            executable = root / executable
        if not executable.exists():
            raise LearningError("pilot preflight evaluator executable is missing")
        validate_command_support(command, design_owned=False)
        phase_tasks = design.get("phase_tasks")
        if isinstance(phase_tasks, Mapping):
            for task in phase_tasks.values():
                _validate_executable_mutation_fixture(task)
        mutation_contracts = design.get("mutation_contracts")
        raw_design = design.get("raw_design")
        if mutation_contracts is None and isinstance(raw_design, Mapping):
            mutation_contracts = raw_design.get("mutation_contracts")
        if mutation_contracts is not None and not isinstance(mutation_contracts, Mapping):
            raise LearningError("pilot preflight mutation contracts are invalid")
        for contract in (mutation_contracts or {}).values():
            if not isinstance(contract, Mapping):
                raise LearningError("pilot preflight mutation contract is invalid")
            test_command = contract.get("test_command")
            if isinstance(test_command, str) and test_command.strip():
                raise LearningError("provider-authored execution command is forbidden")