from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any, cast

from .deep_study import DeepStudyEngine
from .engine import CeilingBaselineError, LearningEngine, LearningError
from .learner import Learner, LearnerError


CAPABILITY_KINDS = {"declarative", "procedure", "workflow", "context", "reference"}
_STOPWORDS = {"a", "an", "and", "be", "by", "for", "from", "in", "into", "of", "on", "or", "the", "to", "with"}
_COHERENCE_WEAK_TERMS = _STOPWORDS | {
    "apply", "artifact", "assert", "behavior", "case", "class", "complete", "contract",
    "criterion", "dependency", "expected", "fixture", "input", "output", "public", "result",
    "response", "source", "supplied", "test", "testing", "tests", "use", "using", "value",
    "diagnose", "diagnosis", "identify", "real",
}


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "\n".join(f"{index + 1}. {_text(item)}" for index, item in enumerate(value) if _text(item))
    if isinstance(value, Mapping):
        return "\n".join(f"{key}: {_text(item)}" for key, item in value.items() if _text(item))
    return "" if value is None else str(value).strip()


def _list(value: Any) -> list[str]:
    values = value if isinstance(value, list) else [value]
    return [item for item in (_text(item) for item in values) if item]


def _usable_learning_plan(value: Any) -> bool:
    text = _text(value)
    return bool(text) and len(text.split()) >= 3 and not re.search(r"\[[^\]]+\]|\{[^{}]+\}", text)


def _behavioral_evidence_candidate(value: Any) -> bool:
    """Screen evidence plans for observable and discriminating outcomes.

    This is only a treatment-selection heuristic. Trusted evaluators still
    determine competence, and provider text never becomes evidence by itself.
    """
    text = _text(value).casefold()
    if not _usable_learning_plan(text):
        return False
    observable = ("compile", "run", "test", "assert", "output", "artifact", "reproduce", "diagnose", "execute", "solve", "identify", "identifies", "calculate", "coverage", "measure")
    discriminating = ("held-out", "held out", "novel", "fresh", "expected", "required", "assert", "pass", "fail", "criteria", "correct", "appropriate", "suitable", "inappropriate", "unrelated", "not-applicable", "statement", "branch")
    coverage_discriminating = "coverage" in text and any(
        token in text for token in ("percentage", "percentages", "uncovered path", "uncovered paths")
    )
    return any(token in text for token in observable) and (
        any(token in text for token in discriminating) or coverage_discriminating
    )


def _tokens(value: Any) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9][a-z0-9_-]+", _text(value).casefold()) if token not in _STOPWORDS}


def _coherence_tokens(value: Any) -> set[str]:
    """Return distinctive terms for pre-exposure design coherence checks."""
    tokens = {token for token in _tokens(value) if token not in _COHERENCE_WEAK_TERMS}
    stems = set(tokens)
    for token in tokens:
        if token.endswith("ion") and len(token) > 5:
            stems.add(token[:-3])
        if token.endswith("ed") and len(token) > 4:
            stems.add(token[:-2])
    return stems


def _missing_learner_contract_fields(task: str, contract: Any) -> list[str]:
    """Find required artifact fields omitted from the learner-visible task.

    Independent evaluators may describe a machine-readable contract, but the
    learner is only given the task.  A design that hides required output keys
    behind evaluator metadata is not executable by an otherwise honest learner.
    """
    if not isinstance(contract, str) or not contract.strip():
        return []
    try:
        parsed = json.loads(contract)
    except json.JSONDecodeError:
        return []
    required = parsed.get("required_fields") if isinstance(parsed, Mapping) else None
    if not isinstance(required, list):
        return []
    missing: list[str] = []
    for field in required:
        if not isinstance(field, str) or not field.strip():
            continue
        leaf = field.rsplit(".", 1)[-1].replace("[]", "")
        if leaf and not re.search(rf"(?<![A-Za-z0-9_]){re.escape(leaf)}(?![A-Za-z0-9_])", task):
            missing.append(field)
    return missing


def _missing_learner_contract_requirements(task: str, contract: Any) -> list[str]:
    """Find executable constraints hidden from the learner.

    A field-name list is not enough for an executable contract: the learner
    must also be told when a field is an array, a boolean, has a minimum
    cardinality, or has an exact scalar value. This remains a design validation
    check; it never supplies an answer or expected value.
    """
    missing = _missing_learner_contract_fields(task, contract)
    if not isinstance(contract, str) or not contract.strip():
        return missing
    try:
        parsed = json.loads(contract)
    except json.JSONDecodeError:
        return missing
    if not isinstance(parsed, Mapping):
        return missing
    task_text = task.casefold()
    field_types = parsed.get("field_types")
    if isinstance(field_types, Mapping):
        for field, field_type in field_types.items():
            if not isinstance(field, str) or not isinstance(field_type, str):
                continue
            field_marker = re.escape(field.casefold())
            if field_type.casefold() == "array" and re.search(rf"{field_marker}[^\n]{{0,120}}(?:array|list|items|records|entries)", task_text) is None:
                missing.append(f"$.{field} requires an array value")
            elif field_type.casefold() == "boolean" and re.search(rf"{field_marker}[^\n]{{0,120}}(?:boolean|booleans|true/false|true and false|bool)", task_text) is None:
                missing.append(f"$.{field} requires a boolean value")
    assertions = parsed.get("assertions")
    if isinstance(assertions, list):
        for assertion in assertions:
            if not isinstance(assertion, Mapping):
                continue
            path = assertion.get("path")
            value = assertion.get("value")
            operator = assertion.get("operator")
            if operator == "value_is":
                # Exact evaluator values are intentionally not learner-visible:
                # exposing them would turn the contract into answer scaffolding.
                # The evaluator retains this assertion and checks it against the
                # submitted artifact after the source-free attempt.
                continue
            if operator != "min_items":
                continue
            if not isinstance(path, str) or not isinstance(value, int) or isinstance(value, bool):
                continue
            field = path.rsplit(".", 1)[-1]
            marker = re.escape(field.casefold())
            count_pattern = rf"(?:{marker}[^\n]{{0,120}}(?:at least|minimum(?: of)?)[^\d]{{0,12}}{value}\b|(?:at least|minimum(?: of)?)[^\d]{{0,12}}{value}\b[^\n]{{0,120}}{marker})"
            operator_pattern = rf"(?:{re.escape(path.casefold())}|{marker})[^\n]{{0,120}}\bmin_items\b[^\d]{{0,12}}{value}\b|\bmin_items\b[^\d]{{0,12}}{value}\b[^\n]{{0,120}}{marker}"
            if re.search(count_pattern, task_text) is None and re.search(operator_pattern, task_text) is None:
                missing.append(f"{path} requires an array with at least {value} items")
    return list(dict.fromkeys(missing))


def _contract_artifact_type(contract: Any) -> str | None:
    if not isinstance(contract, str) or not contract.strip():
        return None
    try:
        parsed = json.loads(contract)
    except json.JSONDecodeError:
        return None
    artifact_type = parsed.get("artifact_type") if isinstance(parsed, Mapping) else None
    return artifact_type.strip() if isinstance(artifact_type, str) else None


def _coherence_anchor_tokens(value: Any) -> set[str]:
    """Collapse simple morphological variants into distinct coherence anchors."""
    anchors = set()
    for token in _coherence_tokens(value):
        anchor = token
        for suffix in ("ing", "ion", "ed"):
            if anchor.endswith(suffix) and len(anchor) > len(suffix) + 3:
                anchor = anchor[: -len(suffix)]
                break
        anchors.add(anchor)
    return anchors


def _declared_fixture_capabilities(value: Any) -> list[str]:
    """Extract explicit JSON fixture capability labels for coherence checks."""
    text = _text(value)
    return [match.strip() for match in re.findall(r"[\"']capability[\"']\s*:\s*[\"']([^\"']+)[\"']", text) if match.strip()]


def _coherent_independent_phase(
    *, phase: str, relevance: set[str], phase_text: str
) -> bool:
    """Allow explicit fresh-transfer framing to carry a single anchor."""
    overlap = len(relevance & _coherence_anchor_tokens(phase_text))
    if overlap >= 2:
        return True
    if phase != "fresh" or overlap < 1:
        return False
    text = phase_text.casefold()
    transfer_markers = (
        "fresh", "unseen", "transfer", "same contract", "same capability",
        "earlier cases", "previous cases",
    )
    return any(marker in text for marker in transfer_markers)


def _explicit_ids(value: Any) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {item.strip() for item in value if isinstance(item, str) and item.strip()}


def _baseline_headroom(value: Any) -> float | None:
    """Read an optional discovery-time estimate without treating it as evidence."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if 0.0 <= numeric <= 1.0 else None


def _measured_baseline_headroom(value: Any) -> float | None:
    """Convert a trusted baseline score into measurable room for learning."""
    if isinstance(value, Mapping):
        value = value.get("score")
    score = _baseline_headroom(value)
    return None if score is None else 1.0 - score


def _surface(result: Mapping[str, Any]) -> Mapping[str, Any]:
    responses = result.get("responses")
    if isinstance(responses, list):
        matches = [item for item in responses if isinstance(item, Mapping) and item.get("case") == "capability_discovery"]
        answers: list[Mapping[str, Any]] = [item["answer"] for item in matches if isinstance(item.get("answer"), Mapping)]
        if len(answers) == 1:
            answer = answers[0]
            if "capabilities" in answer:
                return answer
            if _text(answer.get("capability")) or _text(answer.get("claim")):
                return {"capabilities": [answer]}
        if len(answers) > 1 and all(_text(answer.get("capability")) or _text(answer.get("claim")) for answer in answers):
            return {"capabilities": answers}
    return result


RELATIONSHIPS = {"new", "refine", "support", "extend", "contradict", "prerequisite", "context", "reference", "irrelevant"}
_DISCOVERY_PACKET_CHARS = 12000


def _discovery_capabilities(result: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Extract the provider's capability objects without accepting emptiness."""
    surface = _surface(result)
    raw_proposals = surface.get("capabilities")
    if raw_proposals is None and (_text(surface.get("capability")) or _text(surface.get("claim"))):
        raw_proposals = [surface]
    if not isinstance(raw_proposals, list):
        return []
    return [item for item in raw_proposals if isinstance(item, Mapping)]


def _relationship(proposal: Mapping[str, Any], skills: Sequence[Mapping[str, Any]]) -> tuple[str, list[str], dict[str, Any]]:
    kind = str(proposal.get("kind", "procedure"))
    if kind == "context":
        return "context", [], {"method": "capability_kind", "matched_tokens": 0}
    if kind == "reference":
        return "reference", [], {"method": "capability_kind", "matched_tokens": 0}
    proposal_tokens = _tokens(proposal.get("capability") or proposal.get("claim"))
    prerequisite_tokens = _tokens(proposal.get("prerequisites"))
    explicit_contradictions = _explicit_ids(proposal.get("contradicts"))
    explicit_extensions = _explicit_ids(proposal.get("extends"))
    matches: list[tuple[int, Mapping[str, Any]]] = []
    for skill in skills:
        old_tokens = _tokens(" ".join(_list(skill.get("applicability"))) + " " + _text(skill.get("claim")))
        overlap = len(proposal_tokens & old_tokens)
        if overlap >= 2:
            matches.append((overlap, skill))
    if matches:
        matches.sort(key=lambda item: (-item[0], str(item[1].get("id", ""))))
        ids = [str(item[1]["id"]) for item in matches]
        matched_ids = set(ids)
        if explicit_contradictions & matched_ids:
            relationship = "contradict"
            method = "explicit_existing_skill_id"
        elif explicit_extensions & matched_ids:
            relationship = "extend"
            method = "explicit_existing_skill_id"
        elif prerequisite_tokens and any(prerequisite_tokens & _tokens(skill.get("claim")) for _, skill in matches):
            relationship = "prerequisite"
            method = "prerequisite_token_overlap"
        else:
            relationship = "refine" if proposal.get("procedure") or proposal.get("operational_procedure") or proposal.get("source-grounded procedure") or proposal.get("source_grounded_procedure") else "support"
            method = "capability_token_overlap"
        return relationship, ids, {"method": method, "matched_tokens": matches[0][0], "matched_skill_ids": ids}
    if not proposal.get("procedure") and not proposal.get("operational_procedure") and not proposal.get("source-grounded procedure") and not proposal.get("source_grounded_procedure") and not proposal.get("proposed_practice"):
        return "reference", [], {"method": "no_behavioral_treatment", "matched_tokens": 0}
    return "new", [], {"method": "no_repertoire_token_match", "matched_tokens": 0}


def discover_capabilities(
    *,
    engine: LearningEngine,
    learner: Learner,
    source_title: str,
    source_text: str,
    educational_objective: str,
    curriculum_id: str | None = None,
    source_locator: str | None = None,
    source_refs: Sequence[str] | None = None,
    source_id: str | None = None,
    source_hash: str | None = None,
    packet_chars: int = _DISCOVERY_PACKET_CHARS,
) -> dict[str, Any]:
    """Discover source-grounded capabilities without caller-supplied skill labels.

    This is a proposal boundary. It records provider output and repertoire
    relationships, but never treats a plausible proposal as a qualified skill.
    Existing acquisition owns practice, trusted evaluation, promotion, and use.
    """
    if not all(isinstance(value, str) and value.strip() for value in (source_title, source_text, educational_objective)):
        raise LearningError("source title, source text, and educational objective are required")
    if not isinstance(packet_chars, int) or isinstance(packet_chars, bool) or packet_chars < 1:
        raise LearningError("packet_chars must be a positive integer")
    curriculum = engine.store.read("curricula", curriculum_id) if curriculum_id else engine.create_curriculum(source_title, educational_objective)
    if curriculum.get("status") != "active":
        raise LearningError("capability discovery curriculum must be active")
    source_hash = source_hash or hashlib.sha256(source_text.encode()).hexdigest()
    if source_id:
        source = engine.store.read("sources", source_id)
        if source.get("curriculum_id") != curriculum["id"]:
            raise LearningError("capability discovery source/curriculum mismatch")
        if source.get("content_hash") != source_hash:
            raise LearningError("capability discovery source content mismatch")
    else:
        source = engine.add_source(curriculum["id"], source_title, source_locator or f"memory://{source_hash}", content_hash=source_hash)
    curriculum_intent = DeepStudyEngine(engine.store).create_curriculum_intent(
        {
            "key": f"capability-discovery:{curriculum['id']}",
            "title": curriculum["title"],
            "target_capability": educational_objective,
            "competence_level": "discover-and-qualify-relevant-capabilities",
        },
        provenance={
            "source": "capability_discovery",
            "evidence_refs": [curriculum["id"]],
            "operation_id": f"capability-discovery-intent:{curriculum['id']}",
        },
    )
    task = (
        "capability_discovery|case=capability_discovery|Determine what this instructional material is trying to make a competent learner "
        "able to understand, decide, perform, diagnose, construct, or improve. Return exactly one response object with case "
        "capability_discovery. Put an object containing a capabilities list in that response's answer field; do not return one "
        "response per capability and do not repeat the case ID. Propose at most two appropriately scoped capabilities from "
        "this packet, keep every field concise, and emit no prose outside the JSON object. "
        "For each capability, return key, capability, kind (declarative, procedure, or workflow), instructional_purpose, "
        "source-grounded procedure when applicable, prerequisites, limitations, exact_details, generalizable_principle, "
        "supporting_source_refs, proposed_practice, and proposed_evidence. supporting_source_refs must "
        "contain only exact references from the supplied packet. Preserve uncertainty and distinguish context/reference material."
        f"\nEDUCATIONAL OBJECTIVE:\n{educational_objective}"
    )
    prompt_hash = hashlib.sha256(task.encode()).hexdigest()
    attempt_id = "discovery-attempt-" + hashlib.sha256(
        f"{source['id']}|{learner.session_id}|{prompt_hash}".encode()
    ).hexdigest()[:16]
    attempt = {
        "schema": "rex-learning-capability-discovery-attempt-v1",
        "id": attempt_id,
        "status": "started",
        "curriculum_id": curriculum["id"],
        "source_id": source["id"],
        "source_hash": source_hash,
        "provider": learner.provider,
        "session_id": learner.session_id,
        "role": "capability_discoverer",
        "operation": "capability_discovery",
        "prompt_hash": prompt_hash,
    }
    engine.store.write("capability_discovery_attempts", attempt_id, attempt)
    packets = [source_text[index:index + packet_chars] for index in range(0, len(source_text), packet_chars)]
    provider_results: list[dict[str, Any]] = []
    raw_proposals: list[Mapping[str, Any]] = []
    try:
        for packet_index, packet in enumerate(packets):
            packet_task = (
                task
                + f"\nThis is discovery packet {packet_index + 1} of {len(packets)}. Use only this packet's material."
                + "\nAVAILABLE EXACT SOURCE REFERENCES (choose only references that support a capability):\n"
                + "\n".join(f"- {ref}" for ref in (source_refs or []))
            )
            packet_result = learner.answer(task=packet_task, material=packet)
            provider_results.append(packet_result)
            raw_proposals.extend(_discovery_capabilities(packet_result))
            engine.store.write(
                "capability_discovery_attempts", attempt_id,
                {**attempt, "status": "in_progress", "packet_index": packet_index + 1,
                 "packet_count": len(packets), "raw_provider_results": provider_results},
            )
    except LearnerError as exc:
        engine.store.write(
            "capability_discovery_attempts", attempt_id,
            {**attempt, "status": "failed", "packet_index": len(provider_results),
             "packet_count": len(packets), "raw_provider_results": provider_results,
             "error": {"code": exc.code, "message": str(exc), "details": exc.details}},
        )
        raise
    if not raw_proposals:
        raw_surface = {"capabilities": []}
        provider_payload: Any = provider_results[0] if len(provider_results) == 1 else provider_results
        provider_provenance = {
            "provider": learner.provider, "session_id": learner.session_id, "role": "capability_discoverer",
            "operation": "capability_discovery", "prompt_hash": prompt_hash,
            "packet_count": len(packets),
            "packet_hashes": [hashlib.sha256(packet.encode()).hexdigest() for packet in packets],
            "output_hash": hashlib.sha256(json.dumps(provider_payload, sort_keys=True, default=str).encode()).hexdigest(),
        }
        engine.store.write(
            "capability_discovery_attempts", attempt_id,
            {**attempt, "status": "failed", "provider": provider_provenance, "packet_count": len(packets),
             "raw_provider_result": provider_payload, "raw_provider_results": provider_results,
             "raw_surface": raw_surface,
             "error": {"code": "empty_capabilities", "message": "capability discovery response must contain a non-empty capabilities list"}},
        )
        raise LearningError("capability discovery response must contain a non-empty capabilities list", details={"code": "empty_capabilities"})
    provider_result = provider_results[0] if len(provider_results) == 1 else {"packet_results": provider_results}
    provider_provenance = {
        "provider": learner.provider, "session_id": learner.session_id, "role": "capability_discoverer",
        "operation": "capability_discovery", "prompt_hash": prompt_hash,
        "packet_count": len(packets),
        "packet_hashes": [hashlib.sha256(packet.encode()).hexdigest() for packet in packets],
        "output_hash": hashlib.sha256(json.dumps(provider_result, sort_keys=True, default=str).encode()).hexdigest(),
    }
    raw_surface = {"capabilities": [dict(item) for item in raw_proposals]}
    engine.store.write(
        "capability_discovery_attempts", attempt_id,
        {**attempt, "status": "completed", "provider": provider_provenance,
         "raw_provider_result": provider_result, "raw_provider_results": provider_results,
         "raw_surface": raw_surface, "proposal_count": len(raw_proposals)},
    )
    skills = engine.store.list("skills")
    proposal_source_refs = [source["id"], *[ref for ref in (source_refs or []) if ref and ref != source["id"]]]
    proposals: list[dict[str, Any]] = []
    seen_proposal_fingerprints: set[str] = set()
    used_proposal_ids: set[str] = set()
    for index, raw in enumerate(raw_proposals):
        if not isinstance(raw, Mapping):
            raise LearningError("capability proposals must be objects")
        capability = _text(raw.get("capability") or raw.get("claim"))
        if not capability:
            raise LearningError("capability proposal requires a capability")
        key = _text(raw.get("key")) or f"capability-{index + 1}"
        kind = _text(raw.get("kind")) or "procedure"
        if kind not in CAPABILITY_KINDS:
            raise LearningError(f"unsupported capability kind: {kind}")
        relationship, existing_ids, relationship_evidence = _relationship(raw, skills)
        proposal_fingerprint = hashlib.sha256(json.dumps(dict(raw), sort_keys=True, default=str).encode()).hexdigest()
        if proposal_fingerprint in seen_proposal_fingerprints:
            continue
        seen_proposal_fingerprints.add(proposal_fingerprint)
        proposal_id = "capability-" + hashlib.sha256(f"{source['id']}|{key}|{capability}".encode()).hexdigest()[:16]
        if proposal_id in used_proposal_ids:
            proposal_id += "-variant-" + proposal_fingerprint[:16]
        used_proposal_ids.add(proposal_id)
        proposal = {
            "schema": "rex-learning-capability-proposal-v1", "id": proposal_id,
            "curriculum_id": curriculum["id"], "source_id": source["id"], "source_refs": proposal_source_refs,
            "source_hash": source_hash, "key": key, "capability": capability, "kind": kind,
            "curriculum_intent_id": curriculum_intent["id"],
            "supporting_source_refs": [
                ref for ref in _list(raw.get("supporting_source_refs"))
                if ref in proposal_source_refs and ref != source["id"]
            ],
            "instructional_purpose": _text(raw.get("instructional_purpose")),
            "procedure": _text(raw.get("procedure") or raw.get("operational_procedure") or raw.get("source-grounded procedure") or raw.get("source_grounded_procedure")),
            "prerequisites": _list(raw.get("prerequisites")), "constraints": _list(raw.get("constraints")),
            "limitations": _list(raw.get("limitations")), "exact_details": _list(raw.get("exact_details")),
            "generalizable_principle": _text(raw.get("generalizable_principle")), "example_details": _list(raw.get("example_details")),
            "uncertainties": _list(raw.get("uncertainties")), "proposed_practice": _text(raw.get("proposed_practice")),
            "proposed_evidence": _text(raw.get("proposed_evidence")), "relationship": relationship,
            "baseline_headroom": _baseline_headroom(raw.get("baseline_headroom")),
            "existing_skill_ids": existing_ids,
            "relationship_evidence": relationship_evidence,
            "provider_relationship": _text(raw.get("relationship")) if _text(raw.get("relationship")) in RELATIONSHIPS else "",
            "actionable": bool(
                kind in {"declarative", "procedure", "workflow"}
                and _text(raw.get("proposed_practice"))
                and _text(raw.get("proposed_evidence"))
            ),
            "behavioral_evidence_candidate": _behavioral_evidence_candidate(raw.get("proposed_evidence")),
            "raw_proposal": dict(raw), "provider": provider_provenance,
        }
        engine.store.write("capability_proposals", proposal_id, proposal)
        proposals.append(proposal)
    return {
        "schema": "rex-learning-capability-discovery-v1", "curriculum": curriculum, "source": {**source, "text": source_text},
        "curriculum_intent": curriculum_intent,
        "provider": provider_provenance,
        "raw_surface": raw_surface, "proposals": proposals,
        "actionable_proposal_keys": [item["key"] for item in proposals if item["actionable"]],
    }


def discover_capabilities_from_units(
    *,
    engine: LearningEngine,
    learner: Learner,
    source_title: str,
    source_units: Sequence[Mapping[str, Any]],
    educational_objective: str,
    curriculum_id: str | None = None,
    source_id: str | None = None,
    packet_chars: int = _DISCOVERY_PACKET_CHARS,
) -> dict[str, Any]:
    """Discover capabilities from bounded instructional units.

    Unit references are carried alongside the durable source record. The
    source text itself remains an input surface; it is not persisted as a
    trusted claim or qualification evidence.
    """
    if not source_units:
        raise LearningError("at least one source unit is required")
    normalized: list[tuple[str, str, str]] = []
    for unit in source_units:
        if not isinstance(unit, Mapping):
            raise LearningError("source units must be objects")
        unit_id = _text(unit.get("unit_id"))
        title = _text(unit.get("title")) or unit_id
        text = _text(unit.get("text"))
        source_ref = _text(unit.get("source_ref") or unit.get("source_ref_id") or unit_id)
        if not unit_id or not text or not source_ref:
            raise LearningError("source units require unit_id, text, and source_ref")
        normalized.append((title, text, source_ref))
    source_text = "\n\n".join(f"## {title}\n{text}" for title, text, _ in normalized)
    discovery_source_id = source_id
    if source_id is not None:
        if curriculum_id is None:
            raise LearningError("source-unit discovery requires a curriculum for a parent source")
        parent_source = engine.store.read("sources", source_id)
        if parent_source.get("curriculum_id") != curriculum_id:
            raise LearningError("capability discovery source/curriculum mismatch")
        excerpt_hash = hashlib.sha256(source_text.encode()).hexdigest()
        excerpt_locator = f"{parent_source['locator']}::instructional-units::{excerpt_hash}"
        excerpt = engine.add_source(
            curriculum_id,
            source_title,
            excerpt_locator,
            kind="source_excerpt",
            content_hash=excerpt_hash,
        )
        excerpt["parent_source_id"] = source_id
        excerpt["source_refs"] = [ref for _, _, ref in normalized]
        engine.store.write("sources", excerpt["id"], excerpt)
        discovery_source_id = excerpt["id"]
    result = discover_capabilities(
        engine=engine, learner=learner, source_title=source_title, source_text=source_text,
        educational_objective=educational_objective, curriculum_id=curriculum_id,
        source_locator=normalized[0][2], source_refs=[ref for _, _, ref in normalized], source_id=discovery_source_id,
 packet_chars=packet_chars,
 )
    result["unit_refs"] = [ref for _, _, ref in normalized]
    return result


def _treatment_disposition(relationship: str) -> tuple[str, str]:
    if relationship in {"refine", "extend", "support"}:
        return "integrate_existing", "awaiting_existing_skill_integration"
    if relationship == "contradict":
        return "investigate_conflict", "awaiting_conflict_review"
    return "acquire", "awaiting_qualification_dependencies"


def build_learning_treatments(
    discovery: Mapping[str, Any], *, selected_keys: Sequence[str] | None = None
) -> list[dict[str, Any]]:
    """Project actionable discovery proposals into acquisition specifications.

    This is intentionally a planning boundary, not a qualification shortcut:
    graders, reviewer, and learner contexts remain explicit dependencies for
    the existing acquisition engine.
    """
    if discovery.get("schema") != "rex-learning-capability-discovery-v1":
        raise LearningError("unsupported capability discovery result")
    raw_proposals = discovery.get("proposals")
    if not isinstance(raw_proposals, list):
        raise LearningError("capability discovery result must contain proposals")
    allowed = set(selected_keys) if selected_keys is not None else None
    treatments: list[dict[str, Any]] = []
    for proposal in raw_proposals:
        if not isinstance(proposal, Mapping) or not proposal.get("actionable"):
            continue
        key = _text(proposal.get("key"))
        if allowed is not None and key not in allowed:
            continue
        if not _usable_learning_plan(proposal.get("proposed_practice")) or not _usable_learning_plan(proposal.get("proposed_evidence")):
            continue
        relationship = str(proposal["relationship"])
        treatment_mode, treatment_status = _treatment_disposition(relationship)
        if treatment_mode == "acquire" and proposal.get("behavioral_evidence_candidate") is not True:
            continue
        treatments.append({
            "schema": "rex-learning-treatment-v1",
            "proposal_id": str(proposal["id"]),
            "curriculum_id": str(proposal["curriculum_id"]),
            "curriculum_intent_id": str(proposal.get("curriculum_intent_id", "")),
            "source_id": str(proposal["source_id"]),
            "source_refs": list(proposal.get("supporting_source_refs") or proposal.get("source_refs", [])),
            "source_hash": str(proposal["source_hash"]),
            "intent_key": key,
            "target_capability": str(proposal["capability"]),
            "skill_kind": str(proposal["kind"]),
            "relationship": relationship,
            "treatment_mode": treatment_mode,
            "existing_skill_ids": list(proposal.get("existing_skill_ids", [])),
            "procedure": str(proposal.get("procedure") or (proposal["capability"] if proposal.get("kind") == "declarative" else "")),
            "practice_task": str(proposal["proposed_practice"]),
            "evidence_plan": str(proposal["proposed_evidence"]),
            "missing_dependencies": ["grader", "reviewer", "fresh_learner", "control_learner"] if treatment_mode == "acquire" else [],
            "status": treatment_status,
        })
    return treatments


def build_competence_test_requests(
    discovery: Mapping[str, Any], *, selected_keys: Sequence[str] | None = None
) -> list[dict[str, Any]]:
    """Project discovered capabilities into requests for independent test design.

    The provider's practice and evidence suggestions remain explicitly
    untrusted inputs. This function creates no evaluator, answer key, attempt,
    evaluation, or qualification record; a separate trusted test designer must
    satisfy the returned requirements before acquisition can qualify behavior.
    """
    treatments = build_learning_treatments(discovery, selected_keys=selected_keys)
    requests: list[dict[str, Any]] = []
    for treatment in treatments:
        if treatment.get("treatment_mode") != "acquire":
            continue
        requests.append({
            "schema": "rex-learning-test-design-request-v1",
            "discovery_provenance": dict(discovery.get("provider", {})) if isinstance(discovery.get("provider"), Mapping) else {},
            "proposal_id": str(treatment["proposal_id"]),
            "intent_key": str(treatment["intent_key"]),
            "curriculum_id": str(treatment["curriculum_id"]),
            "curriculum_intent_id": str(treatment.get("curriculum_intent_id", "")),
            "source_id": str(treatment["source_id"]),
            "source_refs": list(treatment.get("source_refs", [])),
            "source_hash": str(treatment["source_hash"]),
            "target_capability": str(treatment["target_capability"]),
            "skill_kind": str(treatment["skill_kind"]),
            "procedure": str(treatment.get("procedure", "")),
            "untrusted_proposed_practice": str(treatment["practice_task"]),
            "untrusted_proposed_evidence": str(treatment["evidence_plan"]),
            "status": "awaiting_independent_design",
            "evaluator_requirements": {
                "source_free": True,
                "novel_transfer": True,
                "independent_evaluator": True,
                "negative_applicability": True,
            },
        })
    return requests


_INDEPENDENT_PHASES = ("baseline", "practice", "posttest", "retest", "negative", "fresh", "control")
_BASELINE_SOLUTION_KEYS = frozenset({
    "answer", "answer_key", "correct_answer", "expected", "expected_case_results",
    "expected_result", "formula", "rounding", "solution", "worked_example",
})


def _baseline_has_solution_scaffolding(task: str) -> bool:
    """Reject explicit answer-bearing fixture fields in source-free baselines."""
    marker = "LEARNER-VISIBLE FIXTURE:"
    if marker not in task:
        return False
    try:
        fixture, _ = json.JSONDecoder().raw_decode(task.split(marker, 1)[1].lstrip())
    except json.JSONDecodeError:
        return False
    pending: list[Any] = [fixture]
    while pending:
        value = pending.pop()
        if isinstance(value, Mapping):
            if any(str(key).casefold() in _BASELINE_SOLUTION_KEYS for key in value):
                return True
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
    return bool(re.search(r"(?i)\b(?:formula|decision\s+rule|worked\s+example|correct\s+answer)\s*[:=]", task))


def _task_contains_expected_answer_assertion(task: str) -> bool:
    """Reject evaluator-style expected-value assertions in learner text."""
    return bool(re.search(
        r"(?is)(?:"
        r"\$\.[A-Za-z0-9_.*\[\]-]+\s+must\s+satisfy\s+value_is\s+with\s+value\s+|"
        r"\b(?:requires?|must\s+equal|must\s+be)\s+(?:the\s+)?exact\s+value\s+|"
        r"\b(?:expected|correct\s+answer|answer\s+key)\s*[:=]|"
        r"\b(?:the\s+)?real\s+dependency\s+defect\s+is\b|"
        r"\bthe\s+mocked\s+behavior\s+(?:is|as)\b|"
        r"\bthe\s+equivalent\s+mutant\s+produces\s+the\s+same\s+observable\s+result\b|"
        r"\bmutation\s+observations?\s*:\s*|"
        r"\b(?:the\s+)?(?:boundary|equivalent|unchanged(?:-behavior)?|no-coverage)\s+mutant\s+(?:is\s+)?(?:marked|surviv(?:es|ed)|produces)\b|"
        r"\bthe\s+specification\s+rule\s+is\s+explicit\b|"
        r"\bthe\s+(?:original|real)\s+(?:implementation|class)\s+satisfies\s+(?:the\s+)?(?:stated\s+)?rule\b"
        r")",
        task,
    ))


def _task_contains_conflicting_todo_instructions(task: str) -> bool:
    """Reject learner tasks that both preserve and replace the same TODO."""
    task_text = task.casefold()
    preserves_todo = re.search(r"\b(?:preserve|keep|retain)\b[^.\n]{0,100}\btodo\b", task_text)
    changes_todo = re.search(r"\b(?:replace|remove|delete|complete|fix|resolve)\b[^.\n]{0,100}\btodo\b", task_text)
    return bool(preserves_todo and changes_todo)


def _has_incoherent_implementation_surface(task: str, contract: Mapping[str, Any]) -> bool:
    """Reject designs that demand implementation while forbidding its artifact.

    A decision/rationale response can document a judgment, but it cannot
    independently demonstrate a task whose learner-visible instructions both
    require implementation and explicitly prohibit implementation output.
    This catches an internally contradictory design before learner exposure;
    it does not infer answers or impose a capability-specific output schema.
    """
    task_text = task.casefold()
    implementation_demand = re.search(r"\b(?:implement|implementation|executable)\b", task_text)
    output_prohibition = re.search(
        r"\b(?:without|do not|don't)\b[^.\n]{0,100}\b(?:implementation|code|executable)\s+output\b",
        task_text,
    )
    required = contract.get("required_fields")
    if not implementation_demand or not output_prohibition or not isinstance(required, list):
        return False
    return bool(required) and set(required) <= {"decision", "rationale"}


def normalize_independent_test_design(design: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize the two supported independent-designer envelopes.

    Providers commonly return ``{"phases": [{"phase": ...}, ...]}`` or a
    phase-keyed map, while the Hermes binding record uses phase-keyed maps.
    Accepting either envelope here is a wire-format normalization only: the
    result still requires all seven unique phases, complete learner tasks, and
    executable contracts.
    No answer or expected-result data is synthesized.
    """
    if not isinstance(design, Mapping):
        raise LearningError("independent test design must be an object")
    if "answer_key" in design or "expected_case_results" in design:
        raise LearningError("learner-authored answer scaffolding is not accepted")
    mutation_contracts = design.get("mutation_contracts")
    if isinstance(mutation_contracts, Mapping):
        for contract in mutation_contracts.values():
            if isinstance(contract, Mapping) and isinstance(contract.get("test_command"), str) and contract["test_command"].strip():
                raise LearningError("provider-authored execution command is forbidden")
    phases = design.get("phases")
    if phases is None and all(
        phase in design and isinstance(design[phase], Mapping)
        for phase in _INDEPENDENT_PHASES
    ):
        phases = [dict(design[phase], phase=phase) for phase in _INDEPENDENT_PHASES]
    if isinstance(phases, list):
        for phase in phases:
            if isinstance(phase, Mapping):
                contract = phase.get("mutation_contract")
                if isinstance(contract, Mapping) and isinstance(contract.get("test_command"), str) and contract["test_command"].strip():
                    raise LearningError("provider-authored execution command is forbidden")
    normalized = dict(design)
    if phases is not None:
        if not isinstance(phases, list) or len(phases) != len(_INDEPENDENT_PHASES):
            raise LearningError("independent test design phases must contain exactly seven entries")
        by_name: dict[str, Mapping[str, Any]] = {}
        for item in phases:
            if not isinstance(item, Mapping) or not isinstance(item.get("phase"), str):
                raise LearningError("independent test design phase entry is invalid")
            phase = item["phase"].strip()
            if phase in by_name:
                raise LearningError("independent test design phases must be unique")
            by_name[phase] = item
        if set(by_name) != set(_INDEPENDENT_PHASES):
            raise LearningError("independent test design phases must cover all seven required phases")
        normalized.pop("phases", None)
        normalized["phase_cases"] = {}
        normalized["phase_tasks"] = {}
        normalized["evaluation_contracts"] = {}
        normalized["cases"] = []
        for phase in _INDEPENDENT_PHASES:
            item = by_name[phase]
            case_id = item.get("case_id")
            task = item.get("task", item.get("learner_visible_task", item.get("learner_visible_fixture_task", item.get("learner-visible-task", item.get("learner-visible task")))))
            schema = item.get("artifact_schema")
            if not isinstance(case_id, str) or not case_id.strip() or not isinstance(task, str) or not task.strip() or not isinstance(schema, Mapping):
                raise LearningError(f"{phase} independent test phase is incomplete")
            case_id = case_id.strip()
            if _task_contains_expected_answer_assertion(task):
                raise LearningError(f"{phase} independent test exposes an expected answer in learner-visible task")
            if _task_contains_conflicting_todo_instructions(task):
                raise LearningError(f"{phase} independent test contains contradictory TODO instructions")
            normalized_contract = _normalize_executable_contract(schema, phase)
            if _has_incoherent_implementation_surface(task, normalized_contract):
                raise LearningError(f"{phase} independent test has an incoherent implementation artifact surface")
            normalized["cases"].append(case_id)
            normalized["phase_cases"][phase] = case_id
            normalized["phase_tasks"][phase] = task.strip()
            normalized["evaluation_contracts"][phase] = json.dumps(normalized_contract, sort_keys=True)
    elif "phase_cases" not in design or "phase_tasks" not in design or "evaluation_contracts" not in design:
        raise LearningError("independent test design must contain exactly seven phases")
    baseline_task = normalized.get("phase_tasks", {}).get("baseline")
    for phase, task in normalized.get("phase_tasks", {}).items():
        if isinstance(task, str) and _task_contains_expected_answer_assertion(task):
            raise LearningError(f"{phase} independent test exposes an expected answer in learner-visible task")
        if isinstance(task, str) and _task_contains_conflicting_todo_instructions(task):
            raise LearningError(f"{phase} independent test contains contradictory TODO instructions")
    if isinstance(baseline_task, str) and _baseline_has_solution_scaffolding(baseline_task):
        raise LearningError("independent test baseline contains solution scaffolding")
    if normalized.get("self_contained_tasks") is True:
        negative_contract = normalized.get("evaluation_contracts", {}).get("negative")
        if isinstance(negative_contract, str):
            try:
                negative_contract = json.loads(negative_contract)
            except json.JSONDecodeError:
                # Legacy open-ended designs use prose contracts. They are
                # validated by their evaluator route; the structured
                # applicability-field gate below only applies when Hermes has
                # a machine-readable contract to inspect.
                negative_contract = None
        if isinstance(negative_contract, Mapping):
            required = negative_contract.get("required_fields", [])
            applicability_fields = {str(field).casefold() for field in required if isinstance(field, str)}
            if not any(any(token in field for token in ("applicab", "scope", "relevan", "decision", "classif")) for field in applicability_fields):
                raise LearningError("self-contained independent test negative phase must expose an applicability or scope decision")
    return normalized


def _normalize_executable_contract(raw: Mapping[str, Any], phase: str) -> dict[str, Any]:
    """Convert common equivalent assertion spellings to Hermes's contract form."""
    contract = dict(raw)
    assertions = contract.get("assertions")
    if not isinstance(contract.get("artifact_type"), str) or not isinstance(contract.get("required_fields"), list) or not isinstance(contract.get("field_types"), Mapping) or not isinstance(assertions, list):
        raise LearningError(f"{phase} evaluation contract is incomplete")
    converted: list[dict[str, Any]] = []
    for assertion in assertions:
        if not isinstance(assertion, Mapping):
            raise LearningError(f"{phase} evaluation contract assertion is invalid")
        source = dict(assertion)
        nested = next((value for key, value in source.items() if key in {"type_is", "value_is", "min_items", "array_length", "all_unique"} and isinstance(value, Mapping)), None)
        if nested is not None:
            source = dict(nested) | {"op": next(key for key, value in assertion.items() if key in {"type_is", "value_is", "min_items", "array_length", "all_unique"} and isinstance(value, Mapping))}
        operator = source.get("operator", source.get("op", source.get("assertion_type")))
        if operator is None:
            operator = next((key for key in ("type_is", "value_is", "min_items", "array_length", "all_unique") if key in source), None)
        path = source.get("path", source.get("field"))
        if not isinstance(operator, str) or not isinstance(path, str) or operator not in {"type_is", "value_is", "min_items", "array_length", "all_unique"}:
            raise LearningError(f"{phase} evaluation contract assertion is unsupported")
        if not path.startswith("$."):
            path = "$." + path
        item = {"path": path, "operator": operator}
        if operator == "type_is":
            value = source.get("value", source.get("type", source.get("type_is", source.get("value_type"))))
            if not isinstance(value, str):
                raise LearningError(f"{phase} evaluation contract type assertion lacks a type")
            item["value"] = value
        elif operator != "all_unique":
            value = source.get("value", source.get(operator))
            if value is None:
                raise LearningError(f"{phase} evaluation contract assertion lacks value")
            item["value"] = value
        converted.append(item)
    contract["assertions"] = converted
    return contract


def _validate_executable_contract(raw: Any, phase: str) -> None:
    """Reject contracts the trusted evaluator cannot execute, before exposure."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LearningError(f"{phase} evaluation contract is not valid JSON") from exc
    if not isinstance(raw, Mapping):
        raise LearningError(f"{phase} evaluation contract must be an object")
    allowed_types = {"string", "array", "object", "boolean", "number", "integer", "null"}
    required = raw.get("required_fields")
    field_types = raw.get("field_types")
    assertions = raw.get("assertions")
    if not isinstance(raw.get("artifact_type"), str) or not raw["artifact_type"].strip():
        raise LearningError(f"{phase} evaluation contract lacks artifact_type")
    if not isinstance(required, list) or not required or len(required) != len(set(required)):
        raise LearningError(f"{phase} evaluation contract required_fields are invalid")
    if not isinstance(field_types, Mapping) or any(field_types.get(field) not in allowed_types for field in required):
        raise LearningError(f"{phase} evaluation contract uses unsupported field type")
    if not isinstance(assertions, list) or not assertions:
        raise LearningError(f"{phase} evaluation contract assertions are invalid")
    allowed_operators = {"type_is", "value_is", "min_items", "array_length", "all_unique"}
    for assertion in assertions:
        if not isinstance(assertion, Mapping) or not isinstance(assertion.get("path"), str) or not assertion["path"].startswith("$."):
            raise LearningError(f"{phase} evaluation contract assertion path is invalid")
        operator = assertion.get("operator")
        if operator not in allowed_operators:
            raise LearningError(f"{phase} evaluation contract uses unsupported assertion operator")
        if operator != "all_unique" and "value" not in assertion:
            raise LearningError(f"{phase} evaluation contract assertion lacks value")


def bind_independent_competence_test_design(
    *,
    request: Mapping[str, Any],
    design: Mapping[str, Any],
    designer_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind an independently authored test design to one discovery request.

    This validates lineage and isolation declarations, but does not create a
    trusted evaluator or accept an answer key. The returned record remains an
    unexecuted design until a registered evaluator is bound by the engine.
    """
    if request.get("schema") != "rex-learning-test-design-request-v1":
        raise LearningError("unsupported competence test-design request")
    if request.get("status") != "awaiting_independent_design":
        raise LearningError("competence test-design request is not awaiting design")
    if not isinstance(design, Mapping) or not isinstance(designer_provenance, Mapping):
        raise LearningError("independent test design and provenance are required")
    if "answer_key" in design or "expected_case_results" in design:
        raise LearningError("learner-authored answer scaffolding is not accepted")
    if not all(isinstance(designer_provenance.get(field), str) and designer_provenance[field].strip() for field in ("provider", "session_id", "role")):
        raise LearningError("independent test designer provenance is incomplete")
    if designer_provenance["role"] not in {"test_designer", "independent_evaluator"}:
        raise LearningError("independent test designer role is invalid")
    lineage_fields = ("proposal_id", "curriculum_id", "curriculum_intent_id", "source_id", "source_hash", "target_capability", "skill_kind")
    if any(design.get(field) != request.get(field) for field in lineage_fields):
        raise LearningError("independent test design does not match request lineage")
    if list(design.get("source_refs", [])) != list(request.get("source_refs", [])):
        raise LearningError("independent test design source references do not match request")
    cases = design.get("cases")
    if not isinstance(cases, list) or not cases or any(not isinstance(case, str) or not case.strip() for case in cases) or len(cases) != len(set(cases)):
        raise LearningError("independent test design cases must be unique, non-empty strings")
    required_phases = ("baseline", "practice", "posttest", "retest", "negative", "fresh", "control")
    phase_cases = design.get("phase_cases")
    phase_tasks = design.get("phase_tasks")
    if not isinstance(phase_cases, Mapping) or not isinstance(phase_tasks, Mapping):
        raise LearningError("independent test design requires complete phase contracts")
    if set(phase_cases) != set(required_phases) or set(phase_tasks) != set(required_phases):
        raise LearningError("independent test design requires baseline, practice, posttest, retest, negative, fresh, and control phases")
    if any(not isinstance(phase_cases[phase], str) or not phase_cases[phase].strip() for phase in required_phases):
        raise LearningError("independent test design phase case IDs must be non-empty strings")
    if len(set(phase_cases.values())) != len(required_phases) or any(phase_cases[phase] not in cases for phase in required_phases):
        raise LearningError("independent test design phase case IDs must be unique and listed in cases")
    if any(not isinstance(phase_tasks[phase], str) or not phase_tasks[phase].strip() for phase in required_phases):
        raise LearningError("independent test design phase tasks must be non-empty strings")
    contradictory_wire_phases = [
        phase for phase in required_phases
        if "responses" in phase_tasks[phase].casefold()
        and any(marker in phase_tasks[phase].casefold() for marker in ("do not return a json envelope", "must not return a json envelope", "without a json envelope"))
    ]
    if contradictory_wire_phases:
        raise LearningError(
            "independent test design contradicts the learner responses envelope: "
            + ", ".join(contradictory_wire_phases)
        )
    if design.get("self_contained_tasks") is True and any(
        "learner-visible fixture:" not in phase_tasks[phase].casefold()
        and "learner-visible starter:" not in phase_tasks[phase].casefold()
        for phase in required_phases
    ):
        raise LearningError("self-contained independent test design must provide a learner-visible fixture or starter artifact for every phase")
    if design.get("self_contained_tasks") is True:
        relevance = _coherence_anchor_tokens(request.get("target_capability")) | _coherence_anchor_tokens(request.get("procedure"))
        if not relevance:
            raise LearningError("self-contained independent test design lacks distinctive capability terms for coherence validation")
        declared_capabilities = {
            capability
            for phase in required_phases
            for capability in _declared_fixture_capabilities(phase_tasks[phase])
        }
        if declared_capabilities and not any(
            len(relevance & _coherence_anchor_tokens(capability)) >= 2
            for capability in declared_capabilities
        ):
            raise LearningError("self-contained independent test design is not coherent with the discovered capability: explicit fixture capability drift")
        contracts = design.get("evaluation_contracts")
        if not isinstance(contracts, Mapping) or set(contracts) != set(required_phases) or any(
            not isinstance(contracts[phase], str) or not contracts[phase].strip()
            for phase in required_phases
        ):
            raise LearningError("self-contained independent test design requires complete evaluation contracts")
        contract_field_gaps = {
            phase: _missing_learner_contract_requirements(phase_tasks[phase], contracts[phase])
            for phase in required_phases
        }
        contract_field_gaps = {phase: fields for phase, fields in contract_field_gaps.items() if fields}
        if contract_field_gaps:
            raise LearningError(
                "independent test task omits evaluator-required learner output fields: "
                + "; ".join(f"{phase}={','.join(fields)}" for phase, fields in contract_field_gaps.items())
            )
        json_shape_gaps = [
            phase for phase in required_phases
            if "output json shape" not in phase_tasks[phase].casefold()
            and _contract_artifact_type(contracts[phase]) == "json_response"
        ]
        if json_shape_gaps:
            raise LearningError(
                "self-contained JSON response tasks require an explicit OUTPUT JSON SHAPE: "
                + ", ".join(json_shape_gaps)
            )
        if design.get("evaluator_type") in {"executable_grader", "executable_process_grader", "contract_artifact_evaluator"}:
            for phase in required_phases:
                _validate_executable_contract(contracts[phase], phase)
        incoherent_phases = [
            phase for phase in required_phases
            if not _coherent_independent_phase(
                phase=phase,
                relevance=relevance,
                phase_text=f"{phase_tasks[phase]} {contracts[phase]} {phase_cases[phase]}",
            )
        ]
        if incoherent_phases:
            raise LearningError(
                "self-contained independent test design is not coherent with the discovered capability: "
                + ", ".join(incoherent_phases)
            )
    if not isinstance(design.get("evaluator_id"), str) or not design["evaluator_id"].strip() or not isinstance(design.get("evaluator_type"), str) or not design["evaluator_type"].strip():
        raise LearningError("independent test design evaluator binding is incomplete")
    if design["skill_kind"] in {"procedure", "workflow"} and design["evaluator_type"] == "contract_artifact_evaluator":
        raise LearningError(
            "schema-only contract evaluator cannot qualify procedure or workflow capabilities; "
            "bind an executable behavioral evaluator"
        )
    if design.get("source_free") is not True or design.get("novel_transfer") is not True or design.get("evaluator_independence") != "independent" or design.get("contamination_status") != "clean" or design.get("negative_applicability") is not True:
        raise LearningError("independent test design must declare clean source-free transfer coverage")
    threshold = design.get("success_threshold", 1.0)
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool) or not 0.0 <= float(threshold) <= 1.0:
        raise LearningError("independent test design threshold is invalid")
    return {
        "schema": "rex-learning-independent-test-design-v1",
        "proposal_id": str(request["proposal_id"]),
        "intent_key": str(request.get("intent_key", "")),
        "curriculum_id": str(request["curriculum_id"]),
        "curriculum_intent_id": str(request.get("curriculum_intent_id", "")),
        "source_id": str(request["source_id"]),
        "source_refs": list(request.get("source_refs", [])),
        "source_hash": str(request["source_hash"]),
        "target_capability": str(request["target_capability"]),
        "skill_kind": str(request["skill_kind"]),
        "cases": list(cases),
        "evaluator_id": str(design["evaluator_id"]),
        "evaluator_type": str(design["evaluator_type"]),
        "success_threshold": float(threshold),
        "source_free": True,
        "novel_transfer": True,
        "evaluator_independence": "independent",
        "contamination_status": "clean",
        "negative_applicability": True,
        "self_contained_tasks": design.get("self_contained_tasks") is True,
        "discovery_provenance": dict(request.get("discovery_provenance", {})) if isinstance(request.get("discovery_provenance"), Mapping) else {},
        "designer_provenance": dict(designer_provenance),
        "raw_design": dict(design),
        **({"evaluation_contracts": {phase: dict(contract) if isinstance(contract, Mapping) else contract for phase, contract in design["evaluation_contracts"].items()}} if isinstance(design.get("evaluation_contracts"), Mapping) else {}),
        **({"phase_cases": dict(design["phase_cases"]), "phase_tasks": dict(design["phase_tasks"])} if "phase_cases" in design and "phase_tasks" in design else {}),
        **({"project_contract": dict(design["project_contract"]), "project_contract_sha256": design["project_contract_sha256"]} if isinstance(design.get("project_contract"), Mapping) and isinstance(design.get("project_contract_sha256"), str) else {}),
        "status": "awaiting_trusted_evaluator_binding",
    }


def define_test_from_independent_design(
    *,
    engine: LearningEngine,
    curriculum_id: str,
    skill_id: str,
    design: Mapping[str, Any],
    phase: str,
) -> dict[str, Any]:
    """Materialize a bound design only through an already trusted evaluator.

    The independent designer supplies cases and evaluator declarations, but
    cannot create an evaluator or provide expected outcomes.  The engine's
    registry and ``define_test`` remain the trust boundary.
    """
    if design.get("schema") != "rex-learning-independent-test-design-v1" or design.get("status") != "awaiting_trusted_evaluator_binding":
        raise LearningError("independent test design is not awaiting trusted evaluator binding")
    if "answer_key" in design or "expected_case_results" in design:
        raise LearningError("learner-authored answer scaffolding is not accepted")
    cases = design.get("cases")
    if not isinstance(cases, list) or not cases or any(not isinstance(case, str) or not case.strip() for case in cases) or len(cases) != len(set(cases)):
        raise LearningError("independent test design cases must be unique, non-empty strings")
    cases = design.get("cases")
    phase_cases = design.get("phase_cases")
    if isinstance(phase_cases, Mapping):
        phase_name = {"pretest": "baseline", "practice": "practice", "posttest": "posttest", "retest": "retest", "negative": "negative", "edge": "fresh", "adversarial": "control"}.get(phase)
        if phase_name is None or not isinstance(phase_cases.get(phase_name), str) or not phase_cases[phase_name].strip():
            raise LearningError("independent test phase case is missing")
        cases = [phase_cases[phase_name]]
    if not isinstance(cases, list) or not cases or any(not isinstance(case, str) or not case.strip() for case in cases) or len(cases) != len(set(cases)):
        raise LearningError("independent test design cases must be unique, non-empty strings")
    evaluator_id = design.get("evaluator_id")
    evaluator_type = design.get("evaluator_type")
    lineage_fields = ("proposal_id", "intent_key", "curriculum_id", "curriculum_intent_id", "source_id", "source_hash", "target_capability", "skill_kind")
    if any(not isinstance(design.get(field), str) or not design[field].strip() for field in lineage_fields):
        raise LearningError("independent test design lineage is incomplete")
    source_refs = design.get("source_refs")
    if not isinstance(source_refs, list) or any(not isinstance(ref, str) or not ref.strip() for ref in source_refs):
        raise LearningError("independent test design source lineage is incomplete")
    discovery_provenance = design.get("discovery_provenance")
    designer_provenance = design.get("designer_provenance")
    if not isinstance(discovery_provenance, Mapping) or not isinstance(designer_provenance, Mapping):
        raise LearningError("independent test design provenance is incomplete")
    if not all(isinstance(designer_provenance.get(field), str) and designer_provenance[field].strip() for field in ("provider", "session_id", "role")) or designer_provenance["role"] not in {"test_designer", "independent_evaluator"}:
        raise LearningError("independent test designer provenance is incomplete")
    if design.get("source_free") is not True or design.get("novel_transfer") is not True or design.get("evaluator_independence") != "independent" or design.get("contamination_status") != "clean" or design.get("negative_applicability") is not True:
        raise LearningError("independent test design must declare clean source-free transfer coverage")
    if not isinstance(evaluator_id, str) or not evaluator_id.strip() or not isinstance(evaluator_type, str) or not evaluator_type.strip():
        raise LearningError("independent test design evaluator binding is incomplete")
    evaluator = engine.trusted_evaluators.get(evaluator_id)
    if evaluator is None:
        raise LearningError("independent test design requires a registered trusted evaluator")
    if getattr(evaluator, "evaluator_type", None) != evaluator_type:
        raise LearningError("independent test design evaluator type does not match registered trusted evaluator")
    source_free = phase != "practice"
    novel_transfer = phase in {"posttest", "retest", "edge", "adversarial"}
    phase_name = {"pretest": "baseline", "practice": "practice", "posttest": "posttest", "retest": "retest", "negative": "negative", "edge": "fresh", "adversarial": "control"}[phase]
    phase_tasks = design.get("phase_tasks")
    test_spec = {
        "phase": phase,
        "cases": list(cases),
        "success_threshold": design.get("success_threshold", 1.0),
        "evaluator_type": evaluator_type,
        "evaluator_id": evaluator_id,
        "evaluator_independence": design.get("evaluator_independence"),
        "contamination_status": design.get("contamination_status"),
        "allowed_resources": list(design.get("allowed_resources", [])),
        "prohibited_resources": list(design.get("prohibited_resources", [])),
        "source_free": source_free,
        "novel_transfer": novel_transfer,
        "independent_design_lineage": {field: design[field] for field in lineage_fields} | {"source_refs": list(source_refs)},
        "independent_design_provenance": {
            "discovery": dict(discovery_provenance),
            "designer": dict(designer_provenance),
        },
    }
    evaluation_contracts = design.get("evaluation_contracts")
    if isinstance(evaluation_contracts, Mapping):
        contract = evaluation_contracts.get(phase_name)
        if isinstance(contract, Mapping) and contract:
            test_spec["evaluation_contract"] = dict(contract)
        elif isinstance(contract, str) and contract.strip():
            test_spec["evaluation_contract"] = contract
    project_contract = design.get("project_contract")
    project_contract_sha256 = design.get("project_contract_sha256")
    if isinstance(project_contract, Mapping) and project_contract and isinstance(project_contract_sha256, str) and project_contract_sha256.strip():
        test_spec["project_contract"] = dict(project_contract)
        test_spec["project_contract_sha256"] = project_contract_sha256
    if isinstance(phase_tasks, Mapping) and isinstance(phase_tasks.get(phase_name), str) and phase_tasks[phase_name].strip():
        test_spec["phase_tasks"] = {phase_name: phase_tasks[phase_name]}
    return engine.define_test(curriculum_id, skill_id, test_spec)


def execute_bound_independent_test(
    *,
    engine: LearningEngine,
    test: Mapping[str, Any],
    learner: Learner,
    task: str,
    material: str = "",
    revision: str = "",
    task_id: str = "",
) -> dict[str, Any]:
    """Execute one independently designed test through trusted evaluation.

    This is deliberately narrower than acquisition: it records an attempt and
    canonical trusted evaluation, but cannot qualify or promote a skill.  The
    source-free declaration on the frozen test also prevents callers from
    quietly exposing instructional material during a transfer test.
    """
    if not isinstance(test, Mapping) or not str(test.get("id", "")).strip():
        raise LearningError("bound independent test is required")
    test_id = str(test["id"])
    persisted = engine.store.read("tests", test_id)
    if persisted.get("schema") != "rex-learning-test-v1":
        raise LearningError("unsupported bound independent test")
    if not persisted.get("independent_design_lineage") or not persisted.get("independent_design_provenance"):
        raise LearningError("bound test lacks independent design provenance")
    if persisted.get("source_free") is True and material.strip():
        raise LearningError("source-free test cannot receive instructional material")
    if not isinstance(task, str) or not task.strip():
        raise LearningError("independent test task is required")
    if not isinstance(getattr(learner, "provider", None), str) or not learner.provider.strip() or not isinstance(getattr(learner, "session_id", None), str) or not learner.session_id.strip():
        raise LearningError("independent test learner provenance is incomplete")
    try:
        response = learner.answer(task=task, material=material, revision=revision)
    except Exception as exc:
        details = {"cause_type": type(exc).__name__}
        code = getattr(exc, "code", None)
        diagnostic = getattr(exc, "diagnostic", None)
        if code is not None:
            details["cause_code"] = code
        if callable(diagnostic):
            details["cause_diagnostic"] = diagnostic()
        raise LearningError(f"independent test learner failed: {exc}", details=details) from exc
    if not isinstance(response, Mapping):
        raise LearningError("independent test learner response must be an object")
    try:
        attempt = engine.record_attempt(test_id, dict(response), task_id=task_id)
    except LearningError as exc:
        if "exactly one response for every frozen case" not in str(exc):
            raise
        rows = response.get("responses") if isinstance(response, Mapping) else None
        actual_cases = [row.get("case") for row in rows if isinstance(row, Mapping)] if isinstance(rows, list) else []
        raise LearningError(
            "independent test learner response does not match frozen case cardinality",
            details={"expected_cases": list(persisted.get("cases", [])), "actual_cases": actual_cases},
        ) from exc
    evaluation = engine.evaluate_trusted_attempt(attempt["id"])
    return {
        "schema": "rex-learning-independent-test-execution-v1",
        "test": persisted,
        "attempt": attempt,
        "evaluation": evaluation,
        "learner_provenance": {
            "provider": learner.provider,
            "session_id": learner.session_id,
            "role": "test_taker",
        },
    }


def derive_independent_behavioral_evidence(
   *,
   engine: LearningEngine,
   executions: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
   """Derive qualification evidence from bound independent-test executions.

   The executions must already have been produced by
   ``execute_bound_independent_test``. This adapter validates their shared
   lineage and phase identity before delegating behavioral gates to the
   engine; it never qualifies or promotes a skill.
   """
   required = ("pretest", "posttest", "retest", "negative")
   if not isinstance(executions, Mapping) or any(key not in executions for key in required):
       raise LearningError("independent test execution records are incomplete")
   records: dict[str, Mapping[str, Any]] = {}
   for name in required:
       execution = executions[name]
       if not isinstance(execution, Mapping) or execution.get("schema") != "rex-learning-independent-test-execution-v1":
           raise LearningError("independent test execution records are invalid")
       test = execution.get("test")
       attempt = execution.get("attempt")
       evaluation = execution.get("evaluation")
       if not all(isinstance(item, Mapping) for item in (test, attempt, evaluation)):
           raise LearningError("independent test execution records are invalid")
       test = cast(Mapping[str, Any], test)
       attempt = cast(Mapping[str, Any], attempt)
       evaluation = cast(Mapping[str, Any], evaluation)
       if not test.get("independent_design_lineage") or not test.get("independent_design_provenance"):
           raise LearningError("independent test execution lacks design provenance")
       if attempt.get("test_id") != test.get("id") or evaluation.get("attempt_id") != attempt.get("id"):
           raise LearningError("independent test execution identity is inconsistent")
       learner_provenance = execution.get("learner_provenance")
       if not isinstance(learner_provenance, Mapping) or learner_provenance.get("role") != "test_taker":
           raise LearningError("independent test learner provenance is invalid")
       records[name] = test
   skill_ids = {str(test.get("skill_id", "")) for test in records.values()}
   if len(skill_ids) != 1 or "" in skill_ids:
       raise LearningError("independent test executions must target one skill")
   baseline_lineage = records["pretest"].get("independent_design_lineage")
   baseline_provenance = records["pretest"].get("independent_design_provenance")
   if any(
       test.get("independent_design_lineage") != baseline_lineage
       or test.get("independent_design_provenance") != baseline_provenance
       for test in records.values()
   ):
       raise LearningError("independent test execution design lineage is inconsistent")
   for name in required:
       execution = executions[name]
       test = cast(Mapping[str, Any], execution["test"])
       attempt = cast(Mapping[str, Any], execution["attempt"])
       evaluation = cast(Mapping[str, Any], execution["evaluation"])
       try:
           persisted_test = engine.store.read("tests", str(test.get("id", "")))
           persisted_attempt = engine.store.read("attempts", str(attempt.get("id", "")))
           persisted_evaluation = engine.store.read("evaluations", str(evaluation.get("id", "")))
       except (AttributeError, KeyError, TypeError) as exc:
           raise LearningError("independent test execution records are not persisted") from exc
       if (
           dict(test) != persisted_test
           or dict(attempt) != persisted_attempt
           or dict(evaluation) != persisted_evaluation
       ):
           raise LearningError("independent test execution records do not match persisted records")
   expected_phases = {"pretest": "pretest", "posttest": "posttest", "retest": "retest", "negative": "negative"}
   if any(records[name].get("phase") != phase for name, phase in expected_phases.items()):
       raise LearningError("independent test execution phases are inconsistent")
   if any(test.get("source_free") is not True for test in records.values()):
       raise LearningError("independent qualification executions must be source-free")
   if any(records[name].get("novel_transfer") is not True for name in ("posttest", "retest", "negative")):
       raise LearningError("independent qualification transfer metadata is missing")
   evidence = engine.derive_candidate_behavioral_evidence(
       pretest_attempt_id=str(executions["pretest"]["attempt"]["id"]),
       pretest_evaluation_id=str(executions["pretest"]["evaluation"]["id"]),
       posttest_evaluation_id=str(executions["posttest"]["evaluation"]["id"]),
       retest_evaluation_id=str(executions["retest"]["evaluation"]["id"]),
       negative_evaluation_id=str(executions["negative"]["evaluation"]["id"]),
   )
   return {
       "schema": "rex-learning-independent-behavioral-evidence-v1",
       "skill_id": next(iter(skill_ids)),
       "execution_ids": {name: str(executions[name]["evaluation"]["id"]) for name in required},
       "evidence": evidence,
   }


def select_learning_proposal_keys(
    proposals: Sequence[Mapping[str, Any]], *, limit: int | None = 2,
    baseline_scores: Mapping[str, Any] | None = None,
) -> list[str]:
    """Select behavioral proposal keys, optionally using measured pretest scores.

    ``baseline_scores`` is caller-supplied trusted pre-exposure evidence, not a
    provider estimate. Known ceiling scores are deferred when there are other
    eligible proposals with measurable headroom; unknown scores retain the
    discovery-time ordering policy.
    """
    if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit < 1):
        raise LearningError("treatment selection limit must be a positive integer")
    eligible = [
        proposal
        for proposal in proposals
        if proposal.get("actionable")
        and (_text(proposal.get("procedure")) or proposal.get("kind") == "declarative")
        and proposal.get("behavioral_evidence_candidate") is True
    ]
    eligible.sort(
        key=lambda proposal: (
            (baseline_scores is not None and str(proposal.get("key")) in baseline_scores
             and _measured_baseline_headroom(baseline_scores[str(proposal["key"])]) == 0.0),
            baseline_scores is None or _measured_baseline_headroom(
                baseline_scores.get(str(proposal["key"]))
            ) is None,
            _baseline_headroom(proposal.get("baseline_headroom")) is None,
            -(_baseline_headroom(proposal.get("baseline_headroom")) or 0.0),
            str(proposal.get("key", "")),
        )
    )
    selected = [str(proposal["key"]) for proposal in eligible]
    return selected if limit is None else selected[:limit]


def select_learning_treatments(
    discovery: Mapping[str, Any], *, limit: int | None = 2,
    baseline_scores: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Select a bounded behavioral subset without caller-enumerated skill keys.

    This policy prefers actionable proposals with concrete procedures and
    explicit behavioral evidence candidates. When trusted source-free baseline
    scores are available, measured ceiling candidates are deferred in favor of
    eligible candidates with headroom; unknown scores remain eligible. It only
    selects treatments; the existing treatment builder still enforces proposal
    eligibility and lineage.
    """
    proposals = discovery.get("proposals")
    if not isinstance(proposals, list):
        raise LearningError("capability discovery result must contain proposals")
    if not all(isinstance(proposal, Mapping) for proposal in proposals):
        raise LearningError("capability proposals must be objects")
    selected_keys = select_learning_proposal_keys(
        cast(Sequence[Mapping[str, Any]], proposals),
        limit=limit,
        baseline_scores=baseline_scores,
    )
    return build_learning_treatments(discovery, selected_keys=selected_keys)


def plan_existing_skill_integrations(
    *,
    engine: LearningEngine,
    discovery: Mapping[str, Any],
    treatments: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Persist non-acquisition dispositions awaiting behavioral requalification.

    This records an integration/conflict-review work item only. It never
    mutates an existing skill or treats a provider proposal as competence.
    """
    if discovery.get("schema") != "rex-learning-capability-discovery-v1":
        raise LearningError("unsupported capability discovery result")
    curriculum = discovery.get("curriculum")
    source = discovery.get("source")
    if not isinstance(curriculum, Mapping) or not isinstance(source, Mapping):
        raise LearningError("discovery curriculum and source are required")
    curriculum_id = str(curriculum.get("id", ""))
    source_id = str(source.get("id", ""))
    if not curriculum_id or not source_id:
        raise LearningError("discovery curriculum and source are required")
    proposals_by_id = {
        str(proposal["id"]): proposal
        for proposal in discovery.get("proposals", [])
        if isinstance(proposal, Mapping) and proposal.get("id")
    }
    planned = {str(item["proposal_id"]): item for item in build_learning_treatments(discovery)}
    records: list[dict[str, Any]] = []
    for treatment in treatments:
        if treatment.get("schema") != "rex-learning-treatment-v1":
            raise LearningError("unsupported learning treatment")
        proposal_id = str(treatment.get("proposal_id", ""))
        proposal = proposals_by_id.get(proposal_id)
        expected = planned.get(proposal_id)
        if proposal is None or expected is None or any(treatment.get(field) != expected.get(field) for field in (
            "curriculum_id", "source_id", "source_hash", "intent_key", "target_capability", "skill_kind",
            "relationship", "treatment_mode", "procedure", "practice_task", "evidence_plan",
        )) or list(treatment.get("source_refs", [])) != list(expected.get("source_refs", [])) or list(treatment.get("existing_skill_ids", [])) != list(expected.get("existing_skill_ids", [])):
            raise LearningError("integration treatment does not match discovery proposal")
        if treatment["treatment_mode"] == "acquire":
            raise LearningError("integration planning requires an existing-skill disposition")
        target_skill_versions: list[dict[str, Any]] = []
        for skill_id in treatment.get("existing_skill_ids", []):
            skill = engine.skill(str(skill_id))
            if skill.get("curriculum_id") != curriculum_id:
                raise LearningError("integration target skill/curriculum mismatch")
            version = skill.get("version")
            if not isinstance(version, int) or version < 0:
                raise LearningError("integration target skill version is invalid")
            target_skill_versions.append({"skill_id": str(skill_id), "version": version})
        status = "awaiting_conflict_review" if treatment["treatment_mode"] == "investigate_conflict" else "awaiting_behavioral_requalification"
        plan_id = "skill-integration-" + hashlib.sha256(f"{source_id}|{proposal_id}|{treatment['treatment_mode']}".encode()).hexdigest()[:16]
        record = {
            "schema": "rex-learning-skill-integration-plan-v1", "id": plan_id,
            "curriculum_id": curriculum_id, "source_id": source_id,
            "source_hash": str(treatment["source_hash"]), "source_refs": list(treatment["source_refs"]),
            "proposal_id": proposal_id, "intent_key": str(treatment["intent_key"]),
            "target_capability": str(treatment["target_capability"]), "skill_kind": str(treatment["skill_kind"]),
            "relationship": str(treatment["relationship"]), "treatment_mode": str(treatment["treatment_mode"]),
            "existing_skill_ids": list(treatment["existing_skill_ids"]),
            "target_skill_versions": target_skill_versions,
            "procedure": str(treatment["procedure"]), "practice_task": str(treatment["practice_task"]),
            "evidence_plan": str(treatment["evidence_plan"]), "status": status,
            "qualification_status": "not_started",
        }
        engine.store.write("skill_integration_plans", plan_id, record)
        records.append(record)
    return records


def preserve_conflict_plan(*, engine: LearningEngine, plan_id: str) -> dict[str, Any]:
    """Persist competing claims without resolving or mutating either claim."""
    plan = engine.store.read("skill_integration_plans", str(plan_id))
    if plan.get("schema") != "rex-learning-skill-integration-plan-v1":
        raise LearningError("unsupported skill integration plan")
    if plan.get("treatment_mode") != "investigate_conflict":
        raise LearningError("skill integration plan is not a conflict review")
    review_id = "skill-conflict-review-" + hashlib.sha256(str(plan_id).encode()).hexdigest()[:16]
    if plan.get("status") == "conflict_preserved":
        return engine.store.read("skill_conflict_reviews", review_id)
    if plan.get("status") != "awaiting_conflict_review":
        raise LearningError("conflict plan is not awaiting review")
    target_ids = plan.get("existing_skill_ids")
    target_versions = plan.get("target_skill_versions")
    if not isinstance(target_ids, list) or not target_ids or not all(isinstance(item, str) and item.strip() for item in target_ids):
        raise LearningError("conflict review requires existing skill targets")
    if not isinstance(target_versions, list) or len(target_versions) != len(target_ids):
        raise LearningError("conflict review target version binding is invalid")
    snapshots: list[dict[str, Any]] = []
    for target_id, binding in zip(target_ids, target_versions, strict=True):
        if not isinstance(binding, Mapping) or binding.get("skill_id") != target_id or not isinstance(binding.get("version"), int) or binding["version"] < 0:
            raise LearningError("conflict review target version binding is invalid")
        skill = engine.skill(target_id)
        if skill.get("curriculum_id") != plan.get("curriculum_id") or skill.get("version") != binding["version"]:
            raise LearningError("conflict review target skill changed")
        snapshots.append({"skill_id": target_id, "version": binding["version"], "claim": str(skill.get("claim", ""))})
    source = engine.store.read("sources", str(plan.get("source_id", "")))
    if source.get("curriculum_id") != plan.get("curriculum_id") or source.get("content_hash") != plan.get("source_hash"):
        raise LearningError("conflict review source lineage is invalid")
    review = {
        "schema": "rex-learning-skill-conflict-review-v1", "id": review_id,
        "plan_id": str(plan_id), "curriculum_id": str(plan["curriculum_id"]),
        "source_id": str(plan["source_id"]), "source_hash": str(plan["source_hash"]),
        "source_refs": list(plan.get("source_refs", [])), "proposal_id": str(plan["proposal_id"]),
        "source_claim": str(plan["target_capability"]), "existing_skill_snapshots": snapshots,
        "status": "unresolved", "resolution": None,
    }
    engine.store.write("skill_conflict_reviews", review_id, review)
    plan["status"] = "conflict_preserved"
    plan["qualification_status"] = "blocked_pending_conflict_resolution"
    plan["conflict_review_id"] = review_id
    engine.store.write("skill_integration_plans", str(plan_id), plan)
    return review


def integrate_existing_skill_plan(
    *,
    engine: LearningEngine,
    plan_id: str,
    revision: Mapping[str, Any],
    evidence_ids: Sequence[str],
) -> dict[str, Any]:
    """Create an unqualified version from one pinned integration plan.

    This consumes only an ``integrate_existing`` plan, verifies that its
    target skill is still at the planned version, and requires clean,
    independent trusted evaluations for that target.  The existing engine
    revision path creates the immutable prior-version snapshot.  The result
    remains a practicing candidate; behavioral requalification and promotion
    are separate operations.
    """
    plan = engine.store.read("skill_integration_plans", str(plan_id))
    if plan.get("schema") != "rex-learning-skill-integration-plan-v1":
        raise LearningError("unsupported skill integration plan")
    if plan.get("integrated_skill_version") is not None:
        if not isinstance(plan["integrated_skill_version"], int):
            raise LearningError("skill integration result version is invalid")
        integrated_ids = plan.get("existing_skill_ids")
        if not isinstance(integrated_ids, list) or len(integrated_ids) != 1 or not isinstance(integrated_ids[0], str):
            raise LearningError("skill integration result target is invalid")
        integrated = engine.skill(integrated_ids[0])
        if integrated.get("version") != plan["integrated_skill_version"] or integrated.get("integration", {}).get("plan_id") != str(plan_id):
            raise LearningError("skill integration result no longer matches the plan")
        return integrated
    if plan.get("treatment_mode") != "integrate_existing" or plan.get("status") != "awaiting_behavioral_requalification":
        raise LearningError("skill integration plan is not awaiting behavioral requalification")
    target_ids = plan.get("existing_skill_ids")
    target_versions = plan.get("target_skill_versions")
    if not isinstance(target_ids, list) or len(target_ids) != 1 or not isinstance(target_ids[0], str) or not target_ids[0].strip():
        raise LearningError("skill integration currently requires exactly one target skill")
    if not isinstance(target_versions, list) or len(target_versions) != 1 or not isinstance(target_versions[0], Mapping):
        raise LearningError("skill integration target version binding is invalid")
    target_id = target_ids[0]
    target_binding = target_versions[0]
    if target_binding.get("skill_id") != target_id or not isinstance(target_binding.get("version"), int) or target_binding["version"] < 0:
        raise LearningError("skill integration target version binding is invalid")
    skill = engine.skill(target_id)
    if skill.get("curriculum_id") != plan.get("curriculum_id"):
        raise LearningError("skill integration target skill/curriculum mismatch")
    if skill.get("version") != target_binding["version"]:
        raise LearningError("skill integration target skill version changed")
    source = engine.store.read("sources", str(plan.get("source_id", "")))
    if source.get("curriculum_id") != plan.get("curriculum_id") or source.get("content_hash") != plan.get("source_hash"):
        raise LearningError("skill integration source lineage is invalid")
    if not isinstance(revision, Mapping) or not revision:
        raise LearningError("skill integration revision is required")
    revision_fields = {"claim", "applicability", "operational_procedure", "preconditions", "success_criteria", "failure_criteria", "limitations"}
    if not any(field in revision for field in revision_fields):
        raise LearningError("skill integration revision has no supported skill fields")
    normalized_evidence = [str(item) for item in evidence_ids if isinstance(item, str) and item.strip()]
    if not normalized_evidence or len(normalized_evidence) != len(evidence_ids) or len(normalized_evidence) > 20:
        raise LearningError("skill integration requires bounded evidence IDs")
    trusted_fields = ("attempt_id", "test_spec_id", "skill_id", "score", "passed", "status", "evaluator_id", "evaluator_type", "independence", "contamination", "validation_status", "evidence", "case_results", "provider_provenance")
    for evidence_id in normalized_evidence:
        evaluation = engine.store.read("evaluations", evidence_id)
        if evaluation.get("skill_id") != target_id or evaluation.get("validation_status") != "verified":
            raise LearningError("skill integration evidence is not trusted for the target skill")
        if evaluation.get("independence") != "independent" or evaluation.get("contamination", {}).get("status") != "clean":
            raise LearningError("skill integration evidence is not independent and clean")
        canonical = engine.evaluate_trusted_attempt(str(evaluation.get("attempt_id", "")))
        if any(evaluation.get(field) != canonical.get(field) for field in trusted_fields):
            raise LearningError("skill integration evidence does not match canonical evaluator output")
        test = engine.store.read("tests", str(evaluation.get("test_spec_id", "")))
        if not test.get("source_free"):
            raise LearningError("skill integration evidence must be source-free")
    updated = engine.revise_skill(
        target_id,
        dict(revision),
        reason=f"discovery integration plan {plan_id}",
        evidence_ids=normalized_evidence,
    )
    updated["qualification_status"] = "candidate"
    updated["integration"] = {
        "plan_id": str(plan_id), "proposal_id": str(plan["proposal_id"]),
        "source_id": str(plan["source_id"]), "source_hash": str(plan["source_hash"]),
        "source_refs": list(plan.get("source_refs", [])), "prior_version": target_binding["version"],
        "evidence_ids": normalized_evidence, "status": "awaiting_behavioral_requalification",
    }
    updated["provenance"].setdefault("integration_plan_ids", []).append(str(plan_id))
    engine.store.write("skills", target_id, updated)
    plan["integrated_skill_version"] = updated["version"]
    plan["integration_evidence_ids"] = normalized_evidence
    plan["qualification_status"] = "candidate_revision_created"
    engine.store.write("skill_integration_plans", str(plan_id), plan)
    return updated


def requalify_existing_skill_plan(
    *,
    engine: LearningEngine,
    plan_id: str,
    pretest_attempt_id: str,
    pretest_evaluation_id: str,
    posttest_evaluation_id: str,
    retest_evaluation_id: str,
    negative_evaluation_id: str,
) -> dict[str, Any]:
    """Promote an integrated revision only after the complete behavioral gate."""
    plan = engine.store.read("skill_integration_plans", str(plan_id))
    integrated_version = plan.get("integrated_skill_version")
    target_ids = plan.get("existing_skill_ids")
    if plan.get("schema") != "rex-learning-skill-integration-plan-v1":
        raise LearningError("unsupported skill integration plan")
    if plan.get("treatment_mode") != "integrate_existing" or plan.get("status") not in {
        "awaiting_behavioral_requalification", "behaviorally_requalified"
    }:
        raise LearningError("skill integration plan is not eligible for behavioral requalification")
    if not isinstance(integrated_version, int) or not isinstance(target_ids, list) or len(target_ids) != 1 or not isinstance(target_ids[0], str):
        raise LearningError("skill integration result is required before requalification")
    skill = engine.skill(target_ids[0])
    if plan.get("status") == "behaviorally_requalified":
        requalified_version = plan.get("requalified_skill_version")
        if isinstance(requalified_version, int) and skill.get("version") == requalified_version and skill.get("integration", {}).get("plan_id") == str(plan_id):
            return skill
        raise LearningError("requalified skill no longer matches the plan")
    if skill.get("version") != integrated_version or skill.get("integration", {}).get("plan_id") != str(plan_id):
        raise LearningError("integrated skill no longer matches the plan")
    evidence_ids = [pretest_evaluation_id, posttest_evaluation_id, retest_evaluation_id, negative_evaluation_id]
    if any(not isinstance(value, str) or not value.strip() for value in evidence_ids) or len(set(evidence_ids)) != len(evidence_ids):
        raise LearningError("behavioral requalification requires distinct evidence IDs")
    trusted_fields = ("attempt_id", "test_spec_id", "skill_id", "score", "passed", "status", "evaluator_id", "evaluator_type", "independence", "contamination", "validation_status", "evidence", "case_results", "provider_provenance")
    for evidence_id in evidence_ids:
        evaluation = engine.store.read("evaluations", evidence_id)
        canonical = engine.evaluate_trusted_attempt(str(evaluation.get("attempt_id", "")))
        if any(evaluation.get(field) != canonical.get(field) for field in trusted_fields):
            raise LearningError("behavioral requalification evidence does not match canonical evaluator output")
    evidence = engine.derive_candidate_behavioral_evidence(
        pretest_attempt_id=pretest_attempt_id,
        pretest_evaluation_id=pretest_evaluation_id,
        posttest_evaluation_id=posttest_evaluation_id,
        retest_evaluation_id=retest_evaluation_id,
        negative_evaluation_id=negative_evaluation_id,
    )
    qualified = engine.promote_demonstrated(
        target_ids[0],
        pretest_attempt_id=pretest_attempt_id,
        pretest_evaluation_id=pretest_evaluation_id,
        posttest_evaluation_id=posttest_evaluation_id,
    )
    qualified["qualification_status"] = "behaviorally_requalified"
    qualified["integration"] = {
        **dict(skill.get("integration", {})),
        "plan_id": str(plan_id),
        "evidence_ids": evidence_ids,
        "behavioral_evidence": evidence,
        "status": "behaviorally_requalified",
    }
    qualified = engine._save("skills", qualified, "existing_skill_requalified")
    plan["status"] = "behaviorally_requalified"
    plan["qualification_status"] = "promoted"
    plan["requalification_evidence_ids"] = evidence_ids
    plan["requalified_skill_version"] = qualified["version"]
    engine.store.write("skill_integration_plans", str(plan_id), plan)
    return qualified


def execute_learning_treatments(
    *,
    engine: LearningEngine,
    discovery: Mapping[str, Any],
    treatments: Sequence[Mapping[str, Any]],
    dependencies: Mapping[str, Mapping[str, Any]],
    learner: Learner | None = None,
    independent_designs: Mapping[str, Mapping[str, Any]] | None = None,
    source_text: str | None = None,
    source_text_by_intent: Mapping[str, str] | None = None,
    successful_acquisition_limit: int | None = None,
    continue_on_ceiling: bool = False,
) -> dict[str, Any]:
    """Run selected discovery treatments through the existing acquisition lifecycle.

    Qualification dependencies are deliberately injected per treatment. This
    adapter does not create graders, reviewers, fresh learners, or controls,
    and it reuses the source already persisted by capability discovery.
    """
    source = discovery.get("source")
    if not isinstance(source, Mapping):
        raise LearningError("discovery result must contain a source")
    source_text = source_text or source.get("text")
    source_title = source.get("title")
    curriculum = discovery.get("curriculum")
    curriculum_id = curriculum.get("id") if isinstance(curriculum, Mapping) else None
    source_id = source.get("id")
    if not all(isinstance(value, str) and value.strip() for value in (source_text, source_title, curriculum_id, source_id)):
        raise LearningError("discovery source and curriculum metadata are required")
    if discovery.get("schema") != "rex-learning-capability-discovery-v1":
        raise LearningError("unsupported capability discovery result")
    if not treatments:
        raise LearningError("at least one learning treatment is required")
    if successful_acquisition_limit is not None and (
        not isinstance(successful_acquisition_limit, int)
        or isinstance(successful_acquisition_limit, bool)
        or successful_acquisition_limit < 1
    ):
        raise LearningError("successful_acquisition_limit must be a positive integer")

    from .acquisition import NonDiscriminatingControlError, learn_from_source

    proposals_by_id = {
        str(proposal["id"]): proposal
        for proposal in discovery.get("proposals", [])
        if isinstance(proposal, Mapping) and proposal.get("id")
    }

    def validate_treatment(treatment: Mapping[str, Any]) -> None:
        proposal_id = str(treatment.get("proposal_id", ""))
        proposal = proposals_by_id.get(proposal_id)
        if proposal is None or not proposal.get("actionable"):
            raise LearningError("treatment does not match discovery proposal")
        relationship = str(proposal["relationship"])
        expected_mode, _ = _treatment_disposition(relationship)
        expected = {
            "curriculum_id": str(proposal["curriculum_id"]),
            "source_id": str(proposal["source_id"]),
            "source_hash": str(proposal["source_hash"]),
            "intent_key": str(proposal["key"]),
            "target_capability": str(proposal["capability"]),
            "skill_kind": str(proposal["kind"]),
            "relationship": relationship,
            "treatment_mode": expected_mode,
            "procedure": str(proposal.get("procedure", "")),
            "practice_task": str(proposal["proposed_practice"]),
            "evidence_plan": str(proposal["proposed_evidence"]),
        }
        if any(treatment.get(field) != value for field, value in expected.items()):
            raise LearningError("treatment does not match discovery proposal")
        if list(treatment.get("source_refs", [])) != list(proposal.get("source_refs", [])):
            raise LearningError("treatment does not match discovery proposal")
        if list(treatment.get("existing_skill_ids", [])) != list(proposal.get("existing_skill_ids", [])):
            raise LearningError("treatment does not match discovery proposal")

    runs: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for treatment in treatments:
        if treatment.get("schema") != "rex-learning-treatment-v1":
            raise LearningError("unsupported learning treatment")
        validate_treatment(treatment)
        if treatment.get("treatment_mode") != "acquire":
            raise LearningError("treatment requires existing-skill integration or conflict review")
        key = _text(treatment.get("intent_key"))
        supplied = dependencies.get(key)
        if not isinstance(supplied, Mapping):
            raise LearningError(f"treatment {key} is missing qualification dependencies")
        required = ("grader", "reviewer", "fresh_learner", "control_learner")
        missing = next((field for field in required if field not in supplied or supplied[field] is None), None)
        if missing:
            raise LearningError(f"treatment {key} is missing {missing}")
        treatment_learner = supplied.get("learner", learner)
        if treatment_learner is None:
            raise LearningError(f"treatment {key} is missing learner")
        try:
            treatment_source_text = str((source_text_by_intent or {}).get(key) or source_text)
            if not treatment_source_text.strip():
                raise LearningError(f"treatment {key} has no supporting instructional material")
            runs.append(learn_from_source(
                engine=engine,
                learner=cast(Learner, treatment_learner),
                source_title=str(source_title),
                source_text=treatment_source_text,
                intent_key=key,
                target_capability=str(treatment["target_capability"]),
                grader=cast(Any, supplied["grader"]),
                reviewer=cast(Any, supplied["reviewer"]),
                fresh_learner=cast(Learner, supplied["fresh_learner"]),
                control_learner=cast(Learner, supplied["control_learner"]),
                independent_design=(cast(Mapping[str, Any], supplied["independent_design"])
                                    if isinstance(supplied.get("independent_design"), Mapping)
                                    else (independent_designs or {}).get(key)),
                baseline_task=str(supplied.get("baseline_task", "Perform the target capability on a concrete held-out task and return the result.")),
                practice_task=str(treatment["practice_task"]),
                runtime_task=str(supplied.get("runtime_task", "Apply the learned capability to the held-out runtime task and return the result.")),
                negative_task=str(supplied.get("negative_task", "Decide whether the learned capability applies to this unrelated task. Return exactly one label: applicable or not-applicable.")),
                skill_kind=str(treatment["skill_kind"]),
                curriculum_id=str(curriculum_id),
                source_locator=str(source.get("locator", "")) or None,
                source_id=str(source_id),
                source_content_hash=str(source.get("content_hash", "")),
                material_content_hash=hashlib.sha256(treatment_source_text.encode()).hexdigest(),
                qualification_tasks=cast(Mapping[str, str], supplied["qualification_tasks"]) if isinstance(supplied.get("qualification_tasks"), Mapping) else None,
                baseline_preflight_id=(str(supplied["baseline_preflight_id"])
                                      if supplied.get("baseline_preflight_id") is not None else None),
            ))
        except (CeilingBaselineError, NonDiscriminatingControlError) as error:
            if not continue_on_ceiling:
                raise
            skipped.append({"proposal_id": str(treatment["proposal_id"]), "intent_key": key, "disposition": error.code, "diagnostic": error.diagnostic()})
        if successful_acquisition_limit is not None and len(runs) >= successful_acquisition_limit:
            break
    return {
        "schema": "rex-learning-treatment-execution-v1",
        "discovery_schema": discovery["schema"],
        "curriculum_id": curriculum_id,
        "source_id": source_id,
        "treatment_keys": [str(treatment["intent_key"]) for treatment in treatments],
        "runs": runs,
        "skipped": skipped,
        "successful_acquisitions": len(runs),
    }