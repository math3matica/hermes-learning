"""Provider-neutral instructional-abstraction controller.

Providers make semantic judgments. Hermes owns boundaries, provenance, claim
identity, epistemic status, and structural admissibility. In particular, the
controller must constrain unsupported cognition without replacing it with a
blanket currentness warning or a fabricated source invariant.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping

from .engine import LearningError
from .instructional_abstraction import CurriculumIntent, SourceContext
from .store import LearningStore

SOURCE_ESTABLISHED = "SOURCE_ESTABLISHED"
INTENT_ESTABLISHED = "INTENT_ESTABLISHED"
INFERRED = "INFERRED"
UNKNOWN = "UNKNOWN"
EXTERNAL_VERIFICATION_REQUIRED = "EXTERNAL_VERIFICATION_REQUIRED"

CURRENTNESS_STATUSES = {
    "NOT_APPLICABLE",
    "SOURCE_ESTABLISHED",
    "INTENT_ESTABLISHED",
    "CURRENTLY_VERIFIED",
    "VERSION_BOUND",
    "HISTORICAL",
    "STALE_RISK",
    "UNKNOWN_RELEVANT",
    UNKNOWN,
    EXTERNAL_VERIFICATION_REQUIRED,
}
CURRENTNESS_APPLICABILITY = {"true", "false", "uncertain", "TRUE", "FALSE", "UNCERTAIN", True, False}
CLAIM_STATUSES = {"PROPOSED", "SUPPORTED", "BOUNDED_INFERENCE", "UNSUPPORTED", "UNKNOWN", "REQUIRES_EXTERNAL_VERIFICATION", "REJECTED"}
CRITIQUE_FINDING_TYPES = {
    "UNSUPPORTED", "OVERBROAD", "UNDERGENERALIZED", "LOST_EXACT_DETAIL",
    "CURRENTNESS_UNSUPPORTED", "ACTION_UNSUPPORTED", "EVIDENCE_PATH_UNSUPPORTED",
    "SOURCE_INTENT_CONTAMINATION", "DEPENDENCY_MISREAD", "OPTIONALITY_MISREAD",
}

STAGES = (
    "extract_instructional_evidence",
    "reconstruct_source_purpose",
    "derive_candidate_abstraction",
    "ground_abstraction_claims",
    "apply_curriculum_intent",
    "select_learning_evidence",
    "critique_instructional_abstraction",
    "finalize_instructional_abstraction",
)

EPISTEMIC_POLICY = [
    "do not assert unsupported source facts",
    "mark inference explicitly with its basis",
    "preserve uncertainty without converting it into an action",
    "keep source meaning separate from goal-conditioned use",
]


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def _text(value: Any, default: str = "") -> str:
    return value.strip() if isinstance(value, str) and value.strip() else default


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _boolish(value: Any) -> str:
    if value is True or (isinstance(value, str) and value.casefold() == "true"):
        return "true"
    if value is False or (isinstance(value, str) and value.casefold() == "false"):
        return "false"
    return "uncertain"


@dataclass(frozen=True)
class LedgerEntry:
    id: str
    status: str
    claim: str
    basis: str
    support_refs: list[str] = field(default_factory=list)
    source_refs: list[str] = field(default_factory=list)
    intent_refs: list[str] = field(default_factory=list)
    confidence: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EpistemicLedger:
    """Append-only evidence ledger for one abstraction run."""

    def __init__(self, entries: list[LedgerEntry] | None = None):
        self._entries = list(entries or [])

    def add(
        self,
        kind: str,
        claim: str,
        *,
        basis: str = "",
        support_refs: list[str] | None = None,
        source_refs: list[str] | None = None,
        intent_refs: list[str] | None = None,
        confidence: str | None = None,
        entry_id: str | None = None,
    ) -> str:
        status = {
            "source_fact": SOURCE_ESTABLISHED,
            "intent_fact": INTENT_ESTABLISHED,
            "inference": INFERRED,
            "unknown": UNKNOWN,
            "external_verification_required": EXTERNAL_VERIFICATION_REQUIRED,
        }.get(kind, kind if kind in {SOURCE_ESTABLISHED, INTENT_ESTABLISHED, INFERRED, UNKNOWN, EXTERNAL_VERIFICATION_REQUIRED} else "")
        if not status or not _text(claim):
            raise LearningError("ledger entry requires a recognized epistemic status and claim")
        identifier = entry_id or f"ledger-{len(self._entries)+1:04d}-{digest({'claim': claim, 'status': status})[:12]}"
        if any(item.id == identifier for item in self._entries):
            raise LearningError(f"duplicate ledger entry: {identifier}")
        self._entries.append(LedgerEntry(identifier, status, claim.strip(), basis.strip(), list(support_refs or []), list(source_refs or []), list(intent_refs or []), confidence))
        return identifier

    def items(self) -> list[LedgerEntry]:
        return list(self._entries)

    def by_id(self) -> dict[str, LedgerEntry]:
        return {item.id: item for item in self._entries}

    def to_dict(self) -> list[dict[str, Any]]:
        return [item.to_dict() for item in self._entries]


def _support_refs(value: Any) -> list[str]:
    return [str(ref) for ref in _list(value)]


def _resolve_ref(ref: str, ledger_ids: set[str]) -> str | None:
    if ref in ledger_ids:
        return ref
    suffix = ref.rsplit("-", 1)[-1]
    return next((item for item in ledger_ids if item.rsplit("-", 1)[-1] == suffix), None)


def validate_staged_abstraction(candidate: Mapping[str, Any], *, ledger_ids: set[str], ledger_kinds: Mapping[str, str] | None = None) -> None:
    """Validate relationships only; semantic truth remains provider/evaluator-owned."""
    ledger_kinds = dict(ledger_kinds or {})
    for claim in _list(candidate.get("claims")):
        if not isinstance(claim, Mapping):
            raise LearningError("claim support record must be an object")
        status = str(claim.get("status", ""))
        refs = _support_refs(claim.get("support_refs"))
        if status not in {"REJECTED", "UNKNOWN", "UNSUPPORTED", "PROPOSED"} and (not refs or any(_resolve_ref(ref, ledger_ids) is None for ref in refs)):
            raise LearningError("claim has missing or invalid support")
        if status not in {"REJECTED", "UNKNOWN", "UNSUPPORTED", "PROPOSED"} and claim.get("support_type") == "source_established" and any(ledger_kinds.get(_resolve_ref(ref, ledger_ids) or "") != SOURCE_ESTABLISHED for ref in refs):
            raise LearningError("source_established claim cites non-source support")
    for field_name in ("relevance", "source_invariant", "goal_conditioned_interpretation", "learning_action", "evidence_path"):
        value = candidate.get(field_name)
        if not isinstance(value, Mapping):
            continue
        refs = _support_refs(value.get("support_refs"))
        if field_name == "goal_conditioned_interpretation":
            continue
        if refs and any(_resolve_ref(ref, ledger_ids) is None for ref in refs):
            raise LearningError(f"{field_name} has missing or invalid support")


def _fallback_evidence(intent: CurriculumIntent, source: SourceContext, ledger: EpistemicLedger) -> dict[str, Any]:
    """Create source and intent anchors, not a universal currentness judgment."""
    source_id = ledger.add("source_fact", f"The visible source is titled {source.title!r} and contains the supplied material.", basis="visible source context", source_refs=source.source_refs)
    text_id = ledger.add("source_fact", source.text or "The visible source text is empty.", basis="visible source text", source_refs=source.source_refs)
    intent_id = ledger.add("intent_fact", f"The stated target capability is {intent.target_capability}", basis="curriculum intent", intent_refs=[intent.key])
    return {
        "explicit_source_claims": [source_id, text_id],
        "explicit_curriculum_facts": [intent_id],
        "extraction_gaps": [],
        "unresolved_questions": [],
        "epistemic_policy": list(EPISTEMIC_POLICY),
    }


def _provider_stage(provider: Callable[[str, Mapping[str, Any]], Mapping[str, Any]] | None, operation: str, inputs: Mapping[str, Any]) -> dict[str, Any]:
    if provider is None:
        return {}
    output = provider(operation, inputs)
    if not isinstance(output, Mapping):
        raise LearningError(f"stage {operation} provider output must be an object")
    return dict(output)


def _merge_ledger_entries(ledger: EpistemicLedger, output: Mapping[str, Any]) -> None:
    for raw in _list(output.get("ledger_entries")):
        if not isinstance(raw, Mapping):
            raise LearningError("provider ledger entry must be an object")
        entry_id = str(raw["id"]) if raw.get("id") else None
        if entry_id and entry_id in ledger.by_id():
            continue
        ledger.add(
            str(raw.get("kind", "")), _text(raw.get("claim")), basis=_text(raw.get("basis")),
            support_refs=_support_refs(raw.get("support_refs")), source_refs=_support_refs(raw.get("source_refs")),
            intent_refs=_support_refs(raw.get("intent_refs")), confidence=raw.get("confidence"), entry_id=entry_id,
        )


def _source_ledger(ledger: EpistemicLedger) -> list[dict[str, Any]]:
    return [item.to_dict() for item in ledger.items() if item.status == SOURCE_ESTABLISHED]


def _intent_ledger(ledger: EpistemicLedger) -> list[dict[str, Any]]:
    return [item.to_dict() for item in ledger.items() if item.status == INTENT_ESTABLISHED]


def _claim_text(raw: Mapping[str, Any], role: str) -> str:
    for key in ("claim", "detail", "mechanism", "invariant", "statement", "interpretation"):
        if _text(raw.get(key)):
            return _text(raw[key])
    return f"{role} proposition is not stated."


def _normalize_claim(raw: Mapping[str, Any], *, role: str, index: int, ledger_ids: set[str]) -> dict[str, Any] | None:
    claim = _claim_text(raw, role)
    if claim.endswith("is not stated."):
        return None
    refs = _support_refs(raw.get("support_refs"))
    stable = _text(raw.get("id")) or f"{role}-{index}-{digest({'claim': claim, 'refs': refs})[:10]}"
    status = str(raw.get("status", "PROPOSED")).upper()
    if status == "SUPPORTED":
        status = "SUPPORTED"
    elif status not in CLAIM_STATUSES:
        status = "PROPOSED"
    if refs and any(ref not in ledger_ids for ref in refs):
        status = "UNKNOWN"
    support_type = _text(raw.get("support_type"), "inference" if status == "BOUNDED_INFERENCE" else "")
    return {
        "id": stable, "claim": claim, "semantic_role": role, "epistemic_status": status,
        "status": status, "support_refs": refs, "support_type": support_type,
        "inference_basis": _text(raw.get("inference_basis")), "applicability": raw.get("applicability", "unknown"),
        "source_or_goal_scope": _text(raw.get("source_or_goal_scope"), "source" if role in {"exact_detail", "source_invariant", "contingent_detail"} else "goal"),
        "confidence": raw.get("confidence"), "derived_from_claim_ids": list(raw.get("derived_from_claim_ids", [])),
    }


def _collect_claims(candidate: Mapping[str, Any], ledger_ids: set[str]) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    seen: set[str] = set()
    raw_claims = _list(candidate.get("claims")) + _list(candidate.get("proposed_claims"))
    for role, field_name in (("exact_detail", "exact_details"), ("source_invariant", "invariants"), ("transferable_mechanism", "transferable_mechanisms"), ("contingent_detail", "contingent_details")):
        raw_claims += [dict(item, semantic_role=role) for item in _list(candidate.get(field_name)) if isinstance(item, Mapping)]
    for index, raw in enumerate(raw_claims):
        if not isinstance(raw, Mapping):
            continue
        role = _text(raw.get("semantic_role"), "proposition")
        item = _normalize_claim(raw, role=role, index=index, ledger_ids=ledger_ids)
        if item and item["id"] not in seen:
            claims.append(item); seen.add(item["id"])
    return claims


def _is_fake_policy_claim(value: Any) -> bool:
    return _text(value).casefold() == "retain only what the visible source supports."


def _source_invariant(candidate: Mapping[str, Any], claims: list[dict[str, Any]], source_ids: set[str]) -> dict[str, Any]:
    raw = candidate.get("source_invariant")
    if not isinstance(raw, Mapping) or _is_fake_policy_claim(raw.get("claim")):
        matching = next((item for item in claims if item["semantic_role"] == "source_invariant" and set(item["support_refs"]) & source_ids), None)
        if matching:
            return dict(matching)
        return {"claim": "Source invariant is unknown.", "status": "unknown", "epistemic_status": "UNKNOWN", "support_refs": [], "support_type": "unknown"}
    item = _normalize_claim(raw, role="source_invariant", index=0, ledger_ids=source_ids)
    if not item or not set(item["support_refs"]).issubset(source_ids):
        return {"claim": "Source invariant is unknown.", "status": "unknown", "epistemic_status": "UNKNOWN", "support_refs": [], "support_type": "unknown"}
    return item


def _currentness_default(intent: CurriculumIntent, source: SourceContext) -> dict[str, Any]:
    text = f"{source.title} {source.text}".casefold()
    temporal = f"{intent.temporal_requirements} {' '.join(intent.intended_use)} {' '.join(intent.intended_transfer)}".casefold()
    operational = any(word in text + " " + temporal for word in ("current", "version", "api", "procedure", "install", "deployment", "tool", "service", "production", "execute", "configuration"))
    if not operational and not source.version_metadata and not source.source_date:
        return {"applicable": "false", "status": "NOT_APPLICABLE", "support_refs": []}
    return {"applicable": "uncertain", "status": "UNKNOWN_RELEVANT", "support_refs": []}


def _normalize_currentness(value: Any, intent: CurriculumIntent, source: SourceContext) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        value = _currentness_default(intent, source)
    applicable = _boolish(value.get("applicable", value.get("currentness_applicable")))
    if applicable == "false":
        status = "NOT_APPLICABLE"
    else:
        status = str(value.get("status", "UNKNOWN_RELEVANT")).upper()
        if status not in CURRENTNESS_STATUSES:
            status = "UNKNOWN_RELEVANT"
    return {"applicable": applicable.upper(), "status": status, "support_refs": _support_refs(value.get("support_refs")), "reason": _text(value.get("reason"))}


def _normalize_critique(raw: Mapping[str, Any]) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    for index, item in enumerate(_list(raw.get("findings"))):
        if not isinstance(item, Mapping):
            continue
        finding_type = str(item.get("finding_type", "UNSUPPORTED")).upper()
        if finding_type not in CRITIQUE_FINDING_TYPES:
            finding_type = "UNSUPPORTED"
        findings.append({
            "finding_id": _text(item.get("finding_id"), f"finding-{index+1:03d}"),
            "target_claim_id": _text(item.get("target_claim_id")), "target_path": _text(item.get("target_path")),
            "finding_type": finding_type, "severity": _text(item.get("severity"), "warning"),
            "rationale": _text(item.get("rationale")), "recommended_disposition": _text(item.get("recommended_disposition"), "REVIEW"),
            "support_refs": _support_refs(item.get("support_refs")),
        })
    for index, claim_id in enumerate(_list(raw.get("unsupported_claim_ids")) + _list(raw.get("unsupported_claims"))):
        if isinstance(claim_id, str):
            findings.append({"finding_id": f"legacy-unsupported-{index+1:03d}", "target_claim_id": claim_id, "target_path": "", "finding_type": "UNSUPPORTED", "severity": "error", "rationale": "legacy unsupported-claim field", "recommended_disposition": "REJECT", "support_refs": []})
    for index, path in enumerate(_list(raw.get("overbroad_applicability"))):
        if isinstance(path, Mapping):
            path = path.get("id", path.get("target_path", ""))
        findings.append({"finding_id": f"legacy-overbroad-{index+1:03d}", "target_claim_id": "", "target_path": str(path), "finding_type": "OVERBROAD", "severity": "error", "rationale": "legacy overbroad applicability finding", "recommended_disposition": "NARROW_OR_REJECT", "support_refs": []})
    for index, path in enumerate(_list(raw.get("actions_not_supported_by_evidence"))):
        target = path.get("id", "learning_action") if isinstance(path, Mapping) else str(path)
        findings.append({"finding_id": f"legacy-action-{index+1:03d}", "target_claim_id": "", "target_path": target, "finding_type": "ACTION_UNSUPPORTED", "severity": "error", "rationale": "legacy unsupported action finding", "recommended_disposition": "UNRESOLVED", "support_refs": []})
    return {"schema": "rex-learning-critique-v2", "findings": findings, "notes": _text(raw.get("notes"))}


@dataclass
class StagedAbstractionResult:
    ledger: EpistemicLedger
    stages: dict[str, dict[str, Any]]
    final: dict[str, Any]
    provenance: dict[str, Any]

    def persist(self, store: LearningStore, *, intent_id: str, source_hash: str) -> dict[str, Any]:
        durable = [claim for claim in self.final.get("claims", []) if claim.get("status") in {"SUPPORTED", "BOUNDED_INFERENCE"} and claim.get("support_refs")]
        record = {
            "schema": "rex-learning-staged-abstraction-run-v2", "id": f"staged-run-{digest({'intent_id': intent_id, 'source_hash': source_hash, 'output_hash': self.provenance['output_hash']})[:24]}",
            "curriculum_intent_id": intent_id, "source_hash": source_hash, "ledger": self.ledger.to_dict(), "stages": self.stages,
            "final": self.final, "durable_claims": durable, "provenance": self.provenance, "created_at": time.time(),
        }
        try:
            existing = store.read("staged_abstraction_runs", record["id"])
        except KeyError:
            return store.write("staged_abstraction_runs", record["id"], record)
        comparable = {k: v for k, v in record.items() if k != "created_at"}
        if {k: v for k, v in existing.items() if k != "created_at"} != comparable:
            raise LearningError("staged abstraction run is immutable")
        return existing


class StagedAbstractionPipeline:
    """Eight bounded stages: seven provider operations plus deterministic finalization."""

    def run(self, intent: CurriculumIntent, source: SourceContext, *, provider: Callable[[str, Mapping[str, Any]], Mapping[str, Any]] | None = None, session_id: str = "") -> StagedAbstractionResult:
        ledger = EpistemicLedger(); stages: dict[str, dict[str, Any]] = {}
        evidence = _fallback_evidence(intent, source, ledger)
        evidence.update(_provider_stage(provider, STAGES[0], {"source": source.to_dict(), "source_ledger": _source_ledger(ledger), "ledger": _source_ledger(ledger), "epistemic_policy": EPISTEMIC_POLICY}))
        _merge_ledger_entries(ledger, evidence); stages[STAGES[0]] = evidence
        purpose = _provider_stage(provider, STAGES[1], {"source": source.to_dict(), "source_evidence": evidence, "source_ledger": _source_ledger(ledger), "ledger": _source_ledger(ledger), "epistemic_policy": EPISTEMIC_POLICY})
        _merge_ledger_entries(ledger, purpose); purpose.setdefault("source_purpose", "Author purpose is unknown from the supplied source."); stages[STAGES[1]] = purpose
        candidate = _provider_stage(provider, STAGES[2], {"source": source.to_dict(), "source_evidence": evidence, "source_purpose": purpose, "source_ledger": _source_ledger(ledger), "ledger": _source_ledger(ledger), "epistemic_policy": EPISTEMIC_POLICY})
        _merge_ledger_entries(ledger, candidate); stages[STAGES[2]] = candidate
        source_ids = {item.id for item in ledger.items() if item.status == SOURCE_ESTABLISHED}; intent_ids = {item.id for item in ledger.items() if item.status == INTENT_ESTABLISHED}; ledger_ids = {item.id for item in ledger.items()}
        claims = _collect_claims(candidate, ledger_ids); invariant = _source_invariant(candidate, claims, source_ids)
        grounding = _provider_stage(provider, STAGES[3], {"candidate": {**candidate, "claims": claims, "source_invariant": invariant}, "source_ledger": _source_ledger(ledger), "ledger": _source_ledger(ledger), "epistemic_policy": EPISTEMIC_POLICY})
        _merge_ledger_entries(ledger, grounding)
        grounding_has_claims = bool(_list(grounding.get("claims")) or _list(grounding.get("proposed_claims")))
        grounded_claims = _collect_claims(grounding, ledger_ids) if grounding_has_claims else claims
        by_id = {item["id"]: item for item in claims}
        for item in grounded_claims:
            if item["id"] in by_id:
                by_id[item["id"]].update(item)
        if not grounding_has_claims:
            for item in claims:
                if item.get("support_refs") and item.get("status") == "PROPOSED":
                    item["status"] = item["epistemic_status"] = "SUPPORTED"
        for raw in _list(grounding.get("claims")):
            if isinstance(raw, Mapping) and _text(raw.get("id")) in by_id and str(raw.get("status", "")).upper() in CLAIM_STATUSES:
                by_id[str(raw["id"])] ["status"] = str(raw["status"]).upper()
                by_id[str(raw["id"])] ["epistemic_status"] = str(raw["status"]).upper()
        for item in claims:
            item.update(by_id[item["id"]])
        ledger_kinds = {item.id: item.status for item in ledger.items()}
        for item in claims:
            if item.get("support_type") == "source_established" and any(ledger_kinds.get(ref) != SOURCE_ESTABLISHED for ref in item.get("support_refs", [])):
                item["status"] = item["epistemic_status"] = "REJECTED"
        grounding["claims"] = claims; grounding["source_invariant"] = invariant; stages[STAGES[3]] = grounding
        relevance = _provider_stage(provider, STAGES[4], {"candidate": {**candidate, "claims": claims, "source_invariant": invariant}, "grounding": grounding, "intent": intent.to_dict(), "intent_ledger": _intent_ledger(ledger), "epistemic_policy": EPISTEMIC_POLICY})
        _merge_ledger_entries(ledger, relevance); relevance.setdefault("relevance", {"label": "unresolved", "support_refs": []}); relevance.setdefault("goal_conditioned_interpretation", {"claim": f"Goal-conditioned use for {intent.target_capability} is unresolved.", "status": "UNKNOWN", "support_refs": []}); stages[STAGES[4]] = relevance
        action = _provider_stage(provider, STAGES[5], {"candidate": {**candidate, "claims": claims, "source_invariant": invariant}, "relevance": relevance, "grounding": grounding, "intent": intent.to_dict(), "intent_ledger": _intent_ledger(ledger), "epistemic_policy": EPISTEMIC_POLICY})
        _merge_ledger_entries(ledger, action); stages[STAGES[5]] = action
        currentness = _normalize_currentness(relevance.get("currentness", action.get("currentness")), intent, source)
        assembled = {**candidate, **relevance, **action, "claims": claims, "source_invariant": invariant, "currentness": currentness, "epistemic_policy": list(EPISTEMIC_POLICY)}
        validate_staged_abstraction(assembled, ledger_ids={item.id for item in ledger.items()}, ledger_kinds={item.id: item.status for item in ledger.items()})
        critique_raw = _provider_stage(provider, STAGES[6], {"source": source.to_dict(), "intent": intent.to_dict(), "candidate": assembled, "ledger": ledger.to_dict(), "epistemic_policy": EPISTEMIC_POLICY})
        _merge_ledger_entries(ledger, critique_raw); critique = _normalize_critique(critique_raw); stages[STAGES[6]] = {**critique_raw, **critique}
        final = self._finalize(assembled, critique, ledger); stages[STAGES[7]] = final
        provenance = {"schema": "rex-learning-staged-abstraction-provenance-v2", "operation": "staged_instructional_abstraction", "session_id": session_id, "stage_count": len(STAGES), "provider_semantic_stage_count": len(STAGES) - 1, "stage_operations": list(STAGES), "input_hash": digest({"intent": intent.to_dict(), "source": source.to_dict()}), "output_hash": digest(final), "epistemic_policy_hash": digest(EPISTEMIC_POLICY), "currentness_policy": "applicability_sensitive", "claim_schema": "rex-learning-claim-v2", "critique_schema": "rex-learning-critique-v2"}
        return StagedAbstractionResult(ledger, stages, final, provenance)

    @staticmethod
    def _finalize(candidate: dict[str, Any], critique: dict[str, Any], ledger: EpistemicLedger) -> dict[str, Any]:
        final = dict(candidate); claims = [dict(item) for item in _list(candidate.get("claims"))]; by_id = {item.get("id"): item for item in claims}
        path_map = {f"{item.get('semantic_role')}[{index}]": item.get("id") for index, item in enumerate(claims)}
        for finding in critique.get("findings", []):
            target = finding.get("target_claim_id") or path_map.get(finding.get("target_path", ""))
            if target in by_id and finding.get("finding_type") in {"UNSUPPORTED", "OVERBROAD", "LOST_EXACT_DETAIL", "CURRENTNESS_UNSUPPORTED", "SOURCE_INTENT_CONTAMINATION", "DEPENDENCY_MISREAD", "OPTIONALITY_MISREAD"}:
                by_id[target]["status"] = "REJECTED" if finding.get("recommended_disposition", "").upper() in {"REJECT", "NARROW_OR_REJECT"} or finding.get("finding_type") == "UNSUPPORTED" else "BOUNDED_INFERENCE"
                by_id[target]["epistemic_status"] = by_id[target]["status"]
                by_id[target]["critique_finding_ids"] = [finding.get("finding_id")]
            if finding.get("finding_type") == "ACTION_UNSUPPORTED" and finding.get("target_path") in {"", "action", "learning_action"}:
                final["learning_action"] = "unresolved"
            if finding.get("finding_type") == "EVIDENCE_PATH_UNSUPPORTED" and finding.get("target_path") in {"", "evidence_path"}:
                final["evidence_path"] = []
        final["claims"] = claims
        if final.get("currentness", {}).get("applicable") == "FALSE":
            final["currentness"]["status"] = "NOT_APPLICABLE"
            if final.get("learning_action") == "verify_current_authoritative_source":
                final["learning_action"] = "unresolved"
        elif final.get("currentness", {}).get("applicable") == "TRUE" and final.get("currentness", {}).get("status") == EXTERNAL_VERIFICATION_REQUIRED:
            if final.get("learning_action") in {None, "", "unresolved"}:
                final["learning_action"] = "verify_current_authoritative_source"
        final["critique"] = critique; final["ledger_entry_count"] = len(ledger.items()); final["final_evaluation_surface"] = {key: final.get(key) for key in ("source_invariant", "exact_details", "transferable_mechanisms", "goal_conditioned_interpretation", "relevance", "dependency", "optionality", "currentness", "learning_action", "evidence_path", "claims")}
        return final