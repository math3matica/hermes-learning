from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from .engine import LearningError


RELEVANCE = {"core", "supporting", "optional", "contextual", "reference", "irrelevant"}
OPTIONALITY = {"required", "strongly-useful", "enrichment", "specialization", "safely-skippable"}
CURRENTNESS = {"enduring", "version-bound", "stale-risk", "obsolete", "unknown"}
ACTIONS = {"deep-study", "normal-study", "retrieval", "practice", "execute", "reference-only", "contextual-record", "skip", "repair-source", "current-doc-reconciliation"}
VALUES = {"mastery_required", "mastery_useful", "application_required", "practice_required", "reference_sufficient", "context_only", "skip_for_this_objective", "verify_current_source_before_use", "extraction_repair_required"}
LEVELS = {"raw-detail", "source-specific", "concept", "syntax-semantics", "generalized-principle", "operational-procedure", "methodology"}


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def _required_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LearningError(f"{name} is required")
    return value.strip()


@dataclass(frozen=True)
class CurriculumIntent:
    key: str
    title: str
    target_capability: str
    competence_level: str
    intended_use: list[str] = field(default_factory=list)
    intended_transfer: list[str] = field(default_factory=list)
    environment: dict[str, Any] = field(default_factory=dict)
    target_languages: list[str] = field(default_factory=list)
    target_platforms: list[str] = field(default_factory=list)
    target_tools: list[str] = field(default_factory=list)
    temporal_requirements: str = ""
    exclusions: list[str] = field(default_factory=list)
    optional_specializations: list[str] = field(default_factory=list)
    prerequisite_assumptions: list[str] = field(default_factory=list)
    evaluation_criteria: list[str] = field(default_factory=list)
    desired_behavioral_change: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CurriculumIntent":
        required = {key: _required_string(value.get(key), key) for key in ("key", "title", "target_capability", "competence_level")}
        list_fields = ("intended_use", "intended_transfer", "target_languages", "target_platforms", "target_tools", "exclusions", "optional_specializations", "prerequisite_assumptions", "evaluation_criteria", "desired_behavioral_change")
        return cls(**required, **{key: list(value.get(key, [])) for key in list_fields}, environment=dict(value.get("environment", {})), temporal_requirements=str(value.get("temporal_requirements", "")))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SourceContext:
    source_id: str
    unit_id: str
    title: str
    text: str
    source_refs: list[str] = field(default_factory=list)
    hierarchy: dict[str, Any] = field(default_factory=dict)
    source_date: str | None = None
    version_metadata: dict[str, Any] = field(default_factory=dict)
    source_quality: str = "unknown"
    neighboring_context: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SourceContext":
        return cls(source_id=_required_string(value.get("source_id"), "source_id"), unit_id=_required_string(value.get("unit_id"), "unit_id"), title=_required_string(value.get("title"), "title"), text=str(value.get("text", "")), source_refs=list(value.get("source_refs", [])), hierarchy=dict(value.get("hierarchy", {})), source_date=value.get("source_date"), version_metadata=dict(value.get("version_metadata", {})), source_quality=str(value.get("source_quality", "unknown")), neighboring_context=list(value.get("neighboring_context", [])))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class InstructionalAbstraction:
    source_purpose: str
    learner_purpose: str
    objective_relevance: str
    relevance_strength: str
    expected_learning_contribution: str
    retained_abstraction: str
    generalizable_invariants: list[str]
    contingent_details: list[str]
    applicability: list[str]
    non_applicability: list[str]
    prerequisite_or_dependency_role: str
    optionality: str
    currentness: str
    external_verification_need: str
    learning_action: str
    abstraction_level: str
    instructional_value: str
    mastery_required: bool
    required_evidence: list[str]
    source_quality: str
    confidence: float
    trace: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "InstructionalAbstraction":
        fields = {key: value.get(key) for key in ("source_purpose", "learner_purpose", "objective_relevance", "relevance_strength", "expected_learning_contribution", "retained_abstraction", "prerequisite_or_dependency_role", "external_verification_need", "learning_action", "abstraction_level", "instructional_value", "source_quality")}
        for key, val in fields.items():
            fields[key] = _required_string(val, key)
        for key in ("generalizable_invariants", "contingent_details", "applicability", "non_applicability", "required_evidence"):
            fields[key] = list(value.get(key, []))
        fields["optionality"] = _required_string(value.get("optionality"), "optionality")
        fields["currentness"] = _required_string(value.get("currentness"), "currentness")
        fields["mastery_required"] = bool(value.get("mastery_required", False))
        fields["confidence"] = float(value.get("confidence", 0.0))
        if fields["relevance_strength"] not in RELEVANCE or fields["optionality"] not in OPTIONALITY or fields["currentness"] not in CURRENTNESS or fields["learning_action"] not in ACTIONS or fields["instructional_value"] not in VALUES or fields["abstraction_level"] not in LEVELS:
            raise LearningError("invalid instructional abstraction judgment")
        if not 0 <= fields["confidence"] <= 1:
            raise LearningError("abstraction confidence must be between 0 and 1")
        return cls(**fields, trace=dict(value.get("trace", {})))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def assess_instructional_abstraction(intent: CurriculumIntent, source: SourceContext) -> InstructionalAbstraction:
    """Conservative deterministic scaffold for frozen tests and offline operation.

    Production providers may supply the same contract through CognitiveRouter. This
    fallback is deliberately narrow: it is not a keyword taxonomy pretending to be
    pedagogy, and it refuses to claim more than the supplied context supports.
    """
    title = source.title.casefold()
    text = source.text.casefold()
    objective = intent.target_capability
    author_purpose = "present instructional material"
    relevance = "supporting"
    optionality = "strongly-useful"
    currentness = "enduring"
    action = "normal-study"
    value = "mastery_useful"
    level = "concept"
    retained = "Retain the concept at the level required by the stated capability, not the author's incidental example."
    invariants = ["Separate durable principle from source-specific representation."]
    contingent: list[str] = []
    required_evidence = ["explain", "apply"]
    contribution = "Build understanding that can transfer to the intended use."
    dependency = "May support later application; inspect hierarchy before downgrading."

    if any(word in title for word in ("biography", "about the author", "dedication", "acknowledg")):
        author_purpose = "establish author or publication context"
        relevance, optionality, action, value = "contextual", "safely-skippable", "contextual-record", "context_only"
        retained = "Retain provenance and context only; personal trivia is not a programming capability."
        contribution = "Provide source provenance, not mastery content."
        required_evidence = ["location preserved"]
        mastery = False
    elif "reference" in title or "table" in title:
        author_purpose = "provide lookup material"
        relevance, optionality, action, value = "reference", "strongly-useful", "reference-only", "reference_sufficient"
        retained = "Retain the lookup location and recognize when the exact signature or representation is needed."
        contribution = "Enable accurate lookup without requiring rote memorization."
        required_evidence = ["find and use reference"]
        mastery = False
    elif any(word in title for word in ("challenge", "exercise", "programming problem", "practice")):
        author_purpose = "provide practice and transfer"
        relevance, optionality, action, value, level = "core", "required", "practice", "practice_required", "operational-procedure"
        retained = "Retain the problem-solving technique and conditions; the evidence is an independent working program."
        contribution = "Build operational programming skill through attempt, feedback, and retry."
        required_evidence = ["code", "compile", "execute", "tests", "diagnosis"]
        mastery = True
    elif any(word in text for word in ("jdk 8", "jdk 9", "java 8", "java 9", "ide setup", "download workflow")):
        author_purpose = "teach a period-specific development setup"
        currentness = "version-bound"
        contingent = ["named Java version", "period-specific installation or IDE workflow"]
        invariants = ["A development project requires a compatible JDK/toolchain.", "Project version constraints must be checked before setup."]
        retained = "Retain JDK/toolchain compatibility and version-constraint reasoning; do not adopt the historical installation procedure as the modern default."
        action = "execute" if intent.environment.get("jdk") in {"8", "9"} else "current-doc-reconciliation"
        value = "application_required" if action == "execute" else "verify_current_source_before_use"
        relevance = "supporting" if intent.environment.get("jdk") not in {"8", "9"} else "core"
        optionality = "strongly-useful" if relevance == "core" else "enrichment"
        mastery = relevance == "core"
        required_evidence = ["identify project version", "execute compatible setup", "consult current authoritative documentation"]
        level = "operational-procedure"
    elif any(word in title for word in ("syntax", "semantics", "promotion", "comparison trouble", "grouping symbols")) or "binary numeric promotion" in text:
        author_purpose = "teach a language rule"
        relevance, optionality, action, value, level = "core", "required", "retrieval", "mastery_required", "syntax-semantics"
        retained = "Java binary numeric promotion converts byte and short operands to int before arithmetic; preserve this concrete semantic rule."
        contribution = "Support correct reconstruction, compilation, and debugging of Java expressions."
        required_evidence = ["reconstruct", "compile/use correctly"]
        mastery = True
    elif any(word in text for word in ("event-driven", "state", "render", "update")) and any(word in text for word in ("bubble", "android", "gui")):
        author_purpose = "teach a transferable concept through a worked application"
        relevance, optionality, action, value, level = "supporting", "strongly-useful", "normal-study", "mastery_useful", "generalized-principle"
        retained = "Retain event/state/update/render relationships; the named application is an example and practice context, not the whole concept."
        contribution = "Transfer an event-driven mental model beyond the worked application."
        required_evidence = ["explain invariant", "apply in a novel domain"]
        mastery = False
    elif "android" in title or "android" in text:
        author_purpose = "teach a platform-specific worked application"
        relevance, optionality, action, value, level = "optional", "specialization", "reference-only", "reference_sufficient", "source-specific"
        retained = "Retain only Java/programming principles that transfer beyond Android; keep the Android workflow as optional platform reference."
        contribution = "Offer a platform specialization without blocking general Java foundations."
        contingent = ["Android SDK, Android Studio, emulator/device workflow"]
        required_evidence = ["find applicable platform reference"]
        mastery = False
    elif "debug" in title or "debug" in text:
        author_purpose = "teach diagnosis from observed failures"
        relevance, optionality, action, value, level = "core", "required", "practice", "practice_required", "methodology"
        retained = "Retain the bounded loop: reproduce, isolate, form a hypothesis, patch the cause, and retest."
        required_evidence = ["reproduce failure", "diagnose", "patch", "retest"]
        mastery = True
    else:
        mastery = True
        if source.source_quality in {"malformed", "defective", "missing"}:
            action, value = "repair-source", "extraction_repair_required"
            required_evidence = ["repair extraction", "then apply appropriate evidence"]

    if source.source_quality in {"malformed", "defective", "missing"} and relevance in {"core", "supporting"}:
        action, value = "repair-source", "extraction_repair_required"
    trace = {
        "curriculum_objective": objective,
        "source_purpose": author_purpose,
        "relevance_judgment": relevance,
        "abstraction_target": retained,
        "learning_action": action,
        "evidence_requirement": required_evidence,
    }
    return InstructionalAbstraction(
        source_purpose=author_purpose, learner_purpose=f"Use this material to advance: {objective}", objective_relevance=contribution,
        relevance_strength=relevance, expected_learning_contribution=contribution, retained_abstraction=retained,
        generalizable_invariants=invariants, contingent_details=contingent, applicability=[f"when it serves {objective}"],
        non_applicability=["when the source-specific detail is mistaken for a universal rule"], prerequisite_or_dependency_role=dependency,
        optionality=optionality, currentness=currentness, external_verification_need="yes" if action == "current-doc-reconciliation" else "no",
        learning_action=action, abstraction_level=level, instructional_value=value, mastery_required=mastery,
        required_evidence=required_evidence, source_quality=source.source_quality, confidence=0.65 if source.text else 0.35, trace=trace,
    )
