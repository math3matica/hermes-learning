from pathlib import Path

import pytest

from rex_learning import (
    CurriculumIntent,
    DeepStudyEngine,
    InstructionalAbstraction,
    LearningError,
    LearningStore,
    SourceContext,
)
from rex_learning.cognitive import CognitiveContextPolicy, CognitiveRouter, ProviderPolicy
from rex_learning.instructional_abstraction import assess_instructional_abstraction
from tests.test_cognitive_architecture import FixtureProvider


def intent(**overrides):
    data = {
        "key": "modern-java-foundations",
        "title": "Modern Java foundations",
        "target_capability": "Develop a durable practical foundation in Java programming.",
        "competence_level": "working foundation",
        "intended_use": ["construct", "debug", "test", "modify Java programs"],
        "intended_transfer": ["new Java domains", "unseen executable tasks"],
        "environment": {"jdk": "17", "os": "Linux"},
        "target_languages": ["Java"],
        "target_tools": ["javac", "java"],
        "temporal_requirements": "prefer enduring knowledge; verify operational freshness",
        "exclusions": ["reproduce a 2018 workstation"],
        "optional_specializations": ["Android", "legacy Java 8 maintenance"],
        "prerequisite_assumptions": [],
        "evaluation_criteria": ["compile", "execute", "test", "explain tradeoffs"],
        "desired_behavioral_change": ["validate inputs", "debug from evidence"],
    }
    data.update(overrides)
    return CurriculumIntent.from_dict(data)


def source(title, text, **extra):
    data = {"source_id": "book", "unit_id": "unit-1", "title": title, "text": text, "source_refs": ["book.xhtml#1"], "hierarchy": {"chapter": "Chapter 1"}}
    data.update(extra)
    return SourceContext.from_dict(data)


def test_curriculum_intent_is_versioned_and_immutable_in_store(tmp_path: Path):
    store = LearningStore(tmp_path / "learning")
    engine = DeepStudyEngine(store)
    first = engine.create_curriculum_intent(intent().to_dict(), provenance={"source": "reconstructed", "evidence_refs": ["java-curriculum-report-v1"]})
    again = engine.create_curriculum_intent(intent().to_dict(), provenance={"source": "reconstructed", "evidence_refs": ["java-curriculum-report-v1"]})
    assert first["id"] == again["id"]
    assert first["version"] == 1
    changed = engine.create_curriculum_intent(intent(key="legacy-java-8", title="Legacy Java 8 maintenance", target_capability="Maintain Java 8 applications" ).to_dict(), provenance={"source": "new objective"})
    assert changed["id"] != first["id"]
    assert len(store.list("curriculum_intents")) == 2


def test_abstraction_separates_author_purpose_from_learner_purpose():
    result = assess_instructional_abstraction(
        intent(),
        source("Installation", "Install JDK 8 and JDK 9 using the download workflow described in 2018."),
    )
    assert result.source_purpose
    assert result.learner_purpose != result.source_purpose
    assert "JDK" in result.retained_abstraction
    assert result.currentness in {"version-bound", "stale-risk"}
    assert result.learning_action in {"current-doc-reconciliation", "reference-only"}
    assert result.relevance_strength in {"supporting", "contextual"}
    assert result.trace["curriculum_objective"] == intent().target_capability


def test_goal_sensitivity_changes_same_setup_judgment():
    material = source("JDK 8 setup", "Configure JDK 8 and the period-specific IDE for this Java 8 project.")
    modern = assess_instructional_abstraction(intent(), material)
    legacy = assess_instructional_abstraction(
        intent(key="legacy", target_capability="Maintain an enterprise application constrained to Java 8", competence_level="maintenance competence", environment={"jdk": "8"}),
        material,
    )
    assert modern.relevance_strength != legacy.relevance_strength
    assert legacy.learning_action in {"deep-study", "execute"}
    assert modern.currentness == "version-bound"


def test_biography_is_contextual_and_cannot_block_mastery():
    result = assess_instructional_abstraction(intent(), source("About the author", "The author has a dog, two children, and enjoys hiking."))
    assert result.learning_action == "contextual-record"
    assert result.instructional_value == "context_only"
    assert result.mastery_required is False


def test_exact_language_semantics_are_not_over_generalized():
    result = assess_instructional_abstraction(intent(), source("Numeric promotion", "In Java, binary numeric promotion converts byte and short operands to int before arithmetic."))
    assert result.abstraction_level in {"syntax-semantics", "concept"}
    assert "int" in result.retained_abstraction
    assert result.learning_action in {"deep-study", "retrieval"}
    assert result.required_evidence == ["reconstruct", "compile/use correctly"]


def test_exercise_and_reference_route_differently():
    exercise = assess_instructional_abstraction(intent(), source("Programming Challenge", "Write a program that groups records and test edge cases."))
    reference = assess_instructional_abstraction(intent(), source("Reference table", "A table of method signatures and return types."))
    assert exercise.learning_action == "practice"
    assert exercise.required_evidence == ["code", "compile", "execute", "tests", "diagnosis"]
    assert reference.learning_action == "reference-only"
    assert reference.instructional_value == "reference_sufficient"


def test_router_accepts_instructional_abstraction_with_intent_visible(tmp_path: Path):
    store = LearningStore(tmp_path / "learning")
    router = CognitiveRouter(
        store,
        providers={"local": FixtureProvider("local", "model-a")},
        policy=ProviderPolicy({"instructional_abstraction": {"provider": "local"}}),
    )
    result = router.run(
        "instructional_abstraction",
        {"curriculum_intent": intent().to_dict(), "source_context": source("Concept", "A durable concept").to_dict()},
        context=CognitiveContextPolicy.for_operation("instructional_abstraction"),
    )
    assert result.provenance["operation_type"] == "instructional_abstraction"
    assert store.list("cognitive_operations")[0]["context_policy"]["source_visible"] is True
    assert store.list("cognitive_operations")[0]["context_policy"]["curriculum_intent_visible"] is True


@pytest.mark.parametrize(
    ("title", "text", "quality", "expected_action"),
    [
        ("Optional sidebar", "An advanced sidebar adds enrichment unrelated to the target capability.", "adequate", "normal-study"),
        ("Old API procedure", "Call a deprecated API to configure the client.", "adequate", "normal-study"),
        ("Input validation exercise", "Reject malformed input and report the failure.", "adequate", "practice"),
        ("Broken technical section", "...", "defective", "repair-source"),
    ],
)
def test_adversarial_fragments_route_by_purpose_not_only_age(title, text, quality, expected_action):
    result = assess_instructional_abstraction(intent(), source(title, text, source_quality=quality))
    assert result.learning_action == expected_action
    assert result.trace["curriculum_objective"]


def test_abstraction_record_requires_provenance_and_preserves_history(tmp_path: Path):
    store = LearningStore(tmp_path / "learning")
    engine = DeepStudyEngine(store)
    record = engine.record_instructional_abstraction(
        intent_id="intent-v1",
        source_id="book",
        unit_id="unit-1",
        source_hash="sha-source",
        abstraction=InstructionalAbstraction.from_dict({
            "source_purpose": "teach a concept", "learner_purpose": "build capability", "objective_relevance": "supports goal",
            "relevance_strength": "core", "expected_learning_contribution": "understanding", "retained_abstraction": "principle",
            "generalizable_invariants": ["invariant"], "contingent_details": [], "applicability": ["when needed"],
            "non_applicability": [], "prerequisite_or_dependency_role": "enables later work", "optionality": "required",
            "currentness": "enduring", "external_verification_need": "no", "learning_action": "deep-study",
            "abstraction_level": "generalized-principle", "instructional_value": "mastery_required", "mastery_required": True,
            "required_evidence": ["explain"], "source_quality": "adequate", "confidence": 0.8,
        }),
        provenance={"provider": "local", "model": "model-a", "operation_id": "op-1", "input_hash": "in", "output_hash": "out", "evidence_refs": ["book.xhtml#1"]},
    )
    assert record["schema"] == "rex-learning-instructional-abstraction-v1"
    assert record["provenance"]["operation_id"] == "op-1"
    with pytest.raises(LearningError, match="provenance"):
        engine.record_instructional_abstraction(intent_id="i", source_id="s", unit_id="u", source_hash="h", abstraction=record, provenance={})
