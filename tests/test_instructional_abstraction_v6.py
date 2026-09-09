import json

from rex_learning.lightweight_abstraction import normalize_integrated_answer
from scripts.run_live_instructional_abstraction_qualification_v6 import canonical_surface, staged_integrated_prompt


def test_mapping_lists_remain_semantically_projectable_lists():
    raw = {
        "invariants": [
            {"claim": "Use the fixed quantum", "support_refs": ["v5://case"]},
            {"claim": "Report completion order", "support_refs": ["v5://case"]},
        ],
        "transferable_abstractions": [
            {"claim": "Track state transitions", "support_refs": ["cap"]},
            {"claim": "Report the resulting order", "support_refs": ["cap"]},
        ],
    }
    normalized = normalize_integrated_answer(raw)

    assert isinstance(normalized["source_invariant"], list)
    assert normalized["source_invariant"] == raw["invariants"]
    assert isinstance(normalized["transferable_abstractions"], list)
    assert normalized["transferable_abstractions"] == raw["transferable_abstractions"]
    assert normalized["representation_fidelity"]["operations"]
    assert normalized["representation_fidelity"]["incompatibilities"] == []


def test_scalar_lists_for_list_like_abstraction_fields_are_preserved():
    raw = {
        "source_facts": ["The source states a rule."],
        "transferable_abstractions": ["Apply the rule to a new proposition."],
        "contingent_example_specific_details": [],
    }
    normalized = normalize_integrated_answer(raw)

    assert normalized["source_facts"] == raw["source_facts"]
    assert normalized["transferable_abstractions"] == raw["transferable_abstractions"]
    assert normalized["representation_fidelity"]["incompatibilities"] == []


def test_v6_prompt_requests_general_staged_analysis_without_case_answers():
    case = {
        "id": "v5-08",
        "source": {"text": "Simulate a task using a fixed parameter."},
        "intent": {"target_capability": "Analyze the task."},
    }
    prompt = staged_integrated_prompt(case)
    lower = prompt.casefold()

    for gate in (
        "source comprehension",
        "source adequacy",
        "invariant preservation",
        "bounded inference",
        "instructional role",
        "learning action",
        "evidence",
    ):
        assert gate in lower
    assert "v5-08" not in prompt
    assert "fixed quantum" not in prompt
    assert '"responses"' in prompt


def test_final_projection_preserves_semantic_fields_omitted_by_old_surface():
    answer = {
        "applicability": {"claim": "Only the described service", "support_refs": ["v5://case"]},
        "currentness": {"applicable": False, "status": "NOT_APPLICABLE"},
        "bounded_inferences": ["No broader rule is established."],
        "source_facts": ["A source fact"],
    }
    projected = canonical_surface(answer)
    assert projected["applicability"] == answer["applicability"]
    assert projected["currentness"] == answer["currentness"]
    assert projected["bounded_inferences"] == answer["bounded_inferences"]
