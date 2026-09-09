from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


_PROVIDER_METADATA = frozenset({"_provider_response", "usage", "timings", "model", "id", "object", "created", "system_fingerprint"})


def _render(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return ""
    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)
    return str(value)


def canonicalize_learner_answer(result: Any) -> str:
    """Return the complete semantic answer, excluding provider envelopes.

    The learner contract is intentionally small: response items and explicit
    answer/procedure fields are semantic content; provider metadata is not.
    Ordering is preserved for response lists and keys are stable for nested
    structured values, making the result inspectable and hashable.
    """
    if isinstance(result, str):
        answer = result.strip()
        if not answer:
            raise ValueError("learner answer is empty")
        return answer
    if not isinstance(result, Mapping):
        raise ValueError("learner result must be a mapping or string")

    sections: list[str] = []
    responses = result.get("responses")
    if responses is not None:
        if not isinstance(responses, list):
            raise ValueError("learner responses must be a list")
        for index, item in enumerate(responses):
            if not isinstance(item, Mapping):
                raise ValueError(f"learner response {index} must be an object")
            label = item.get("case", item.get("case_id", f"response_{index + 1}"))
            if not isinstance(label, str) or not label.strip():
                raise ValueError(f"learner response {index} has no label")
            if "answer" not in item:
                raise ValueError(f"learner response {label} has no answer")
            rendered = _render(item["answer"])
            if not rendered:
                raise ValueError(f"learner response {label} is empty")
            sections.append(f"{label.strip()}:\n{rendered}")

    for key in ("answer", "procedure", "final_answer", "rationale", "reasoning", "conclusion"):
        if key in result and result[key] not in (None, "", [], {}):
            rendered = _render(result[key])
            if rendered and not any(section.startswith(f"{key}:") for section in sections):
                sections.append(f"{key}:\n{rendered}")

    if not sections:
        semantic_keys = [key for key in result if key not in _PROVIDER_METADATA and key not in {"responses"}]
        if semantic_keys:
            sections.append(json.dumps({key: result[key] for key in semantic_keys}, ensure_ascii=False, sort_keys=True, indent=2))
    if not sections:
        raise ValueError("learner result contains no semantic answer")
    return "\n\n".join(sections)