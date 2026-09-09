from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol

from .learner_contract import LearnerContractError, validate_learner_response


class LearnerError(LearnerContractError):
    pass


class Learner(Protocol):
    provider: str
    session_id: str

    def answer(self, *, task: str, material: str = "", revision: str = "") -> dict[str, Any]: ...


@dataclass
class HostLLMLearner:
    """Adapter for Hermes' host-owned ``ctx.llm`` auxiliary task surface.

    No provider or model is selected here. The host decides routing and may
    expose a callable accepting ``prompt`` or ``messages``; structured output
    is validated by the same learner contract as external learners.
    """

    llm: Callable[..., Any]
    session_id: str = ""
    provider: str = "host"
    model_id: str = ""

    def answer(self, *, task: str, material: str = "", revision: str = "") -> dict[str, Any]:
        user = "TASK:\n" + task
        if material:
            user += "\\n\\nINSTRUCTIONAL MATERIAL:\n" + material
        if revision:
            user += "\\n\\nREVISION FEEDBACK (no answer key):\\n" + revision
        messages = [{"role": "system", "content": "Return JSON only with a responses list; do not invent hidden answers."}, {"role": "user", "content": user}]
        try:
            raw = self.llm(messages=messages)
        except TypeError:
            try:
                raw = self.llm(prompt=user)
            except TypeError:
                raw = self.llm(user)
        if isinstance(raw, Mapping):
            payload: Any = raw.get("output", raw.get("content", raw))
            if isinstance(payload, Mapping) and "choices" in payload:
                payload = payload["choices"][0]["message"].get("content", "")
        else:
            payload = raw
        if isinstance(payload, str):
            text = payload.strip()
            fenced = re.fullmatch(r"```(?:json)?\\s*(.*?)\\s*```", text, re.DOTALL | re.IGNORECASE)
            try:
                payload = json.loads(fenced.group(1) if fenced else text)
            except json.JSONDecodeError as exc:
                raise LearnerError("host ctx.llm did not return JSON", code="provider_response_not_json") from exc
        if not isinstance(payload, Mapping):
            raise LearnerError("host ctx.llm returned a non-object", code="provider_response_not_json")
        try:
            return validate_learner_response(payload)
        except LearnerContractError as exc:
            raise LearnerError(str(exc), code=exc.code, details=exc.details) from exc


@dataclass
class OpenAICompatibleLearner:
    """Provider-backed learner using an OpenAI-compatible chat endpoint.

    The endpoint produces the learner's responses; this class does not contain
    the target answer key or a domain-specific classifier.
    """

    endpoint: str
    model: str
    provider: str = "openai-compatible"
    session_id: str = ""
    timeout: float = 180.0
    max_tokens: int = 1800
    format_retry_attempts: int = 1
    requester: Callable[[str, bytes, dict[str, str], float], bytes] | None = None

    def _request_once(self, messages: list[dict[str, str]], *, task: str = "") -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "stream": False,
            "response_format": {"type": "json_object"},
            "chat_template_kwargs": {"enable_thinking": False},
        }
        body = json.dumps(payload, ensure_ascii=False).encode()
        if self.requester is not None:
            raw = self.requester(self.endpoint.rstrip("/") + "/v1/chat/completions", body, {"Content-Type": "application/json"}, self.timeout)
        else:
            request = urllib.request.Request(self.endpoint.rstrip("/") + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
        response_payload: Any = {}
        try:
            raw_text = raw.decode("utf-8")
            direct_fenced = re.fullmatch(r"\s*```[^\r\n\\]*(?:\r?\n|\\n)(.*?)(?:\r?\n|\\n)```\s*", raw_text, re.DOTALL)
            if direct_fenced:
                response_payload = {"direct_fenced_response": True}
                content = direct_fenced.group(1).strip()
            else:
                response_payload = json.loads(raw)
                content = response_payload["choices"][0]["message"].get("content") or ""
            try:
                parsed = json.loads(content)
                # Some OpenAI-compatible backends JSON-encode the requested
                # JSON object a second time. Unwrap exactly one string layer;
                # never recursively guess through arbitrary provider output.
                if isinstance(parsed, str):
                    parsed = json.loads(parsed)
            except json.JSONDecodeError:
                # A few compatible servers escape an already JSON-encoded
                # object once more inside message.content. Decode that one
                # transport layer only, then parse the resulting object.
                if '\\"' in content:
                    try:
                        decoded_content = content.encode("utf-8").decode("unicode_escape").strip()
                        decoded_fenced = re.fullmatch(r"```[^\r\n]*\r?\n(.*?)\r?\n?```", decoded_content, re.DOTALL)
                        parsed = json.loads(decoded_fenced.group(1).strip() if decoded_fenced else decoded_content)
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        parsed = None
                    if isinstance(parsed, Mapping):
                        pass
                    elif isinstance(parsed, str):
                        parsed = json.loads(parsed)
                    else:
                        parsed = None
                else:
                    parsed = None
                if isinstance(parsed, Mapping):
                    pass
                else:
                    fenced = re.fullmatch(r"\s*```[^\n]*\n(.*?)\n?```\s*", content, re.DOTALL)
                    case_ids = re.findall(r"(?:^|\|)case=([^|\s]+)", task)
                    if fenced and len(case_ids) == 1 and fenced.group(1).strip():
                        fenced_body = fenced.group(1).strip()
                        try:
                            fenced_json = json.loads(fenced_body)
                        except json.JSONDecodeError:
                            fenced_json = None
                        if isinstance(fenced_json, Mapping):
                            parsed = fenced_json
                        else:
                            parsed = {"responses": [{"case": case_ids[0], "answer": fenced_body}]}
                    else:
                        start, end = content.find("{"), content.rfind("}")
                        if start < 0 or end <= start:
                            raise
                        parsed = json.loads(content[start:end + 1])
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise LearnerError(
                "provider did not return a JSON learner response",
                code="provider_response_not_json",
                details={
                    "response_keys": sorted(response_payload) if isinstance(response_payload, Mapping) else [],
                    "raw_response": raw[:12000].decode("utf-8", errors="replace"),
                    "raw_response_truncated": len(raw) > 12000,
                },
            ) from exc
        if isinstance(parsed, list) and parsed and all(isinstance(item, Mapping) for item in parsed):
            payload_keys = {
                "answer", "artifact", "candidate_artifact", "candidate_artifact_or_answer",
                "candidate_answer", "judgment", "not_applicable", "response",
            }
            if all(
                isinstance(item.get("case", item.get("case_id")), str)
                and item.get("case", item.get("case_id")).strip()
                and len([key for key in payload_keys if key in item]) == 1
                for item in parsed
            ):
                parsed = {"responses": parsed}
        if isinstance(parsed, Mapping) and "responses" not in parsed:
            case_ids = re.findall(r"(?:^|\|)case=([^|\s]+)", task)
            if len(case_ids) == 1:
                parsed = {"responses": [{"case": case_ids[0], "answer": dict(parsed)}]}
        if isinstance(parsed, Mapping) and isinstance(parsed.get("responses"), list):
            payload_keys = {
                "answer", "artifact", "candidate_artifact", "candidate_artifact_or_answer",
                "candidate_answer", "judgment", "not_applicable", "response",
            }
            repaired_responses: list[dict[str, Any]] = []
            repaired = False
            for item in parsed["responses"]:
                if not isinstance(item, Mapping):
                    repaired_responses.append(item)
                    continue
                mapping_answer = item.get("answer")
                if isinstance(mapping_answer, Mapping):
                    nested_case = mapping_answer.get("case", mapping_answer.get("case_id"))
                    nested_payloads = [key for key in payload_keys if key in mapping_answer]
                    if nested_case == item.get("case") and nested_payloads == ["answer"] and set(mapping_answer) <= {"case", "case_id", "answer"}:
                        repaired_responses.append({**dict(item), "answer": mapping_answer["answer"]})
                        repaired = True
                        continue
                if not isinstance(mapping_answer, str):
                    repaired_responses.append(dict(item) if isinstance(item, Mapping) else item)
                    continue
                try:
                    nested = json.loads(mapping_answer)
                except json.JSONDecodeError:
                    repaired_responses.append(dict(item))
                    continue
                if isinstance(nested, Mapping) and isinstance(nested.get("responses"), list):
                    nested = nested["responses"]
                if not isinstance(nested, list) or len(nested) != 1 or not isinstance(nested[0], Mapping):
                    repaired_responses.append(dict(item))
                    continue
                nested_item = nested[0]
                nested_case = nested_item.get("case", nested_item.get("case_id"))
                nested_payloads = [key for key in payload_keys if key in nested_item]
                if nested_case != item.get("case") or len(nested_payloads) != 1:
                    repaired_responses.append(dict(item))
                    continue
                repaired_responses.append({**dict(item), "answer": nested_item[nested_payloads[0]]})
                repaired = True
            if repaired:
                parsed = {**dict(parsed), "responses": repaired_responses}
        try:
            normalized_result = validate_learner_response(parsed)
        except LearnerContractError as exc:
            details = {
                **exc.details,
                "parsed_type": type(parsed).__name__,
                "parsed_keys": sorted(parsed) if isinstance(parsed, Mapping) else [],
                "parsed_preview": json.dumps(parsed, ensure_ascii=False, default=str)[:4000],
            }
            raise LearnerError(str(exc), code=exc.code, details=details) from exc
        result = {**normalized_result, "_provider_response": response_payload}
        if "procedure" in parsed:
            procedure: Any = parsed["procedure"]
            if isinstance(procedure, list) and all(isinstance(step, str) and step.strip() for step in procedure):
                result["procedure"] = "\n".join(step.strip() for step in procedure)
            elif not isinstance(procedure, str):
                raise LearnerError(
                    "learner procedure must be a string or a nonempty list of strings",
                    code="procedure_invalid",
                )
        return result

    def _request(self, messages: list[dict[str, str]], *, task: str = "") -> dict[str, Any]:
        if not isinstance(self.format_retry_attempts, int) or isinstance(self.format_retry_attempts, bool) or self.format_retry_attempts < 0:
            raise ValueError("format_retry_attempts must be a non-negative integer")
        current_messages = list(messages)
        for attempt in range(self.format_retry_attempts + 1):
            try:
                return self._request_once(current_messages, task=task)
            except LearnerError as exc:
                if exc.code != "provider_response_not_json" or attempt >= self.format_retry_attempts:
                    raise
                retry_instruction = (
                    "Formatting retry only: return the same task answer as one valid JSON object. "
                    "Use exactly the required responses shape, with no prose, markdown, commentary, "
                    "or hidden solution material. Do not change the task interpretation."
                )
                current_messages = [
                    *current_messages[:-1],
                    {
                        **current_messages[-1],
                        "content": current_messages[-1]["content"] + "\n\n" + retry_instruction,
                    },
                ]
        raise RuntimeError("learner request retry loop exhausted unexpectedly")

    def answer(self, *, task: str, material: str = "", revision: str = "") -> dict[str, Any]:
        system = (
            "You are the learner in a controlled study. Return JSON only with key responses and, "
            "when requested, procedure. "
            "responses is a list of objects shaped exactly as {case: <requested case identifier>, answer: <artifact or answer payload>}; always put the complete payload under the single answer key, never as sibling fields such as candidate_artifact, judgment, or dependency_bug. "
            "Answer every case exactly once. Do not claim to know hidden answers. Use the "
            "supplied production classes and interfaces as already defined: never declare, "
            "redefine, reimplement, stub, or mock their type definitions in your answer. "
            "Treat any REQUIRED FIELDS section as an artifact schema: include every named field "
            "as a top-level key inside answer, using the exact requested types; never nest, "
            "rename, or replace those fields with a summary. "
            "When a task supplies required inputs and public behaviors, verify the exact "
            "observable output for each required input before relying on a classification, "
            "validity flag, or other derived summary alone. "
            "instructional material when supplied. Before responding, silently self-check every "
            "explicit task requirement, including requested coverage, output format, and applicability. "
            "For code or test artifacts, make every stated behavior observable with a direct assertion "
            "or executable check; for stateful or idempotent operations, assert the relevant state before "
            "and after each invocation, and for boundary rules assert the exact boundary and adjacent outcomes. "
            "Do not emit the checklist or contradictory labels."
        )
        user = "TASK:\n" + task
        if material:
            user += "\n\nINSTRUCTIONAL MATERIAL:\n" + material
        if revision:
            user += "\n\nREVISION FEEDBACK (no answer key):\n" + revision
        return self._request([{"role": "system", "content": system}, {"role": "user", "content": user}], task=task)
