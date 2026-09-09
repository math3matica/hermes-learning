#!/usr/bin/env python3
"""Run frozen V5 with an evidence-derived staged abstraction procedure.

V6 changes only the provider procedure relative to V5.1: it asks Qwen to make
source comprehension, source adequacy, bounded inference, relevance, invariant,
instructional-role, action, and evidence decisions explicitly before emitting
the same final abstraction fields. The repaired lossless representation path
and evaluator are reused. Historical V5/V5.1 artifacts are never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rex_learning import OpenAICompatibleLearner, normalize_integrated_answer  # noqa: E402
from run_live_instructional_abstraction_qualification_v5 import (  # noqa: E402
    ENDPOINT,
    MODEL,
    PROTOCOL,
    answer_object,
    evaluate,
)

INTEGRATED_OPERATION = "instructional_abstraction_integrated"
PROCEDURE = "rex-learning-staged-instructional-abstraction-v1"


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def staged_integrated_prompt(case: dict[str, Any]) -> str:
    return f"""Perform one staged instructional-abstraction analysis. Return JSON only with outer shape {{\"responses\":[{{\"case\":\"{INTEGRATED_OPERATION}\",\"answer\":{{...}}}}]}}.

Use the following general procedure before producing the final fields. Do not mention or infer hidden benchmark requirements. Do not add facts merely because they are common in the domain.

1. SOURCE COMPREHENSION: state only what the visible source actually asserts. Keep source purpose separate from learner purpose.
2. SOURCE ADEQUACY: determine whether the visible source is sufficient for the requested learning action. If correctness-critical information is missing, preserve the known boundary and route to repair or authoritative verification instead of inventing a completion.
3. SUPPORTED EVIDENCE: separate source-established claims, intent-established facts, bounded inferences, unknowns, and external-verification requirements. Attach support references where the output schema permits them.
4. RELEVANCE: select what matters for the visible curriculum intent. Discard irrelevant context and damaged material that is not part of the target capability.
5. INVARIANT PRESERVATION: list rules, exact details, ordering, polarity, flags, values, and safeguards that must not be changed. Keep example-specific details separate from transferable ideas.
6. BOUNDED INFERENCE: state only transfers justified by the source and intent. Do not broaden a single example into a universal rule, add unstated prerequisites, or mutate an operation into a different one.
7. INSTRUCTIONAL ROLE: classify what remains as concept, rule, reference, procedure, example, exercise, prerequisite, context, obsolete detail, or irrelevant material.
8. LEARNING ACTION: distinguish knowledge to retain from capability requiring practice, lookup, implementation, comparison, or verification. Match the action to the intent and source adequacy.
9. LEARNING EVIDENCE: specify what would demonstrate acquisition without inventing source facts. Preserve currentness and uncertainty boundaries.
10. FINAL CHECK: before emitting the final abstraction, check source/instruction separation, exact details, ordering, boundedness, action routing, and evidence path.

Include a compact `staged_analysis` object recording the decisions from steps 1-10. Then include the final abstraction fields: source purpose, source facts, source invariant, exact details, transferable abstractions, contingent/example-specific details, goal relevance, dependency/optionality, applicability, currentness, learning action, evidence path, uncertainties, and claims. Use epistemic categories SOURCE_ESTABLISHED, INTENT_ESTABLISHED, BOUNDED_INFERENCE, UNKNOWN, and EXTERNAL_VERIFICATION_REQUIRED explicitly where appropriate. Claims must include ids, claim text, support_type, and support_refs. Return JSON only.

VISIBLE SOURCE:
{json.dumps(case['source'], ensure_ascii=False, sort_keys=True)}

VISIBLE CURRICULUM INTENT:
{json.dumps(case['intent'], ensure_ascii=False, sort_keys=True)}
"""


def canonical_surface(answer: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "source_purpose", "source_facts", "source_invariant", "exact_details",
        "transferable_abstractions", "contingent_or_example_specific_details",
        "goal_relevance", "dependency_or_optionality", "applicability",
        "currentness", "currentness_applicable", "currentness_status",
        "learning_action", "evidence_path", "uncertainties", "bounded_inferences",
        "claims", "staged_analysis", "representation_fidelity",
    )
    return {"surface_schema": "rex-learning-integrated-final-surface-v3", **{key: answer.get(key) for key in keys}}


def first_failure(record: dict[str, Any]) -> tuple[str, str | None]:
    fidelity = record["representation_fidelity"]
    semantic_pass = bool(record["provider_semantic"]["judgment"].get("pass"))
    end_pass = bool(record["end_to_end"]["judgment"].get("pass"))
    if fidelity.get("status") == "SCHEMA_INCOMPATIBLE":
        return "SCHEMA_ADAPTER_LOSS", "QWEN_SEMANTIC_FAILURE" if not semantic_pass else None
    if not semantic_pass:
        return "QWEN_SEMANTIC_FAILURE", None
    if not end_pass:
        return "PROJECTION_LOSS", None
    return "NONE", None


def run_case(case: dict[str, Any], evaluate_case: bool) -> dict[str, Any]:
    session = f"qwen-v6-{case['id']}-{time.time_ns()}"
    learner = OpenAICompatibleLearner(ENDPOINT, MODEL, provider="qwen-local", session_id=session, timeout=900, max_tokens=2200)
    prompt = staged_integrated_prompt(case)
    raw_envelope = learner.answer(task=prompt)
    raw_answer = answer_object(raw_envelope, INTEGRATED_OPERATION)
    canonical = normalize_integrated_answer(raw_answer)
    record: dict[str, Any] = {
        "schema": "rex-learning-v6-case-v1",
        "procedure": {"name": PROCEDURE, "prompt_sha256": digest(prompt)},
        "case": case,
        "provider": {"provider": "qwen-local", "model": MODEL, "endpoint": ENDPOINT, "session_id": session},
        "prompt": {"sha256": digest(prompt), "text": prompt},
        "raw_provider_response": raw_envelope.get("_provider_response"),
        "parsed_provider_answer": raw_answer,
        "canonical": canonical,
        "representation_fidelity": canonical["representation_fidelity"],
        "hashes": {
            "raw_provider_response_sha256": digest(raw_envelope.get("_provider_response")),
            "parsed_provider_answer_sha256": digest(raw_answer),
            "canonical_sha256": digest(canonical),
        },
        "critic": {"enabled": False, "reason": "excluded from clean staged-procedure baseline"},
    }
    if evaluate_case:
        faithful = canonical["representation_fidelity"]["faithful_surface"]
        product = canonical_surface(canonical)
        record["provider_semantic"] = evaluate(case, "V6-staged-provider-semantic", faithful)
        record["end_to_end"] = evaluate(case, "V6-staged-end-to-end", product)
        primary, secondary = first_failure(record)
        record["attribution"] = {"primary_first_failure": primary, "secondary_failure": secondary}
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--case", action="append", dest="case_ids", default=[])
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    run_root = Path(args.run_root)
    run_root.mkdir(parents=True, exist_ok=False)
    protocol_hash = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    cases = [case for case in protocol["cases"] if case["id"] in args.case_ids] if args.case_ids else protocol["cases"]
    cases = cases[:args.limit] if args.limit else cases
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, case in enumerate(cases, 1):
        print(f"v6 {index}/{len(cases)} {case['id']}", flush=True)
        started = time.time()
        try:
            record = run_case(case, args.evaluate)
            record["execution"] = {"status": "completed", "elapsed_seconds": time.time() - started}
            (run_root / f"{case['id']}.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            records.append(record)
        except Exception as exc:
            failure = {"case": case, "execution": {"status": "failed", "error": repr(exc), "elapsed_seconds": time.time() - started}}
            (run_root / f"{case['id']}.failed.json").write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
            failures.append(failure)
            print(f"FAILED {case['id']}: {exc}", file=sys.stderr, flush=True)
    manifest = {
        "schema": "rex-learning-goal-relative-instructional-abstraction-qualification-v6-run",
        "protocol": str(PROTOCOL), "protocol_sha256": protocol_hash,
        "procedure": PROCEDURE,
        "provider": {"provider": "qwen-local", "model": MODEL, "endpoint": ENDPOINT, "temperature": 0, "max_tokens": 2200},
        "evaluator": {"provider": "openai-codex", "model": "gpt-5.6-luna"},
        "critic": {"enabled": False, "reason": "clean staged-procedure baseline"},
        "cases_requested": len(cases), "cases_completed": len(records), "execution_failures": len(failures),
        "status": "completed" if not failures else "partial", "run_root": str(run_root),
        "records": [str(run_root / f"{item['case']['id']}.json") for item in records],
    }
    (run_root / "run-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if args.evaluate:
        report: dict[str, Any] = {"schema": "rex-learning-v6-qualification-report-v1", "protocol_sha256": protocol_hash, "procedure": PROCEDURE, "cases": len(records), "execution_failures": len(failures), "provider_semantic_pass": 0, "representation_fidelity_pass": 0, "end_to_end_pass": 0, "first_failure_counts": {}}
        for item in records:
            if item["provider_semantic"]["judgment"].get("pass"): report["provider_semantic_pass"] += 1
            if item["representation_fidelity"]["status"] != "SCHEMA_INCOMPATIBLE": report["representation_fidelity_pass"] += 1
            if item["end_to_end"]["judgment"].get("pass"): report["end_to_end_pass"] += 1
            label = item["attribution"]["primary_first_failure"]
            report["first_failure_counts"][label] = report["first_failure_counts"].get(label, 0) + 1
        for key in ("provider_semantic_pass", "representation_fidelity_pass", "end_to_end_pass"):
            report[key.replace("_pass", "_pass_rate")] = report[key] / len(records) if records else 0
        (run_root / "qualification-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
