#!/usr/bin/env python3
"""Run frozen V5 through a lossless, critic-free attribution path.

V5.1 keeps the frozen protocol and provider prompt, but evaluates two distinct
surfaces: the faithful provider answer (capability) and the adapted canonical
answer (product). The adapter records every shape conversion and retains the
raw answer in each case artifact.
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
    integrated_prompt,
)

OUT = ROOT / "docs/rex-learning-deep-study-evidence/goal-relative-instructional-abstraction-qualification-v5"


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def canonical_surface(answer: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "source_purpose", "source_facts", "source_invariant", "exact_details",
        "transferable_abstractions", "contingent_or_example_specific_details",
        "goal_relevance", "dependency_or_optionality", "currentness_applicable",
        "currentness_status", "learning_action", "evidence_path", "uncertainties",
        "claims", "representation_fidelity",
    )
    return {"surface_schema": "rex-learning-integrated-final-surface-v2", **{key: answer.get(key) for key in keys}}


def first_failure(record: dict[str, Any]) -> tuple[str, str | None]:
    fidelity = record["representation_fidelity"]
    semantic_pass = bool(record["provider_semantic"]["judgment"].get("pass"))
    end_pass = bool(record["end_to_end"]["judgment"].get("pass"))
    status = fidelity.get("status")
    if status == "SCHEMA_INCOMPATIBLE":
        return "SCHEMA_ADAPTER_LOSS", "PROVIDER_SEMANTIC_FAILURE" if not semantic_pass else None
    if not semantic_pass:
        return "QWEN_SEMANTIC_FAILURE", None
    if not end_pass:
        return "PROJECTION_LOSS", None
    return "NONE", None


def run_case(case: dict[str, Any], evaluate_case: bool) -> dict[str, Any]:
    session = f"qwen-v51-{case['id']}-{time.time_ns()}"
    learner = OpenAICompatibleLearner(ENDPOINT, MODEL, provider="qwen-local", session_id=session, timeout=900, max_tokens=2200)
    prompt = integrated_prompt(case)
    raw_envelope = learner.answer(task=prompt)
    raw_answer = answer_object(raw_envelope, "instructional_abstraction_integrated")
    canonical = normalize_integrated_answer(raw_answer)
    record: dict[str, Any] = {
        "schema": "rex-learning-v51-case-v1",
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
        "critic": {"enabled": False, "reason": "excluded from clean V5.1 baseline"},
    }
    if evaluate_case:
        faithful = canonical["representation_fidelity"]["faithful_surface"]
        product = canonical_surface(canonical)
        record["provider_semantic"] = evaluate(case, "V5.1-provider-semantic", faithful)
        record["end_to_end"] = evaluate(case, "V5.1-end-to-end", product)
        record["attribution"] = {}
        primary, secondary = first_failure(record)
        record["attribution"] = {"primary_first_failure": primary, "secondary_failure": secondary}
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    run_root = Path(args.run_root)
    run_root.mkdir(parents=True, exist_ok=False)
    protocol_hash = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    cases = protocol["cases"][:args.limit] if args.limit else protocol["cases"]
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, case in enumerate(cases, 1):
        print(f"v5.1 {index}/{len(cases)} {case['id']}", flush=True)
        started = time.time()
        try:
            record = run_case(case, args.evaluate)
            record["execution"] = {"status": "completed", "elapsed_seconds": time.time() - started}
            path = run_root / f"{case['id']}.json"
            path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            records.append(record)
        except Exception as exc:
            failure = {"case": case, "execution": {"status": "failed", "error": repr(exc), "elapsed_seconds": time.time() - started}}
            (run_root / f"{case['id']}.failed.json").write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
            failures.append(failure)
            print(f"FAILED {case['id']}: {exc}", file=sys.stderr, flush=True)
    manifest = {
        "schema": "rex-learning-goal-relative-instructional-abstraction-qualification-v51-run",
        "protocol": str(PROTOCOL), "protocol_sha256": protocol_hash,
        "provider": {"provider": "qwen-local", "model": MODEL, "endpoint": ENDPOINT, "temperature": 0, "max_tokens": 2200},
        "evaluator": {"provider": "openai-codex", "model": "gpt-5.6-luna"},
        "critic": {"enabled": False, "reason": "clean baseline"},
        "cases_requested": len(cases), "cases_completed": len(records), "execution_failures": len(failures),
        "status": "completed" if not failures else "partial", "run_root": str(run_root),
        "records": [str(run_root / f"{item['case']['id']}.json") for item in records],
    }
    (run_root / "run-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if args.evaluate:
        report: dict[str, Any] = {"schema": "rex-learning-v51-qualification-report-v1", "protocol_sha256": protocol_hash, "cases": len(records), "execution_failures": len(failures), "provider_semantic_pass": 0, "representation_fidelity_pass": 0, "end_to_end_pass": 0, "first_failure_counts": {}}
        for item in records:
            if item["provider_semantic"]["judgment"].get("pass"):
                report["provider_semantic_pass"] += 1
            if item["representation_fidelity"]["status"] != "SCHEMA_INCOMPATIBLE":
                report["representation_fidelity_pass"] += 1
            if item["end_to_end"]["judgment"].get("pass"):
                report["end_to_end_pass"] += 1
            label = item["attribution"]["primary_first_failure"]
            report["first_failure_counts"][label] = report["first_failure_counts"].get(label, 0) + 1
        for key in ("provider_semantic_pass", "representation_fidelity_pass", "end_to_end_pass"):
            report[key.replace("_pass", "_pass_rate")] = report[key] / len(records) if records else 0
        (run_root / "qualification-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
    (run_root / "run-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
