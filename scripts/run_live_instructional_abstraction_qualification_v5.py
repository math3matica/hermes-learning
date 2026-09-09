#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rex_learning import (  # noqa: E402
    CRITIC_OPERATION,
    INTEGRATED_OPERATION,
    CurriculumIntent,
    OpenAICompatibleLearner,
    SourceContext,
    apply_targeted_critic,
    assess_risks,
    normalize_integrated_answer,
    validate_integrated_answer,
)

OUT = ROOT / "docs/rex-learning-deep-study-evidence/goal-relative-instructional-abstraction-qualification-v5"
PROTOCOL = OUT / "protocol.json"
ENDPOINT = "http://127.0.0.1:8080"
MODEL = "/models/Qwen3.8-27B-UD-Q4_K_XL.gguf"


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def as_intent(case: dict[str, Any]) -> CurriculumIntent:
    return CurriculumIntent.from_dict(case["intent"])


def as_source(case: dict[str, Any]) -> SourceContext:
    data = {"source_id": "v5-holdout", "unit_id": case["id"], "hierarchy": {"domain": case["domain"]}}
    data.update(case["source"])
    return SourceContext.from_dict(data)


def answer_object(result: dict[str, Any], operation: str) -> dict[str, Any]:
    responses = result.get("responses", [])
    if len(responses) != 1 or not isinstance(responses[0].get("answer"), dict):
        raise RuntimeError(f"{operation}: provider did not return one JSON answer")
    return dict(responses[0]["answer"])


def integrated_prompt(case: dict[str, Any]) -> str:
    return f"""Perform one integrated instructional abstraction judgment. Return JSON only with outer shape {{\"responses\":[{{\"case\":\"{INTEGRATED_OPERATION}\",\"answer\":{{...}}}}]}}.

Reason coherently about what the source says, why it exists, what remains invariant, exact details, transferable abstractions, contingent/example-specific details, goal relevance, dependency/optionality, applicability, currentness, learning action, and evidence path. Keep source purpose separate from learner purpose. Use epistemic categories SOURCE_ESTABLISHED, INTENT_ESTABLISHED, BOUNDED_INFERENCE, UNKNOWN, and EXTERNAL_VERIFICATION_REQUIRED explicitly where appropriate. Claims must include id when possible, claim, support_type, and support_refs. Do not invent facts, currentness, or hidden answers. Unknown is not the same as external verification required.

VISIBLE SOURCE:
{json.dumps(case['source'], ensure_ascii=False, sort_keys=True)}

VISIBLE CURRICULUM INTENT:
{json.dumps(case['intent'], ensure_ascii=False, sort_keys=True)}
"""


def critic_prompt(case: dict[str, Any], answer: dict[str, Any], risks: list[dict[str, Any]]) -> str:
    flagged = {"risks": risks, "claims": answer.get("claims", []), "learning_action": answer.get("learning_action", {}), "evidence_path": answer.get("evidence_path", {})}
    return f"""Perform one narrow targeted critique of the flagged integrated instructional-abstraction claims. Return JSON only with outer shape {{\"responses\":[{{\"case\":\"{CRITIC_OPERATION}\",\"answer\":{{...}}}}]}}.

Do not redo the whole analysis. For named target_claims only, choose decision keep, narrow, reject, mark_unknown, or external_verification. Preserve unrelated claims. Include reasoning_basis and support_refs for every revision. Return affected_learning_action or affected_evidence_path only when the flagged issue actually affects it. Do not use hidden references or invent source facts.

VISIBLE SOURCE:
{json.dumps(case['source'], ensure_ascii=False, sort_keys=True)}
VISIBLE CURRICULUM INTENT:
{json.dumps(case['intent'], ensure_ascii=False, sort_keys=True)}
FLAGGED MATERIAL:
{json.dumps(flagged, ensure_ascii=False, sort_keys=True)}
"""


def final_surface(answer: dict[str, Any]) -> dict[str, Any]:
    keys = ("source_purpose", "source_facts", "source_invariant", "exact_details", "transferable_abstractions", "contingent_or_example_specific_details", "goal_relevance", "dependency_or_optionality", "currentness_applicable", "currentness_status", "learning_action", "evidence_path", "uncertainties", "bounded_inferences", "claims")
    return {"surface_schema": "rex-learning-integrated-final-surface-v1", **{key: answer.get(key) for key in keys}}


def evaluate(case: dict[str, Any], arm: str, surface: dict[str, Any]) -> dict[str, Any]:
    prompt = f"""You are an independent evaluator. Evaluate only this final instructional-abstraction surface against the hidden required/prohibited reference. The candidate never sees this evaluator prompt. Return JSON only: {{\"pass\":true/false,\"critical_errors\":[...],\"unsupported_claim_count\":0,\"unsupported_broad_generalization_count\":0,\"dimensions\":{{\"source_purpose\":0,\"source_invariant\":0,\"exact_details\":0,\"relevance\":0,\"dependency_optionality\":0,\"currentness\":0,\"learning_action\":0,\"evidence_path\":0}},\"notes\":\"...\"}}.

CASE:
{json.dumps(case, ensure_ascii=False, sort_keys=True)}
ARM: {arm}
FINAL SURFACE:
{json.dumps(surface, ensure_ascii=False, sort_keys=True)}
"""
    result = subprocess.run(["hermes", "chat", "-Q", "-q", prompt, "-m", "gpt-5.6-luna", "--provider", "openai-codex", "--ignore-rules", "--ignore-user-config", "--max-turns", "1", "--source", "goal-relative-abstraction-v5-evaluator"], cwd=ROOT, text=True, capture_output=True, timeout=1200, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr[-2000:])
    raw = result.stdout.strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        raise RuntimeError("evaluator did not return JSON")
    return {"provider": "openai-codex", "model": "gpt-5.6-luna", "input_hash": digest(prompt), "candidate_hidden_reference_visible": False, "surface": surface, "judgment": json.loads(raw[start:end + 1])}


def run_arm(case: dict[str, Any], arm: str) -> dict[str, Any]:
    session = f"qwen-v5-{arm}-{case['id']}-{time.time_ns()}"
    learner = OpenAICompatibleLearner(ENDPOINT, MODEL, provider="qwen-local", session_id=session, timeout=900, max_tokens=2200)
    task = integrated_prompt(case)
    raw = learner.answer(task=task)
    initial = normalize_integrated_answer(answer_object(raw, INTEGRATED_OPERATION))
    record: dict[str, Any] = {"arm": arm, "session_id": session, "provider": "qwen-local", "model": MODEL, "operation": INTEGRATED_OPERATION, "prompt_hash": digest(task), "initial": initial, "raw_provider_response": raw.get("_provider_response"), "provider_calls": 1}
    if arm == "A":
        final = initial
        record["validation"] = None
        record["risks"] = []
        record["critic"] = None
    else:
        source, intent = as_source(case), as_intent(case)
        validated = validate_integrated_answer(initial, source, intent)
        record["validation"] = {"issues": validated.issues, "quarantined_claim_ids": validated.quarantined_claim_ids, "validator": validated.canonical.get("integrity", {}).get("validator")}
        final = validated.canonical
        risks = assess_risks(final, source, intent) if arm == "C" else []
        record["risks"] = risks
        record["critic"] = None
        if arm == "C" and risks:
            critic_task = critic_prompt(case, final, risks)
            before_critic = final
            critic_raw = learner.answer(task=critic_task)
            critic = answer_object(critic_raw, CRITIC_OPERATION)
            final = apply_targeted_critic(final, critic)
            record["critic"] = {"input": {"risks": risks, "target_claims": before_critic.get("claims", []), "learning_action": before_critic.get("learning_action", {}), "evidence_path": before_critic.get("evidence_path", {})}, "output": critic, "prompt_hash": digest(critic_task), "provider_calls": 1, "raw_provider_response": critic_raw.get("_provider_response")}
            record["provider_calls"] = 2
    record["final"] = final
    record["final_surface"] = final_surface(final)
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--run-root", default="")
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    run_root = Path(args.run_root) if args.run_root else OUT / f"run-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{time.time_ns() % 1000000:06d}"
    run_root.mkdir(parents=True, exist_ok=False)
    protocol_hash = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    (run_root / "protocol_sha256.txt").write_text(protocol_hash + "\n", encoding="utf-8")
    cases = protocol["cases"][:args.limit] if args.limit else protocol["cases"]
    records, failures = [], []
    for index, case in enumerate(cases, 1):
        print(f"case {index}/{len(cases)} {case['id']}", flush=True)
        started = time.time()
        record = {"case": case, "arms": {}}
        try:
            for arm in ("A", "B", "C"):
                print(f"  arm {arm}", flush=True)
                arm_record = run_arm(case, arm)
                if args.evaluate:
                    arm_record["evaluation"] = evaluate(case, arm, arm_record["final_surface"])
                record["arms"][arm] = arm_record
            record["execution"] = {"status": "completed", "elapsed_seconds": time.time() - started}
            (run_root / f"{case['id']}.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            records.append(record)
        except Exception as exc:
            record["execution"] = {"status": "failed", "error": repr(exc), "elapsed_seconds": time.time() - started, "hidden_reference_rendered_to_candidate": False}
            (run_root / f"{case['id']}.failed.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            failures.append(record)
            print(f"FAILED {case['id']}: {exc}", file=sys.stderr, flush=True)
    manifest = {"schema":"rex-learning-goal-relative-instructional-abstraction-qualification-v5-run","protocol":str(PROTOCOL),"protocol_sha256":protocol_hash,"provider":{"provider":"qwen-local","model":MODEL,"endpoint":ENDPOINT,"temperature":0},"evaluator":{"provider":"openai-codex","model":"gpt-5.6-luna"},"cases_requested":len(cases),"cases_completed":len(records),"execution_failures":len(failures),"status":"completed" if not failures else "partial","run_root":str(run_root),"arms":["A","B","C"],"records":[str(run_root/f"{r['case']['id']}.json") for r in records]}
    (run_root / "run-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
