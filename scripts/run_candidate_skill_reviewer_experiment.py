#!/usr/bin/env python3
"""Compare Qwen-final, Qwen-candidate/Luna-review, and Luna-direct.

The reviewer receives only production-available evidence: visible source,
intent, Qwen candidate package, and provenance. Hidden qualification fields are
never rendered to either model. The independent evaluator is called only after
model generation and is the sole consumer of hidden case requirements.
"""
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

from rex_learning.candidate_skill import attach_review, build_candidate_skill  # noqa: E402
from rex_learning.lightweight_abstraction import normalize_integrated_answer  # noqa: E402
from run_live_instructional_abstraction_qualification_v5 import evaluate  # noqa: E402
from run_live_instructional_abstraction_qualification_v6 import staged_integrated_prompt  # noqa: E402

V6_ROOT = ROOT / "docs/rex-learning-deep-study-evidence/goal-relative-instructional-abstraction-qualification-v5/run-v6-20260825T234414Z-staged"
DEFAULT_CASES = ("v5-01", "v5-04", "v5-12", "v5-18", "v5-19", "v5-23")
REVIEW_MODEL = "gpt-5.6-luna"


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def extract_json(raw: str) -> dict[str, Any]:
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        raise RuntimeError("reviewer did not return JSON")
    value = json.loads(raw[start:end + 1])
    if not isinstance(value, dict):
        raise RuntimeError("model JSON was not an object")
    return value


def call_luna(prompt: str, source: str) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.time()
    result = subprocess.run(
        ["hermes", "chat", "-Q", "-q", prompt, "-m", REVIEW_MODEL, "--provider", "openai-codex", "--ignore-rules", "--ignore-user-config", "--max-turns", "1", "--source", source],
        cwd=ROOT, text=True, capture_output=True, timeout=1200, check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr[-2000:] or result.stdout[-2000:])
    raw = result.stdout.strip()
    return extract_json(raw), {"provider": "openai-codex", "model": REVIEW_MODEL, "source": source, "prompt_sha256": digest(prompt), "elapsed_seconds": time.time() - started, "raw_response": raw}


def reviewer_prompt(case: dict[str, Any], package: dict[str, Any]) -> str:
    return f"""You are a promotion reviewer in a learning system. Review one untrusted candidate skill package. You may accept, narrow, request source verification, request revision and retesting, defer, or reject it. Do not assume any claim is true merely because the candidate states it. Check every important claim against the visible source and curriculum intent. Do not invent hidden benchmark requirements or facts absent from the supplied material.

Return JSON only with this exact shape:
{{"candidate_id":"{package['candidate_id']}","decision":"PROMOTE|PROMOTE_WITH_NARROWER_SCOPE|REVISE_AND_RETEST|NEEDS_SOURCE_VERIFICATION|REJECT_UNSUPPORTED|DUPLICATE_EXISTING_SKILL|DEFER","rationale":"...","accepted_claims":[...],"rejected_claims":[...],"required_changes":[...],"source_refs":[...],"scope_limits":[...],"reviewed_surface":{{...}}}}

A PROMOTE decision means the proposed content is source-supported and appropriately scoped, not that behavioral competence has already been demonstrated. Do not claim durable promotion; the system separately requires independent behavioral evidence. The reviewed_surface must be a complete replacement final surface, preserving correct candidate content while repairing or removing unsupported claims. If the candidate cannot be safely repaired from the supplied evidence, choose REVISE_AND_RETEST, NEEDS_SOURCE_VERIFICATION, or REJECT_UNSUPPORTED instead of promoting it.

VISIBLE CASE:
{json.dumps({'source': case['source'], 'intent': case['intent']}, ensure_ascii=False, sort_keys=True)}

UNTRUSTED CANDIDATE PACKAGE:
{json.dumps(package, ensure_ascii=False, sort_keys=True)}
"""


def direct_prompt(case: dict[str, Any]) -> str:
    return staged_integrated_prompt(case) + "\nYou are the stronger direct-abstraction model. Produce the requested final surface without seeing any evaluator reference."


def load_cases(ids: list[str]) -> list[dict[str, Any]]:
    result = []
    for case_id in ids:
        path = V6_ROOT / f"{case_id}.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        result.append(record["case"])
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append", dest="case_ids", default=[])
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--skip-evaluation", action="store_true")
    args = parser.parse_args()
    case_ids: list[str] = list(args.case_ids) if args.case_ids else list(DEFAULT_CASES)
    run_root = Path(args.run_root)
    run_root.mkdir(parents=True, exist_ok=False)
    records = []
    failures = []
    for case in load_cases(case_ids):
        case_id = case["id"]
        try:
            v6_record = json.loads((V6_ROOT / f"{case_id}.json").read_text(encoding="utf-8"))
            qwen_surface = {"surface_schema": "rex-learning-integrated-final-surface-v3", **v6_record["canonical"]}
            package = build_candidate_skill(case=case, qwen_surface=qwen_surface, provenance={"provider": "qwen-local", "model": v6_record["provider"]["model"], "session_id": v6_record["provider"]["session_id"], "response_digest": v6_record["hashes"]["raw_provider_response_sha256"], "role": "candidate_generator", "source_run": str(V6_ROOT)})
            review_raw, review_meta = call_luna(reviewer_prompt(case, package), "rex-learning-candidate-review-v1")
            reviewed = attach_review(package, review_raw, reviewer={"provider": "openai-codex", "model": REVIEW_MODEL, "session_id": review_meta["prompt_sha256"], "role": "promotion_reviewer"})
            direct_raw, direct_meta = call_luna(direct_prompt(case), "rex-learning-direct-abstraction-control-v1")
            direct_surface = normalize_integrated_answer(direct_raw["responses"][0]["answer"] if len(direct_raw.get("responses", [])) == 1 else direct_raw)
            arms = {
                "QWEN_ALONE": {"surface": qwen_surface, "source_artifact": str(V6_ROOT / f"{case_id}.json"), "evaluation": v6_record.get("provider_semantic")},
                "QWEN_CANDIDATE_LUNA_REVIEW": {"package": reviewed, "review": review_meta, "surface": reviewed["qwen_interpretation"]},
                "LUNA_DIRECT": {"surface": direct_surface, "generation": direct_meta},
            }
            if not args.skip_evaluation:
                arms["QWEN_CANDIDATE_LUNA_REVIEW"]["evaluation"] = evaluate(case, "candidate-luna-review", reviewed.get("reviewed_surface", reviewed["qwen_interpretation"]))
                arms["LUNA_DIRECT"]["evaluation"] = evaluate(case, "luna-direct", direct_surface)
            record = {"schema": "rex-learning-candidate-review-experiment-v1", "case": case, "arms": arms, "hidden_reference_visible_to_models": False, "execution": {"status": "completed"}}
            (run_root / f"{case_id}.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            records.append(record)
            print(f"completed {case_id}", flush=True)
        except Exception as exc:
            failure = {"case": case, "execution": {"status": "failed", "error": repr(exc)}, "hidden_reference_visible_to_models": False}
            (run_root / f"{case_id}.failed.json").write_text(json.dumps(failure, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            failures.append(failure)
            print(f"FAILED {case_id}: {exc}", file=sys.stderr, flush=True)
    summary = {"schema": "rex-learning-candidate-review-experiment-v1", "cases_requested": len(case_ids), "cases_completed": len(records), "execution_failures": len(failures), "cases": case_ids, "arms": ["QWEN_ALONE", "QWEN_CANDIDATE_LUNA_REVIEW", "LUNA_DIRECT"], "reviewer": {"provider": "openai-codex", "model": REVIEW_MODEL}, "hidden_reference_visible_to_models": False}
    (run_root / "run-manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
