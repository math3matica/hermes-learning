from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_live_instructional_abstraction_qualification_v2 import PROTOCOL, evaluate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--protocol", default=str(PROTOCOL))
    args = parser.parse_args()
    run = Path(args.run)
    protocol_path = Path(args.protocol)
    protocol = json.loads(protocol_path.read_text())
    cases = {case["id"]: case for case in protocol["recovery_set"] + protocol["holdout_cases"]}
    artifacts = sorted(run.glob("recovery-*.json")) + sorted(run.glob("holdout-*.json"))
    if args.limit:
        artifacts = artifacts[:args.limit]
    out = run / "independent-evaluations"
    out.mkdir(exist_ok=True)
    results = []
    failures = []
    for index, path in enumerate(artifacts, 1):
        record = json.loads(path.read_text())
        case = cases[record["case"]["id"]]
        print(f"evaluate {index}/{len(artifacts)} {case['id']}", flush=True)
        try:
            judgment = evaluate(case, record["candidate"])
            wrapped = {"case_id": case["id"], "case_set": record["case"]["set"], "source_artifact": str(path), "evaluation": judgment, "candidate_hidden_reference_visible": False, "deterministic_control_visible": False}
            (out / f"{record['case']['set']}-{case['id']}.json").write_text(json.dumps(wrapped, indent=2, ensure_ascii=False) + "\n")
            results.append(wrapped)
        except Exception as exc:
            failure = {"case_id": case["id"], "source_artifact": str(path), "error": repr(exc)}
            (out / f"{record['case']['set']}-{case['id']}.failed.json").write_text(json.dumps(failure, indent=2) + "\n")
            failures.append(failure)
            print(f"FAILED {case['id']}: {exc}", file=sys.stderr, flush=True)
    manifest = {"schema": "rex-learning-independent-evaluation-run-v2", "protocol": str(protocol_path), "protocol_sha256": __import__('hashlib').sha256(protocol_path.read_bytes()).hexdigest(), "source_run": str(run), "cases_requested": len(artifacts), "cases_evaluated": len(results), "failures": len(failures), "candidate_hidden_reference_visible": False, "deterministic_control_visible": False, "evaluator_role": "independent semantic evaluator"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
