#!/usr/bin/env python3
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/rex-learning-deep-study-evidence/goal-relative-instructional-abstraction-qualification-v3"

def main() -> int:
    manifests = sorted(OUT.glob("*/run-manifest.json"), key=lambda p: p.stat().st_mtime)
    if not manifests:
        raise SystemExit("no V3 run manifest")
    manifest_path = manifests[-1]
    manifest = json.loads(manifest_path.read_text())
    run_root = manifest_path.parent
    records = []
    for path in sorted(run_root.glob("v3-*.json")):
        if path.name == "run-manifest.json" or path.name.endswith(".failed.json"):
            continue
        records.append(json.loads(path.read_text()))
    judgments = [r.get("evaluation", {}).get("judgment", {}) for r in records if r.get("evaluation")]
    control_judgments = [r.get("control_evaluation", {}).get("judgment", {}) for r in records if r.get("control_evaluation")]
    source_run = manifest.get("source_run")
    if source_run and not control_judgments:
        source_path = Path(source_run)
        control_records = [json.loads(p.read_text()) for p in source_path.glob("v3-*.json") if not p.name.endswith(".failed.json")]
        control_judgments = [r.get("control_evaluation", {}).get("judgment", {}) for r in control_records if r.get("control_evaluation")]
    def passing(j):
        value = j.get("pass", j.get("overall", j.get("passed", False)))
        return value is True or (isinstance(value, str) and value.lower() in {"pass", "passed", "true"})
    passed = sum(passing(j) for j in judgments)
    control_passed = sum(passing(j) for j in control_judgments)
    failures = list(run_root.glob("v3-*.failed.json"))
    interface = not failures and len(records) == manifest.get("cases_requested") and all(r.get("staged", {}).get("provenance", {}).get("stage_count") == 8 for r in records)
    groups = {}
    for r in records:
        group = r["case"].get("counterfactual_group")
        if group:
            groups.setdefault(group, []).append(r)
    counterfactual = []
    for group, pair in groups.items():
        if len(pair) >= 2:
            a, b = pair[:2]
            af = a["staged"]["final"]
            bf = b["staged"]["final"]
            counterfactual.append({"group": group, "invariant_equal": af.get("source_invariant") == bf.get("source_invariant"), "application_changed": af.get("goal_conditioned_interpretation") != bf.get("goal_conditioned_interpretation"), "action_changed": af.get("learning_action") != bf.get("learning_action")})
    counterfactual_pass = sum(x["invariant_equal"] and (x["application_changed"] or x["action_changed"]) for x in counterfactual)
    report = {
        "schema": "rex-learning-goal-relative-instructional-abstraction-qualification-v3-report",
        "run_manifest": str(manifest_path),
        "protocol_sha256": manifest.get("protocol_sha256"),
        "cases_requested": manifest.get("cases_requested"),
        "cases_completed": len(records),
        "execution_failures": len(failures),
        "interface_gate": {"status": "PASS" if interface else "FAIL", "stage_count_required": 8},
        "staged_semantic_evaluation": {"evaluated": len(judgments), "passed": passed, "pass_rate": passed / len(judgments) if judgments else None},
        "one_shot_control_evaluation": {"evaluated": len(control_judgments), "passed": control_passed, "pass_rate": control_passed / len(control_judgments) if control_judgments else None},
        "counterfactual": {"groups": counterfactual, "passed": counterfactual_pass, "total": len(counterfactual)},
        "decision": "ABSTRACTION_ARCHITECTURE_NOT_VALIDATED",
        "decision_reason": "V3 is not promotion-qualified unless interface, semantic, unseen, and counterfactual gates all satisfy the frozen protocol. This report does not authorize learning promotion or Java rerouting.",
        "record_files": [str(p) for p in sorted(run_root.glob("v3-*.json"))],
    }
    (run_root / "aggregate-report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    md = f"""# V3 staged instructional abstraction qualification\n\nProtocol SHA-256: `{report['protocol_sha256']}`\nRun: `{run_root}`\n\n## Evidence\n\n- Requested: {report['cases_requested']}\n- Completed: {report['cases_completed']}\n- Execution failures: {report['execution_failures']}\n- Interface gate: **{report['interface_gate']['status']}**\n- Staged semantic evaluations: {len(judgments)}; passes: {passed}\n- One-shot control evaluations: {len(control_judgments)}; passes: {control_passed}\n- Counterfactual groups: {len(counterfactual)}; passes: {counterfactual_pass}\n\n## Boundary\n\nRaw provider responses, parsed stage answers, ledger entries, hashes, and provenance remain in each case artifact. The evaluator received the frozen case requirements and candidate output; the candidate did not receive hidden references or evaluator output.\n\n## Decision\n\n# {report['decision']}\n\n{report['decision_reason']}\n"""
    (OUT / "LIVE_GOAL_RELATIVE_INSTRUCTIONAL_ABSTRACTION_QUALIFICATION_V3.md").write_text(md)
    print(json.dumps(report, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
