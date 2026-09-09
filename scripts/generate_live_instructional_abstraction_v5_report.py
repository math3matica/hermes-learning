#!/usr/bin/env python3
from __future__ import annotations
import json, sys
from pathlib import Path


def passed(j):
    value = j.get("pass", j.get("overall", j.get("passed", False)))
    return value is True or (isinstance(value, str) and value.lower() in {"pass", "passed", "true"})


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if root is None:
        raise SystemExit("usage: generate_live_instructional_abstraction_v5_report.py RUN_ROOT")
    manifest = json.loads((root / "run-manifest.json").read_text())
    records = [json.loads(path.read_text()) for path in sorted(root.glob("v5-*.json")) if not path.name.endswith(".failed.json")]
    arms = {arm: [r["arms"][arm] for r in records if arm in r.get("arms", {})] for arm in ("A", "B", "C")}
    summary = {}
    for arm, values in arms.items():
        judgments = [v.get("evaluation", {}).get("judgment", {}) for v in values if v.get("evaluation")]
        summary[arm] = {
            "evaluated": len(judgments),
            "passed": sum(passed(j) for j in judgments),
            "pass_rate": sum(passed(j) for j in judgments) / len(judgments) if judgments else None,
            "unsupported_claims": sum(int(j.get("unsupported_claim_count", 0) or 0) for j in judgments),
            "unsupported_broad_generalizations": sum(int(j.get("unsupported_broad_generalization_count", 0) or 0) for j in judgments),
            "critical_errors": sum(len(j.get("critical_errors", []) or []) for j in judgments),
            "provider_calls": sum(int(v.get("provider_calls", 0)) for v in values),
        }
    def pass_map(arm):
        return {r["case"]["id"]: passed(r["arms"][arm].get("evaluation", {}).get("judgment", {})) for r in records if arm in r.get("arms", {}) and r["arms"][arm].get("evaluation")}
    maps = {arm: pass_map(arm) for arm in ("A", "B", "C")}
    paired = {}
    for left, right in (("A", "B"), ("A", "C"), ("B", "C")):
        ids = sorted(set(maps[left]) & set(maps[right]))
        paired[f"{left}<->{right}"] = {"cases": len(ids), "left_only_pass": sum(maps[left][i] and not maps[right][i] for i in ids), "right_only_pass": sum(maps[right][i] and not maps[left][i] for i in ids), "both_pass": sum(maps[left][i] and maps[right][i] for i in ids), "both_fail": sum(not maps[left][i] and not maps[right][i] for i in ids), "net_right_over_left": sum(maps[right][i] and not maps[left][i] for i in ids) - sum(maps[left][i] and not maps[right][i] for i in ids)}
    risk = {"low": [], "high": []}
    for r in records:
        risk[r["case"].get("risk_class", "high")].append(r)
    risk_summary = {}
    for group, values in risk.items():
        risk_summary[group] = {arm: {"cases": len(values), "passed": sum(passed(r["arms"][arm].get("evaluation", {}).get("judgment", {})) for r in values if arm in r.get("arms", {})), "critic_invocations": sum(bool(r["arms"].get("C", {}).get("critic")) for r in values)} for arm in ("A", "B", "C")}
    c_values = arms["C"]
    invoked = [v for v in c_values if v.get("critic")]
    changes = [v for v in invoked if v.get("critic", {}).get("output")]
    report = {"schema":"rex-learning-goal-relative-instructional-abstraction-qualification-v5-report","run_manifest":str(root/"run-manifest.json"),"protocol_sha256":manifest.get("protocol_sha256"),"cases_requested":manifest.get("cases_requested"),"cases_completed":manifest.get("cases_completed"),"execution_failures":manifest.get("execution_failures"),"conditions":summary,"paired":paired,"risk_subsets":risk_summary,"risk_detector":{"critic_invocations":len(invoked),"critic_invocation_rate":len(invoked)/len(c_values) if c_values else None,"critic_changed_records":len(changes),"low_risk_false_trigger_rate":sum(bool(r["arms"].get("C", {}).get("critic")) for r in risk.get("low", []))/len(risk.get("low", [])) if risk.get("low") else None},"decision":"PENDING_INDEPENDENT_PROCESS_ANALYSIS"}
    (root/"qualification-report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    (root/"LIVE_LIGHTWEIGHT_INSTRUCTIONAL_ABSTRACTION_QUALIFICATION_V5.md").write_text("# Live Lightweight Instructional Abstraction Qualification V5\n\nThis report is generated from the immutable V5 run artifacts. Engineering verification and cognitive qualification are separate.\n\n```json\n"+json.dumps(report,indent=2,ensure_ascii=False)+"\n```\n")
    print(json.dumps(report, indent=2))
    return 0

if __name__ == "__main__": raise SystemExit(main())
