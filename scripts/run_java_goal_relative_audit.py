#!/usr/bin/env python3
"""Create an append-only goal-relative reinterpretation of the frozen Java study."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EVIDENCE = ROOT / "docs/rex-learning-deep-study-evidence"
SOURCE_HASH = "7b4514d64aac41b646ca64e1b6815c43ba689ed307382a412ab0a1eed7b482ed"
RUN_ROOT = EVIDENCE / "learn-java-the-easy-way" / "deep-study-v2"
OUT = EVIDENCE / "java-curriculum-v1" / "goal-relative-audit-v1"

from rex_learning import CurriculumIntent, DeepStudyEngine, LearningStore, SourceContext
from rex_learning.instructional_abstraction import assess_instructional_abstraction


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def java_intent() -> CurriculumIntent:
    return CurriculumIntent.from_dict({
        "key": "modern-java-foundations-v1",
        "title": "Modern Java foundations",
        "target_capability": "Develop a durable practical foundation in Java programming.",
        "competence_level": "working foundation with executable transfer",
        "intended_use": ["understand and construct Java programs", "debug and modify code", "validate input and handle failures", "test programs", "form disciplined programming habits"],
        "intended_transfer": ["unseen Java domains", "new program structures", "different applications and datasets"],
        "environment": {"os": "Linux", "jdk": "17", "compiler": "javac", "runtime": "java", "dependencies": "none"},
        "target_languages": ["Java"], "target_tools": ["javac", "java"],
        "temporal_requirements": "Prefer enduring concepts and current Java knowledge; reconcile operational procedures against current authoritative documentation.",
        "exclusions": ["reproducing a 2018 workstation", "treating source inclusion as mastery requirement"],
        "optional_specializations": ["Java 8 legacy maintenance", "Android application development"],
        "prerequisite_assumptions": [],
        "evaluation_criteria": ["compile", "execute", "handle edge cases", "test", "debug", "explain engineering choices", "transfer"],
        "desired_behavioral_change": ["decompose problems", "validate external input", "debug from evidence", "test edge conditions", "avoid cargo-cult abstraction"],
    })


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ingestion = json.loads((RUN_ROOT / "ingestion.json").read_text(encoding="utf-8"))
    observations = json.loads((RUN_ROOT / "observations.json").read_text(encoding="utf-8"))
    units = [json.loads(path.read_text(encoding="utf-8")) for path in (RUN_ROOT / "store" / "study_units").glob("*.json")]
    sections = {(item["source_path"], item["heading"]): item for item in ingestion["sections"]}
    intent = java_intent()
    store = LearningStore(OUT / "store")
    engine = DeepStudyEngine(store)
    intent_record = engine.create_curriculum_intent(intent.to_dict(), provenance={"source": "reconstructed from frozen Java protocol, source identity, and study report", "evidence_refs": [str(EVIDENCE / "java-curriculum-v1" / "baseline-protocol.json"), str(EVIDENCE / "java-curriculum-v1" / "source-identity.json"), str(EVIDENCE / "java-curriculum-v1" / "java-curriculum-report-v1.md")], "source_hash": SOURCE_HASH})
    distributions = {key: Counter() for key in ("relevance_strength", "optionality", "abstraction_level", "currentness", "source_quality", "learning_action", "instructional_value")}
    rows = []
    for unit in sorted(units, key=lambda item: item["key"]):
        ref = unit.get("source_refs", [""])[0]
        source_path, heading = (ref.split("#", 1) + [""])[:2]
        heading = heading.rsplit(":", 1)[0]
        section = sections.get((source_path, heading), {"source_path": source_path, "heading": heading, "text": ""})
        text = section.get("text", "")
        quality = "defective" if len(text.strip()) < 80 or text.count(".") > max(8, len(text) // 20) else "adequate"
        context = SourceContext(source_id="source-e6c6a0d321c0b1a1", unit_id=unit["id"], title=unit["title"], text=text, source_refs=unit.get("source_refs", []), hierarchy={"parent": unit.get("parent_key")}, source_date="2018", version_metadata={"publication_year": "2018"}, source_quality=quality)
        abstraction = assess_instructional_abstraction(intent, context)
        if abstraction.source_quality != quality:
            abstraction = type(abstraction)(**{**abstraction.to_dict(), "source_quality": quality})
        provenance = {"provider": "deterministic-scaffold", "model": "offline-contract-v1", "context_id": "java-goal-relative-audit-v1", "operation_id": f"offline-{unit['id']}", "input_hash": digest({"intent": intent.to_dict(), "source": context.to_dict()}), "output_hash": digest(abstraction.to_dict()), "timestamp": time.time(), "evidence_refs": unit.get("source_refs", [])}
        record = engine.record_instructional_abstraction(intent_id=intent_record["id"], source_id=context.source_id, unit_id=unit["id"], source_hash=unit["source_hash"], abstraction=abstraction, provenance=provenance)
        for key in distributions:
            distributions[key][abstraction.to_dict()[key]] += 1
        old_state = unit["state"]
        old_gate = "structural_pass" if old_state == "integrated" else "structural_fail"
        reinterpretation = "preserved-as-history"
        if abstraction.instructional_value in {"context_only", "reference_sufficient", "skip_for_this_objective"}:
            reinterpretation = "close-without-mastery"
        elif abstraction.learning_action in {"practice", "execute"}:
            reinterpretation = "route-to-executable-practice"
        elif abstraction.learning_action == "repair-source":
            reinterpretation = "repair-extraction-before-judging-learner"
        elif abstraction.learning_action == "current-doc-reconciliation":
            reinterpretation = "reconcile-current-docs-before-operational-use"
        rows.append({"unit_id": unit["id"], "chapter": unit.get("parent_key"), "title": unit["title"], "source_refs": unit.get("source_refs", []), "old_state": old_state, "old_gate": old_gate, "abstraction_record_id": record["id"], "abstraction": abstraction.to_dict(), "reinterpretation": reinterpretation})
    summary = {key: dict(value) for key, value in distributions.items()}
    report = {
        "schema": "rex-learning-java-goal-relative-audit-v1", "status": "completed_offline_contract_audit", "source_hash": SOURCE_HASH, "curriculum_intent": intent_record,
        "historical_state": {"total_units": len(units), "integrated": sum(x["state"] == "integrated" for x in units), "needs_reread": sum(x["state"] == "needs_reread" for x in units), "history_mutated": False},
        "distributions": summary, "unit_rows": rows,
        "limitations": ["This pass uses the qualified deterministic contract scaffold, not a provider cognition run.", "It creates educational judgments and routing candidates; it does not promote capability evidence.", "Source-quality heuristics identify repair candidates and require human/provider inspection before operational conclusions.", "No held-out post-study task or answer was exposed."],
    }
    (OUT / "curriculum-intent.json").write_text(json.dumps(intent_record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT / "audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# Java Goal-Relative Reinterpretation Report", "", "Status: completed offline contract audit; capability not promoted.", "", f"- Units audited: {len(rows)}", f"- Historical integrated: {report['historical_state']['integrated']}", f"- Historical needs_reread: {report['historical_state']['needs_reread']}", "- Historical records mutated: no", "- Held-out tasks exposed: no", "", "## Distributions"]
    for key, value in summary.items():
        lines.append(f"- {key}: " + ", ".join(f"{k}={v}" for k, v in sorted(value.items())))
    lines += ["", "## Decision boundary", "The old 74/61 split remains a record of structural retrieval outcomes. The new rows explain what each unit should do for this objective; they do not reinterpret integrated as capability or needs_reread as ignorance.", "", "## Examples"]
    wanted = ["About the Author", "Installing Java 8 and 9", "Programming Challenges", "DEBUGGING", "Android", "Preferences"]
    selected = []
    for token in wanted:
        selected.extend(row for row in rows if token.casefold() in row["title"].casefold() and row not in selected)
    selected.extend(row for row in rows if row["abstraction"]["source_quality"] == "defective" and row not in selected)
    for row in selected[:18]:
            a = row["abstraction"]
            lines += [f"### {row['title']}", f"- old state: {row['old_state']}", f"- relevance/action: {a['relevance_strength']} / {a['learning_action']}", f"- retained abstraction: {a['retained_abstraction']}", f"- contingent details: {a['contingent_details']}", f"- currentness/quality: {a['currentness']} / {a['source_quality']}", f"- required evidence: {a['required_evidence']}", ""]
    (OUT / "java-reinterpretation-report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "intent_id": intent_record["id"], "units": len(rows), "distributions": summary, "out": str(OUT)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
