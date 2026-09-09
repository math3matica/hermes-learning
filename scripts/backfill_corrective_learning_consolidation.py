#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from rex_learning.corrective_learning import canonicalize_learner_answer
from rex_learning.engine import LearningEngine
from rex_learning.store import LearningStore

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/rex-learning-deep-study-evidence/corrective-learning-v2"
PROTOCOL = OUT / "protocol.json"


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def main() -> None:
    engine = LearningEngine(LearningStore(OUT / "learning"))
    curricula: dict[str, tuple[dict, dict]] = {}
    for target_path in sorted(OUT.glob("*.json")):
        if target_path.name in {"run-manifest.json", "protocol.json"}:
            continue
        record = json.loads(target_path.read_text())
        if record.get("schema") != "rex-learning-corrective-target-v1":
            continue
        target = record["target"]
        curriculum_name = target["curriculum"]
        if curriculum_name not in curricula:
            curriculum = engine.create_curriculum(f"Corrective v2: {curriculum_name}", "Learner-authored corrective consolidation")
            skill = engine.create_skill_hypothesis(curriculum["id"], {"key": f"corrective.{target['target']}", "kind": "procedure", "claim": f"Corrective procedure for {target['target']}", "applicability": [target["strategy"]]})
            curricula[curriculum_name] = (curriculum, skill)
        _, skill = curricula[curriculum_name]
        for cycle in record["cycles"]:
            if "learner_consolidation" in cycle:
                continue
            activity = cycle["remediation"]["learner_activity"]
            procedure = canonicalize_learner_answer(activity)
            revision = engine.consolidate_learner_revision(
                skill["id"],
                {"claim": f"Learner corrective procedure for {target['target']} cycle {cycle['cycle']}", "operational_procedure": procedure, "applicability": [target["strategy"]], "non_applicability": ["unrelated task families"], "boundaries": ["candidate only until independent semantic validation"], "uncertainty": "Not yet independently qualified."},
                learner_provenance={"provider": "qwen-local", "model": "/models/Qwen3.8-27B-UD-Q4_K_XL.gguf", "session_id": cycle["remediation"]["learner_activity_session"], "role": "learner", "response_digest": digest(procedure)},
                source_provenance={"source_refs": [str(PROTOCOL), f"{target_path}:targeted_material"], "source_hash": digest(target["material"])},
                failure_ref=f"{target['target']}:unseen-retest-{cycle['cycle']}",
                revision_type="corrective_learning_consolidation",
            )
            cycle["learner_consolidation"] = {"status": "candidate", "skill_id": skill["id"], "skill_version": revision["version"], "revision_id": revision["id"]}
        target_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    manifest = json.loads((OUT / "run-manifest.json").read_text())
    manifest["learner_consolidation"] = {"status": "candidate_only", "candidate_revisions": sum(1 for p in OUT.glob("*.json") if p.name not in {"run-manifest.json", "protocol.json"} for c in json.loads(p.read_text()).get("cycles", []) if "learner_consolidation" in c), "qualified_revisions": 0, "store": str(OUT / "learning")}
    (OUT / "run-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("LEARNER_CONSOLIDATION_BACKFILL_COMPLETE")


if __name__ == "__main__":
    main()
