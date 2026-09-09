#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rex_learning.semantic_evaluator import validate_judgment
from run_semantic_feedback_qualification import evaluate_case, LUNA_MODEL
from rex_learning.store import LearningStore

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "docs/rex-learning-deep-study-evidence"
OUT = BASE / "retrieval-semantic-audit-v1"


def choose(slug: str, title: str | None, state: str | None = None) -> dict[str, Any]:
    root = BASE / slug / "deep-study-v2"
    store = LearningStore(root / "store")
    units = [u for u in store.list("study_units") if state is None or u.get("state") == state]
    if title is not None:
        units = [u for u in units if u.get("title") == title]
    unit = units[0]
    retrievals = [r for r in store.list("study_retrievals") if r.get("unit_id") == unit["id"]]
    retrieval = retrievals[-1]
    exposures = [e for e in store.list("study_exposures") if e.get("unit_id") == unit["id"]]
    close = next(e for e in exposures if e.get("phase") == "close_reading")
    ingestion = json.loads((root / "ingestion.json").read_text())
    chunks = [c for c in ingestion["chunks"] if c["source_ref"] in unit["source_refs"]]
    payload = json.loads(Path(retrieval["evidence_ref"]).read_text())
    learner = payload["responses"][0].get("answer", {})
    close_payload = json.loads(Path(close["output_ref"]).read_text())
    close_answer = close_payload["responses"][0].get("answer", {})
    return {"slug": slug, "unit": unit, "retrieval": retrieval, "source": "\n".join(c.get("text", "") for c in chunks), "close_answer": close_answer, "learner": learner}


def main() -> None:
    selected = [
        choose("make-it-stick", "untitled", "needs_reread"),
        choose("make-it-stick", "Empirical Evidence versus Theory, Lore, and Intuition", "needs_reread"),
        choose("how-to-read-a-book", "On Philosophical Method", "needs_reread"),
        choose("how-to-read-a-book", "The Questions Philosophers Ask", "needs_reread"),
        choose("make-it-stick", None, "integrated"),
        choose("how-to-read-a-book", None, "integrated"),
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    for index, item in enumerate(selected, 1):
        task = f"Evaluate this source-free retrieval for the section {item['unit']['title']}. Determine whether the learner has retrieved the source-supported central meaning, conditions, and important boundaries. A refusal can be appropriate only when the task genuinely lacks enough source identity; here the source evidence is supplied to you."
        prompt = f"""You are Luna, independent semantic evaluator. Return JSON only with label, dimensions (semantic_correctness, procedure_execution, applicability_judgment, uncertainty_calibration), diagnoses, missing_propositions, unsupported_claims, feedback, confidence. Use the frozen v2 taxonomy. Do not reward fluent refusal as knowledge. Source evidence:\n{item['source'][:14000]}\nClose-reading evidence (context only):\n{json.dumps(item['close_answer'], ensure_ascii=False)[:10000]}\nLearner source-free retrieval:\n{json.dumps(item['learner'], ensure_ascii=False)}\nTask: {task}"""
        synthetic={"id": f"audit-{index}", "reference": item["source"], "required": []}
        session, raw, judgment = evaluate_case(synthetic, json.dumps(item["learner"], ensure_ascii=False))
        record = {"schema": "rex-learning-retrieval-semantic-audit-v1", "case_number": index, "curriculum": item["slug"], "unit": item["unit"], "structural_retrieval": item["retrieval"], "source_material": item["source"], "close_reading_output": item["close_answer"], "learner_retrieval_output": item["learner"], "evaluator": {"provider": "openai-codex", "model": LUNA_MODEL, "session_id": session, "input": prompt, "output": raw}, "judgment": judgment, "lifecycle_mutated": False}
        (OUT / f"case-{index:02d}.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
        print(f"[Retrieval audit] case={index}/6 curriculum={item['slug']} title={item['unit']['title']} state={item['unit']['state']} judgment={judgment.get('label')} diagnosis={','.join(judgment.get('diagnoses', [])) or 'none'}", flush=True)


if __name__ == "__main__":
    main()
