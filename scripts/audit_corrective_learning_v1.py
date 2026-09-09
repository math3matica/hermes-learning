#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from rex_learning.corrective_learning import canonicalize_learner_answer
from rex_learning.semantic_evaluator import validate_judgment
from run_corrective_learning_qualification import TARGETS

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "docs/rex-learning-deep-study-evidence/corrective-learning-v1-rerun-1"
OUT = ROOT / "docs/rex-learning-deep-study-evidence/corrective-learning-v1-forensic-audit"
INVALID = OUT / "invalid-evaluator"
MODEL = "gpt-5.6-luna"


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def prompt_for(target: dict[str, Any], answer: str, stage: str) -> str:
    return f"""You are Luna, an independent semantic teacher/critic, not the learner. Re-evaluate a historical Qwen/Rex learner answer for a forensic audit of a corrective-learning experiment. This is evaluator-only work: do not treat the result as fresh learner capability evidence. The learner did not see this reference, rubric, your diagnosis, or your output before answering. Stage: {stage}.\n\nTASK:\n{target['retest']}\n\nCOMPLETE CANONICAL LEARNER ANSWER:\n{answer}\n\nSOURCE-GROUNDED REFERENCE (hidden from learner):\n{target['reference']}\n\nREQUIRED ELEMENTS:\n{json.dumps(target['required'])}\n\nReturn JSON only with exactly: label (one allowed semantic label), diagnoses (list containing only these exact taxonomy labels: correct_but_incomplete, misconception, essential_omission, valid_alternative_formulation, unsupported_embellishment, overgeneralization, underqualification, inappropriate_applicability, evaluator_cannot_determine, legitimate_not_applicable), dimensions (object with semantic_correctness, procedure_execution, applicability_judgment, uncertainty_calibration each pass/fail/uncertain/not_applicable), missing_propositions (list), unsupported_claims (list), feedback (specific but concise), confidence (0..1). Accept valid paraphrases and operationally equivalent meaning. Do not fail merely for different terminology. Do not use the historical judgment as evidence."""


def evaluate(target: dict[str, Any], answer: str, stage: str) -> tuple[dict[str, Any], dict[str, Any]]:
    prompt = prompt_for(target, answer, stage)
    result = subprocess.run(
        ["hermes", "chat", "-Q", "-q", prompt, "-m", MODEL, "--provider", "openai-codex", "--ignore-rules", "--ignore-user-config", "--max-turns", "1", "--source", "corrective-learning-forensic-audit"],
        cwd=ROOT, text=True, capture_output=True, timeout=900, check=False,
    )
    raw = result.stdout
    provenance = {
        "provider": "openai-codex", "model": MODEL, "session_id": "unknown",
        "request_hash": digest(prompt), "evaluator_prompt_hash": digest(prompt),
        "raw_output_hash": digest(raw), "invocation_id": digest(prompt + raw)[:32],
        "observed_at": time.time(),
    }
    if result.returncode:
        raise RuntimeError(result.stderr[-2000:])
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    text = lines[-1] if lines else ""
    start, end = text.find("{"), text.rfind("}")
    try:
        if start < 0 or end <= start:
            raise ValueError("no JSON object")
        judgment = json.loads(text[start:end + 1])
        validate_judgment(judgment)
    except (ValueError, json.JSONDecodeError) as exc:
        INVALID.mkdir(parents=True, exist_ok=True)
        (INVALID / f"{digest(stage + raw)[:24]}.json").write_text(json.dumps({"schema": "rex-learning-invalid-evaluator-output-v1", "stage": stage, "error": str(exc), "stdout": raw, "stderr": result.stderr, "learner_answer": answer, "provenance": provenance}, indent=2) + "\n")
        retry = subprocess.run(
            ["hermes", "chat", "-Q", "-q", prompt + "\nPrevious output was schema-invalid. Emit only valid JSON using the exact allowed taxonomy strings.", "-m", MODEL, "--provider", "openai-codex", "--ignore-rules", "--ignore-user-config", "--max-turns", "1", "--source", "corrective-learning-forensic-audit-retry"],
            cwd=ROOT, text=True, capture_output=True, timeout=900, check=False,
        )
        raw = retry.stdout
        provenance.update({"retry_raw_output_hash": digest(raw), "retry_invocation_id": digest(prompt + raw)[:32]})
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        text = lines[-1] if lines else ""
        start, end = text.find("{"), text.rfind("}")
        if retry.returncode or start < 0 or end <= start:
            raise RuntimeError("forensic evaluator retry failed")
        judgment = json.loads(text[start:end + 1])
        validate_judgment(judgment)
    return judgment, provenance


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ledger = {
        "schema": "rex-learning-corrective-learning-invalidation-v1",
        "status": "historical_records_preserved_invalidated_for_evaluator_input_truncation",
        "invalidated_at": time.time(),
        "historical_root": str(OLD),
        "defect": "runner answer_text selected only responses[0].answer and omitted later learner response fields from Luna input",
        "affected_cases": [], "replacement_experiment": "pending_corrective_learning_v2",
        "scientific_boundary": "historical judgments are not fresh learner capability evidence",
    }
    for target in TARGETS:
        path = OLD / f"{target['curriculum']}-{target['target']}.json"
        record = json.loads(path.read_text())
        for cycle in record.get("cycles", []):
            audit_id = f"{target['target']}:cycle-{cycle['cycle']}"
            full = canonicalize_learner_answer(cycle["retest"]["learner"]["output"])
            old_judgment = cycle["retest"]["judgment"]
            corrected, provenance = evaluate(target, full, audit_id)
            artifact = {
                "schema": "rex-learning-corrective-forensic-audit-v1", "audit_id": audit_id,
                "historical_record": str(path), "historical_judgment": old_judgment,
                "historical_judgment_status": "invalidated_evaluator_input_truncation",
                "learner_output_provenance": {"session_id": cycle["retest"]["learner"]["session_id"], "response_count": len(cycle["retest"]["learner"]["output"].get("responses", [])), "canonical_answer": full, "canonical_answer_hash": digest(full)},
                "corrected_forensic_judgment": corrected, "evaluator_provenance": provenance,
                "fresh_capability_evidence": False,
            }
            (OUT / f"{target['curriculum']}-{target['target']}-cycle-{cycle['cycle']}.json").write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n")
            ledger["affected_cases"].append({"audit_id": audit_id, "historical_record": str(path), "replacement_audit": str(OUT / f"{target['curriculum']}-{target['target']}-cycle-{cycle['cycle']}.json")})
            print(f"[Forensic audit] {audit_id} old={old_judgment['label']} corrected={corrected['label']}", flush=True)
    (OUT / "invalidation-ledger.json").write_text(json.dumps(ledger, indent=2) + "\n")
    print("FORENSIC_AUDIT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
