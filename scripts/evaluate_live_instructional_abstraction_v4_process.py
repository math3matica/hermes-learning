#!/usr/bin/env python3
"""Evaluate V4 staged process traces separately from final-product judgments."""
from __future__ import annotations
import json, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT / "docs/rex-learning-deep-study-evidence/goal-relative-instructional-abstraction-qualification-v4"

def digest(value):
 import hashlib
 return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,default=str).encode()).hexdigest()

def evaluate_process(record):
 case=record["case"]; staged=record["staged"]
 prompt=f"""Evaluate the staged instructional-abstraction PROCESS, not its final product. Inspect only the visible source, intent, and recorded stages. Compare against the hidden required/prohibited reference in the case. Return JSON only with:\n{{\"source_evidence_quality\":0,\"source_intent_separation\":0,\"claim_grounding\":0,\"proposition_survival\":0,\"critique_precision\":0,\"finalization_precision\":0,\"action_evidence_routing\":0,\"first_divergence_stage\":\"...\",\"controller_caused\":false,\"provider_caused\":false,\"evaluator_surface_issue\":false,\"notes\":\"...\"}}. Use 0,1,2 scores. Do not invent missing evidence.\nCASE:\n{json.dumps(case,ensure_ascii=False,sort_keys=True)}\nSTAGED PROCESS:\n{json.dumps(staged.get('stages',{}),ensure_ascii=False,sort_keys=True)}\nFINAL SURFACE:\n{json.dumps(staged.get('final_evaluation_surface',staged.get('final',{})),ensure_ascii=False,sort_keys=True)}"""
 r=subprocess.run(["hermes","chat","-Q","-q",prompt,"-m","gpt-5.6-luna","--provider","openai-codex","--ignore-rules","--ignore-user-config","--max-turns","1","--source","goal-relative-abstraction-v4-process-evaluator"],cwd=ROOT,text=True,capture_output=True,timeout=1200,check=False)
 if r.returncode: raise RuntimeError(r.stderr[-2000:])
 raw=r.stdout.strip(); a,b=raw.find("{"),raw.rfind("}")
 if a<0 or b<=a: raise RuntimeError("process evaluator did not return JSON")
 return {"provider":"openai-codex","model":"gpt-5.6-luna","input_hash":digest(prompt),"judgment":json.loads(raw[a:b+1])}

def main():
 import argparse
 ap=argparse.ArgumentParser(); ap.add_argument("run_root"); ap.add_argument("--out",default="")
 args=ap.parse_args(); root=Path(args.run_root); out=Path(args.out) if args.out else root/("process-evaluation-"+time.strftime("%Y%m%dT%H%M%SZ",time.gmtime())) ; out.mkdir(parents=True,exist_ok=False)
 results=[]; failures=[]
 for path in sorted(root.glob("v4-*.json")):
  try:
   rec=json.loads(path.read_text()); ev=evaluate_process(rec); wrapped={"case_id":rec["case"]["id"],"source_artifact":str(path),"candidate_hidden_reference_visible":False,"evaluation":ev}; (out/(path.stem+".json")).write_text(json.dumps(wrapped,indent=2,ensure_ascii=False)+"\n"); results.append(wrapped); print(path.stem,flush=True)
  except Exception as exc:
   failures.append({"case":path.name,"error":repr(exc)}); print("FAILED",path.name,exc,file=sys.stderr,flush=True)
 manifest={"schema":"rex-learning-staged-process-evaluation-v1","cases_requested":len(list(root.glob("v4-*.json"))),"cases_evaluated":len(results),"failures":len(failures),"source_run":str(root),"results":results,"failure_records":failures}
 (out/"manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n"); print(json.dumps({k:manifest[k] for k in ("cases_requested","cases_evaluated","failures","source_run")},indent=2)); return 0 if not failures else 2
if __name__=="__main__": raise SystemExit(main())
