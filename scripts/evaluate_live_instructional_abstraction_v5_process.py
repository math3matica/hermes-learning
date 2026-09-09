#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, subprocess, sys
from pathlib import Path


def digest(v): return hashlib.sha256(json.dumps(v, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()

def main():
    root=Path(sys.argv[1]); out=root/'process-evaluation-v2'; out.mkdir(exist_ok=False)
    rows=[]; failures=[]
    for path in sorted(root.glob('v5-*.json')):
        if path.name.endswith('.failed.json'): continue
        record=json.loads(path.read_text()); c=record['arms']['C']
        prompt=f'''Independently evaluate the selective-critic PROCESS, not just the final product. Compare the integrated answer before critique, the structural risk flags, the targeted critic output, and the final answer against the hidden required/prohibited reference. Return JSON only: {{"classification":"fixed_actual_problem|no_meaningful_change|made_worse|introduced_new_unsupported_claim|removed_correct_claim", "actual_risk":true, "risk_justified":true, "reason":"..."}}. A risk can be unjustified even if the answer contains uncertainty; judge whether it identifies a real problem in this case. Do not invent facts.
CASE:\n{json.dumps(record['case'],ensure_ascii=False,sort_keys=True)}\nBEFORE:\n{json.dumps(c['initial'],ensure_ascii=False,sort_keys=True)}\nRISKS:\n{json.dumps(c.get('risks',[]),ensure_ascii=False,sort_keys=True)}\nCRITIC:\n{json.dumps(c.get('critic'),ensure_ascii=False,sort_keys=True)}\nFINAL:\n{json.dumps(c['final_surface'],ensure_ascii=False,sort_keys=True)}'''
        try:
            result=subprocess.run(['hermes','chat','-Q','-q',prompt,'-m','gpt-5.6-luna','--provider','openai-codex','--ignore-rules','--ignore-user-config','--max-turns','1','--source','goal-relative-abstraction-v5-process-evaluator'],capture_output=True,text=True,timeout=1200,check=False)
            if result.returncode: raise RuntimeError(result.stderr[-2000:])
            raw=result.stdout.strip(); s,e=raw.find('{'),raw.rfind('}')
            if s<0 or e<=s: raise RuntimeError('no JSON')
            judgment=json.loads(raw[s:e+1])
            row={'case_id':record['case']['id'],'input_hash':digest(prompt),'provider':'openai-codex','model':'gpt-5.6-luna','judgment':judgment}
            (out/f"{record['case']['id']}.json").write_text(json.dumps(row,indent=2)+'\n'); rows.append(row)
        except Exception as exc:
            failures.append({'case_id':record['case']['id'],'error':repr(exc)})
    manifest={'schema':'rex-learning-integrated-critic-process-evaluation-v1','cases_evaluated':len(rows),'failures':failures,'rows':[r['case_id'] for r in rows]}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2)); return 0 if not failures else 2
if __name__=='__main__': raise SystemExit(main())
