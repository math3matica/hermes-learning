#!/usr/bin/env python3
"""Replay preserved V3 provider answers through the current pipeline after a code fix."""
from __future__ import annotations
import json, sys, time, hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from rex_learning import CurriculumIntent, SourceContext
from rex_learning.staged_abstraction import StagedAbstractionPipeline, digest
from run_live_instructional_abstraction_qualification_v3 import evaluate, PROTOCOL, OUT, as_intent, as_source

OLD=Path(sys.argv[1]) if len(sys.argv)>1 else None
if OLD is None: raise SystemExit('usage: replay... OLD_RUN_DIR')
run=OUT/f"replay-{time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())}-{time.time_ns()%1000000:06d}"; run.mkdir()
protocol=json.loads(PROTOCOL.read_text()); old_records={json.loads(p.read_text())['case']['id']:json.loads(p.read_text()) for p in OLD.glob('v3-*.json') if not p.name.endswith('.failed.json')}
for case in protocol['cases']:
    old=old_records[case['id']]; answers=list(old['staged']['provider_calls'])
    def provider(operation, inputs, answers=answers):
        item=answers.pop(0)
        if item['operation'] != operation: raise RuntimeError(f"replay operation mismatch {item['operation']} != {operation}")
        return item['answer']
    staged=StagedAbstractionPipeline().run(as_intent(case),as_source(case),provider=provider,session_id='replay-'+case['id'])
    record={'case':case,'mode':'staged_replay','staged':{'ledger':staged.ledger.to_dict(),'stages':staged.stages,'final':staged.final,'provenance':staged.provenance,'provider_calls':old['staged']['provider_calls']},'execution':{'status':'completed','source_run':str(OLD),'provider_calls_replayed':True,'hidden_reference_rendered_to_candidate':False}}
    record['evaluation']=evaluate(case,record['staged'],'staged-replay')
    (run/f"{case['id']}.json").write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n')
manifest={'schema':'rex-learning-goal-relative-instructional-abstraction-replay-v3','protocol_sha256':hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),'cases_requested':len(protocol['cases']),'cases_completed':len(protocol['cases']),'failures':0,'status':'completed','run_root':str(run),'source_run':str(OLD),'provider_answers_replayed':True}
(run/'run-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n'); print(json.dumps(manifest,indent=2))
