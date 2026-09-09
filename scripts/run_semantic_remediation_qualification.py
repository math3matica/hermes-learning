#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
from rex_learning import OpenAICompatibleLearner
from run_semantic_feedback_qualification import evaluate_case, QWEN_ENDPOINT, QWEN_MODEL, LUNA_MODEL, evaluator_prompt
from rex_learning.semantic_evaluator import validate_judgment, build_feedback

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/rex-learning-deep-study-evidence/semantic-qualification-v1/remediation'
CASES=[
 {"curriculum":"make-it-stick","failed":"mist-01","material":"Durable learning requires source-free retrieval followed by feedback. Spacing retrieval across separated sessions improves retention. Rereading can provide an initial model or target a failed proposition, but fluency during rereading is not mastery evidence.","task":"A learner can state a rule immediately after rereading but forgets it a week later. Design a two-week plan using retrieval, feedback, and spacing, and state the exception where brief initial study is appropriate.","reference":"Use source-free retrieval with feedback across separated sessions; use targeted rereading only for failed propositions. Brief initial study is appropriate before a coherent model exists."},
 {"curriculum":"how-to-read-a-book","failed":"hrb-01","material":"Analytical reading identifies the author's problem, important terms, central proposition, supporting reasons, and qualifications before judging truth. Understanding and criticism are separate activities.","task":"Unseen passage: 'A pilot neighborhood reduced car traffic after adding a toll, but the sample was voluntary and the author warns that other neighborhoods may respond differently.' State the problem, central proposition, support, qualification, and one criticism, clearly separating representation from judgment.","reference":"Problem: effect of a toll on car traffic. Proposition: the pilot reduced traffic. Support: the observed pilot result. Qualification: voluntary sample and limited generalization. Criticism: selection and external-validity limits. Understanding is separate from criticism."},
]

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 for case in CASES:
  learner=OpenAICompatibleLearner(endpoint=QWEN_ENDPOINT,model=QWEN_MODEL,provider='qwen-local',session_id='qwen-remediation-'+case['curriculum'],timeout=600,max_tokens=1200)
  exposure=learner.answer(task='Read this remediation material and explain its actionable rule in your own words.\n'+case['material'])
  retest=learner.answer(task='Answer this source-free unseen retest. Do not claim access to hidden references. '+case['task'])
  learner_text=json.dumps(retest.get('responses',[{}])[0].get('answer',''),ensure_ascii=False)
  synthetic={"id":case['failed']+'-retest',"reference":case['reference'],"required":[]}
  session,raw,j=evaluate_case(synthetic,learner_text)
  rec={"schema":"rex-learning-remediation-retest-v1","curriculum":case['curriculum'],"failed_case":case['failed'],"remediation_material":case['material'],"source_exposure":{"provider":"qwen-local","session_id":learner.session_id,"output":exposure},"retest":{"task":case['task'],"source_visibility":"source_free","provider":"qwen-local","session_id":learner.session_id,"output":learner_text},"evaluator":{"provider":"openai-codex","model":LUNA_MODEL,"session_id":session,"output":raw},"judgment":j,"feedback":build_feedback(j),"contamination":{"status":"clean","evaluator_output_returned_to_learner":False},"wall_clock_delay_seconds":0}
  (OUT/(case['curriculum']+'.json')).write_text(json.dumps(rec,indent=2,ensure_ascii=False)+'\n')
  print(case['curriculum'],j['label'],j.get('diagnoses',[]),flush=True)
if __name__=='__main__': main()
