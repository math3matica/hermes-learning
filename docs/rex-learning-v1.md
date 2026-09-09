# Rex Learning V1

## Architectural statement

Rex Learning converts instructional material into candidate knowledge and procedures, tests whether those procedures produce transferable competence, preserves evidence and provenance, and promotes only demonstrated improvements into Rex's persistent skill state.

This is deliberately not a RAG feature. Retrieval can support study or execution, but a source, note, summary, embedding, or model confidence never establishes a skill.

## Boundary and reuse

Rex remains the user-facing identity. Hermes owns durable execution and persistence. Gemma/Pi remains the realtime conversational runtime. Qwen/Hermes performs long-running study, practice, evaluation, revision, and evidence collection.

Learning state is an inspectable filesystem tree of atomic JSON records. It follows the Post-Call V1 pattern: write state before expensive work, use deterministic identities, preserve raw provenance, and treat provider artifacts as evidence only after validation. Rex Learning does not add a scheduler or worker state machine. A future Post-Call worker can create or resume a curriculum using the same durable root; normal long-running execution remains Post-Call's responsibility.

A future voice turn should capture a curriculum assignment. The next call receives only `LearningEngine.prepared_context()`, not books or study logs.

## Domain model

`LearningStore` has separate collections for curricula, sources, study notes, knowledge, skills, immutable skill versions, tests, attempts, evaluations, applications, and events.

- Curriculum: objective, sources, status, and method version.
- Source: locator, format, hash, and unverified source-claim trust boundary.
- Study note: structured claim/procedure/counterexample/limitation with source provenance.
- Knowledge: integrated claim with relation (`new`, `duplicate`, `conflict`, etc.) and preserved provenance.
- Skill hypothesis: typed declarative knowledge, heuristic, decision rule, procedure, or workflow with applicability, preconditions, criteria, limitations, and provenance.
- Test: frozen phase, cases, criteria, resource restrictions, evaluator type, and independence metadata.
- Attempt: learner output and scores. Attempt is not evaluation.
- Evaluation: evaluator result, contamination status, evidence references, and independence. Evaluation is not promotion.
- Application: selected skill version, procedure actually used, task, and outcome evidence.
- Skill version: prior formulation retained when revision or demonstration changes the current record.
- Event: concise lifecycle observability, not model chatter.

## Lifecycle and evidence gates

The implemented state set is: `candidate`, `practicing`, `demonstrated`, `challenged`, plus reserved states `studied`, `provisionally_demonstrated`, `robust`, `conflicted`, `superseded`, and `deprecated`.

Promotion requires all of the following:

1. A pretest attempt tied to a test whose phase is `pretest`.
2. A separate held-out `posttest` or `retest` test.
3. A passed evaluation at or above the frozen threshold.
4. Clean or explicitly not-applicable contamination status.
5. Evaluator independence stronger than shared-context/self-evaluation.
6. Evidence references and skill/test identity match.

A summary, candidate, self-report, or model claim cannot promote a skill. Contaminated and weakly independent evaluations remain useful historical evidence but cannot silently become strong evidence.

The natural lifecycle is:

`candidate -> practicing -> demonstrated -> challenged -> practicing` (after evidence-backed revision), with version snapshots preserved. A real-task failure is represented by `challenge_skill`; one failure does not erase the skill, and it is not ignored.

## Consolidation and conflicts

Knowledge is compared during insertion. Exact claims become `duplicate`/`integrated`; known incompatible fixture principles become `conflicted` and both records retain provenance. The general representation allows future semantic consolidation to classify refinements, alternatives, conditional differences, support, and supersession without throwing away source claims.

The implementation intentionally does not ask a model to average incompatible claims. It is better to preserve an unresolved conflict than to manufacture a mushy rule.

## Contamination and evaluator firewall

Study, practice, and held-out test records carry separate identifiers. Tests freeze criteria before attempts. Test records declare allowed/prohibited resources. Evaluations record whether the learner saw the problem, source solutions were available, retrieval could expose the answer, or the evaluator shared context; these contamination and independence fields are trusted only when produced by the registered evaluator. The deterministic fixture does not independently enforce learner isolation or prove that prohibited resources were inaccessible.

Evaluations are not trusted merely because a caller supplies `passed=true`. The evaluator adapter must mark the result `validation_status: verified`; the engine validates evaluator identity, frozen test identity, attempt identity, finite score, case-level aggregate, threshold pass/fail, evidence references, contamination, independence, and provider/session provenance. The production registry binds evaluator IDs to concrete `ExecutableGrader` trust-root instances; duck-typed impostors are rejected. `ExecutableGrader` is the objective provider-neutral implementation. Learner attempts contain responses only; executable-grader answer keys remain outside the attempt and the learner cannot submit evaluator metadata. Production promotion requires verified executable-grader evidence for both the baseline and held-out result. Unverified, contaminated, or weakly independent results remain historical evidence but cannot promote a skill.

## Behavioral integration

`discover_applicable_skills(task_context)` is the bounded reuse interface: it filters persisted demonstrated/robust skills by task requirements, returns only current versions, and supplies a bounded operational procedure. It does not inject the learning registry or acquisition prompt into a task. `apply_skill(...)` records the selection reason, selected version, bounded operational context, and task reference. Application attempts may carry the task identifier in their integrity-bound record, and `attach_application_evidence(...)` rejects a present task mismatch before upgrading an application to verified. Caller-supplied outcome metadata cannot do that.

Operational Hermes/Pi skill promotion is outside automatic V1 scope. A demonstrated procedure is only a candidate for operational promotion; a future promotion path must require sandbox tests, adversarial tests, regression tests, and human approval. Instructional content is data, never policy or executable authority.

## Recovery and idempotency

Records use deterministic IDs derived from semantic identities and are written with fsync plus atomic replace. Repeating curriculum/source/test/attempt/application creation converges on the same record. Applications with the same task and skill return the existing record. Files are independently readable after a crash. Post-Call owns process locks and retries; LearningStore owns durable learning facts and does not pretend a process claim is evidence.

## Document formats

V1 accepts a source locator and format metadata. Plain text/Markdown fixtures prove the learning lifecycle. `rex_learning.ingestion.ingest_epub` provides the first real instructional-format path: it hashes and persists source identity/provenance, extracts XHTML sections and bounded chunks, records structured study notes with source references, and emits candidate skill claims. It is deliberately not RAG-only. PDF ingestion remains open.

## Controlled bootstrap experiment

`rex_learning.experiment.run_bootstrap_experiment` and `scripts/run_rex_learning_experiment.py` run a small deterministic fixture:

- Procedure A baseline: 1/3 (`33.3%`) on novel boundary cases.
- Study notes include a positive rule and a negative applicability case; knowledge integration preserves a deliberate conflict.
- Practice exposes deployment/domain confusion.
- The procedure is revised with that failure evidence.
- Held-out retest with Procedure B: 3/3 (`100%`), a `+66.7` percentage-point delta.
- The skill is persisted at version 2 with prior version evidence.
- A later novel task retrieves and applies version 2 and records outcome evidence.

This is a deterministic framework qualification fixture, not proof that Qwen/Gemma themselves learned. It exercises the evidence chain, declared contamination gates, durable versioning, later application seam, and bounded context; it does not independently prove learner isolation or provider cognition. Provider-backed learning and physical voice qualification remain open acceptance gates.

`rex_learning.production_experiment.run_production_learning_experiment` and `scripts/run_rex_production_learning_experiment.py` run a deterministic production-mode evaluator slice. It is complemented by `scripts/run_rex_provider_learning_experiment.py`, which uses `OpenAICompatibleLearner` against an OpenAI-compatible Qwen endpoint:

- A real instructional Markdown source is persisted with a content hash and produces structured notes and knowledge claims.
- A pre-study baseline is evaluated by an independent executable grader: `0.0`.
- The first post-study attempt uses the learned procedure but fails an ambiguous held-out case: `0.5`.
- Failure evidence drives a persisted revision and the procedure is rerun on a separate held-out retest: `1.0`.
- The skill is promoted by a default (`allow_fixture=False`) engine at version 2 using verified executable-grader evidence.
- A later task outside the original test sequence selects version 2, uses its procedure, and records a caller-supplied application outcome marked `validation_status: "unverified"`; this is an application seam, not independent transfer evidence.

The provider-backed run produced fresh ad-hoc evidence in the current checkout: baseline `0.0`, post-study retest `1.0`, revision after a failed no-material attempt, promotion by the default production engine, and a later application outcome recorded as `1.0` but marked unverified. The learner's responses came from the configured Qwen endpoint; the answer key remained in the trusted local executable grader and was not sent to the learner. This is evidence for one bounded provider-backed classification task, not broad cognition, generalization, real-world outcome improvement, or physical voice behavior. It is not canonical-suite evidence because it requires a live model endpoint.

## Post-Call and voice workflows

A voice assignment should become a Post-Call work proposal with an explicit curriculum objective and source handles. Post-Call validates the assignment and owns the worker. The worker invokes the LearningEngine in a durable learning root, writes study/test/evaluation records, and reports only verified progress. The next voice session receives curriculum status, demonstrated skills, current weaknesses, and next action through bounded prepared context. It does not receive source text, full notes, or raw execution chatter.

Useful user actions are: create/add/remove/pause/resume curriculum; inspect progress/notes/candidates/demonstrated skills/evidence; challenge/retest; revise; retire/supersede; and separately approve or reject operational promotion.

## Known limitations

- The provider learner currently accepts structured JSON from an OpenAI-compatible endpoint; EPUB ingestion is bounded and PDF ingestion is not yet included.
- Knowledge relation detection is intentionally conservative and not a general semantic contradiction solver.
- The bootstrap experiment is deterministic and cannot establish model cognition or real-world outcome improvement; the provider experiment is one bounded, live-endpoint classification result and requires replication and stronger controls. Promotion now requires a posttest score strictly above the trusted pretest score.
- Resource budgets, worker orchestration, and physical learner-resource isolation remain Post-Call/integration responsibilities; LearningStore currently records state but does not run an autonomous loop.
- Prepared context is bounded by character trimming, not tokenization.
- The fixture qualification helper and deterministic fixture promotion require an explicit `allow_fixture=True` engine; production engines cannot promote caller-submitted fixture attempts. The first production path requires a registered trusted evaluator; arbitrary provider output and learner-supplied metadata cannot establish promotion. Revision evidence must resolve to an existing record owned by the skill or curriculum. Application outcomes are unverified unless a future trusted application evaluator is added.
- Physical qualification is still required before any claim that Rex Voice naturally captures and discusses learning assignments.

## Behavioral reuse qualification

`rex_learning.behavioral_experiment.run_behavioral_reuse_experiment` closes the acquisition engine and reopens the persisted learning root as a fresh later-task context. The experiment records baseline `0.0`, post-study `0.5`, held-out retest `1.0`, persisted demonstrated version `2`, bounded requirement matching, and verified application evidence. The later executor now receives observable task facts rather than a hidden semantic signal, and the no-skill control uses the same executor with an empty procedure. This supports a narrower deterministic procedure-influence fixture, not proof of independent skill selection, provider cognition, broad real-world generalization, or causal learning in a model.

## Acceptance disposition

The earlier stronger acquired-skill-caused-improvement claim was rejected by adversarial review because the original harness exposed a hidden semantic signal and used an asymmetric control. Those defects are repaired and regression-tested, but no `rex-learning-v1-baseline` should be created until the repaired experiment receives a fresh adversarial review and the remaining provider/physical qualification gates are satisfied.
