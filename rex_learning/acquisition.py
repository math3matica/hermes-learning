from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from typing import Any, cast

from .candidate_skill import attach_review, build_candidate_skill
from .engine import CeilingBaselineError, LearningEngine, LearningError
from .evaluator import TrustedEvaluator
from .learner import Learner
from .capability_discovery import bind_independent_competence_test_design, define_test_from_independent_design, execute_bound_independent_test


Reviewer = Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]]


class NonDiscriminatingControlError(LearningError):
    """The frozen matched-control task has no measurable acquisition headroom."""

    code = "non_discriminating_control"


def _persist_baseline_preflight(*, engine: LearningEngine, curriculum_id: str, source_id: str, source_hash: str, intent_key: str, target_capability: str, skill_id: str, test: Mapping[str, Any], attempt: Mapping[str, Any], evaluation: Mapping[str, Any]) -> dict[str, Any]:
    """Persist trusted source-free baseline evidence before treatment."""
    identity = json.dumps({"curriculum_id": curriculum_id, "source_id": source_id, "source_hash": source_hash, "intent_key": intent_key, "test_id": test["id"], "attempt_id": attempt["id"]}, sort_keys=True)
    record = {"schema": "rex-learning-baseline-preflight-v1", "id": "baseline-" + hashlib.sha256(identity.encode()).hexdigest()[:24], "curriculum_id": curriculum_id, "source_id": source_id, "source_hash": source_hash, "intent_key": intent_key, "target_capability": target_capability, "skill_id": skill_id, "test_id": test["id"], "attempt_id": attempt["id"], "evaluation_id": evaluation["id"], "phase": test.get("phase"), "source_free": test.get("source_free"), "contamination_status": evaluation.get("contamination", {}).get("status"), "evaluator_independence": evaluation.get("independence"), "score": evaluation.get("score"), "passed": evaluation.get("passed")}
    return engine.store.write("baseline_preflights", record["id"], record)


def _reuse_baseline_preflight(*, engine: LearningEngine, baseline_preflight_id: str, curriculum_id: str, source_id: str, source_hash: str, intent_key: str, target_capability: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load and validate one trusted baseline before resuming treatment."""
    try:
        record = engine.store.read("baseline_preflights", baseline_preflight_id)
    except KeyError as exc:
        raise LearningError("baseline preflight record is missing") from exc
    required = {
        "schema": "rex-learning-baseline-preflight-v1",
        "curriculum_id": curriculum_id,
        "source_id": source_id,
        "source_hash": source_hash,
        "intent_key": intent_key,
        "target_capability": target_capability,
        "source_free": True,
        "contamination_status": "clean",
        "evaluator_independence": "independent",
        "passed": False,
    }
    if any(record.get(key) != value for key, value in required.items()):
        raise LearningError("baseline preflight record does not match acquisition context")
    if not isinstance(record.get("score"), (int, float)) or float(record["score"]) < 0.0 or float(record["score"]) > 1.0:
        raise LearningError("baseline preflight score is malformed")
    try:
        skill = engine.store.read("skills", str(record["skill_id"]))
        test = engine.store.read("tests", str(record["test_id"]))
        attempt = engine.store.read("attempts", str(record["attempt_id"]))
        evaluation = engine.store.read("evaluations", str(record["evaluation_id"]))
    except KeyError as exc:
        raise LearningError("baseline preflight references missing evidence") from exc
    if (skill.get("id") != record["skill_id"] or skill.get("curriculum_id") != curriculum_id
            or test.get("id") != record["test_id"] or test.get("skill_id") != skill["id"]
            or test.get("phase") != "pretest" or test.get("source_free") is not True
            or attempt.get("id") != record["attempt_id"] or attempt.get("test_id") != test["id"]
            or evaluation.get("id") != record["evaluation_id"] or evaluation.get("attempt_id") != attempt["id"]
            or evaluation.get("score") != record["score"] or evaluation.get("passed") is not False):
        raise LearningError("baseline preflight evidence identity or score mismatch")
    return record, skill, attempt, evaluation


def _reviewer_decision(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise LearningError("reviewer output must be an object")
    return dict(value)


def _invoke_reviewer(reviewer: Mapping[str, Any] | Reviewer | Any, candidate: Mapping[str, Any], evidence: Mapping[str, Any], source_id: str, claim: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Invoke an injected reviewer while keeping review separate from promotion."""
    reviewer_obj: Any = reviewer
    if callable(reviewer_obj):
        decision = _reviewer_decision(reviewer_obj(candidate, evidence))
        provenance = {"provider": "configured-reviewer", "session_id": "review-callback", "invocation": "callable"}
    elif callable(getattr(reviewer_obj, "review", None)):
        decision = _reviewer_decision(reviewer_obj.review(candidate=candidate, evidence=evidence))
        raw_provenance = getattr(reviewer_obj, "provenance", None)
        provenance = dict(cast(Mapping[str, Any], raw_provenance)) if isinstance(raw_provenance, Mapping) else {}
        provenance.setdefault("invocation", "review_method")
    else:
        # A provenance mapping identifies a reviewer; it is not a reviewer
        # judgment.  Never turn that metadata into an implicit promotion.
        decision = {
            "candidate_id": candidate["candidate_id"],
            "decision": "DEFER",
            "rationale": "No reviewer judgment was supplied; promotion remains pending.",
            "accepted_claims": [],
            "rejected_claims": [],
            "required_changes": ["Obtain an explicit independent review with reviewed_surface."],
            "source_refs": [source_id],
            "scope_limits": [],
        }
        provenance = dict(cast(Mapping[str, Any], reviewer)) if isinstance(reviewer, Mapping) else {}
        provenance.setdefault("invocation", "configured_boundary")
    return decision, provenance


def _response(result: Mapping[str, Any], case: str) -> dict[str, Any]:
    responses = result.get("responses")
    if not isinstance(responses, list):
        raise LearningError("learner response must contain responses")
    matches = [item for item in responses if isinstance(item, Mapping) and item.get("case") == case]
    if len(matches) != 1:
        raise LearningError(f"learner must answer case exactly once: {case}")
    return {"case": case, "answer": matches[0].get("answer")}


def _test_spec(grader: TrustedEvaluator, phase: str, case: str, *, source_free: bool, novel_transfer: bool) -> dict[str, Any]:
    return {
        "phase": phase,
        "cases": [case],
        "evaluator_type": grader.evaluator_type,
        "evaluator_id": grader.evaluator_id,
        "success_threshold": 1.0,
        "evaluator_independence": "independent",
        "contamination_status": "clean",
        "source_free": source_free,
        "novel_transfer": novel_transfer,
    }


def _candidate_surface(result: Mapping[str, Any]) -> dict[str, Any]:
    """Unwrap a structured candidate returned as the learner's answer string."""
    responses = result.get("responses")
    if isinstance(responses, list) and len(responses) == 1 and isinstance(responses[0], Mapping):
        answer = responses[0].get("answer")
        if isinstance(answer, Mapping):
            return {**dict(result), **dict(answer), "_candidate_answer": dict(answer)}
        if isinstance(answer, str):
            try:
                parsed = json.loads(answer)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, Mapping):
                return {**dict(result), **dict(parsed), "_candidate_answer": answer}
            if responses[0].get("case") == "procedure" and answer.strip():
                return {**dict(result), "procedure": answer.strip(), "_candidate_answer": answer}
    return dict(result)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return list(value) if isinstance(value, list) else [value]


def _practice_revision_feedback(
    evaluation: Mapping[str, Any],
    remediation_number: int,
    *,
    skill_kind: str,
    target_capability: str,
) -> str:
    """Turn trusted practice diagnostics into bounded, type-aware treatment.

    Evaluator findings are diagnostic evidence, not an answer key.  The
    treatment planner adds retrieval requirements appropriate to the kind of
    capability so a retry must reconstruct and apply the missing behavior,
    rather than merely paraphrase the source or patch one example.
    """
    findings: list[dict[str, Any]] = []
    for result in evaluation.get("case_results", []):
        if not isinstance(result, Mapping):
            continue
        finding = {"case": result.get("case")}
        for key in ("failed_fields", "missing_propositions", "failed_assertions", "diagnosis", "rationale"):
            if key in result:
                finding[key] = result[key]
        findings.append(finding)
    feedback = json.dumps(findings, ensure_ascii=False, sort_keys=True)
    focus_by_kind = {
        "procedure": (
            "reconstruct the ordered steps, state changes, invariants, and observable checks; "
            "then apply them to a different concrete case"
        ),
        "workflow": (
            "reconstruct ordering, entry conditions, decision points, checkpoints, and failure handling; "
            "then apply the workflow to a different case"
        ),
        "declarative": (
            "reconstruct the claim, its distinguishing conditions, applicability boundary, and a counterexample; "
            "then explain and apply it to a different case"
        ),
    }
    focus = focus_by_kind.get(
        skill_kind,
        "reconstruct the claim, conditions, limitations, and observable verification; then apply it to a different case",
    )
    return (
        f"Practice failed on remediation {remediation_number} for {target_capability!r}. "
        f"Do not copy the prior artifact or invent an answer key. {focus}. "
        "Use the trusted diagnostic findings below to identify what was missing, "
        "state the corrected capability representation, and verify every required behavior "
        f"directly in the next response: {feedback}"
    )


def _procedure_text(value: Any) -> str:
    """Normalize provider workflow structures without losing their order.

    Providers commonly return a workflow as an ordered JSON object even when
    the acquisition contract asks for an actionable procedure.  Stringifying
    that object produces Python-repr noise and makes fresh execution useless.
    Preserve mapping insertion order and render nested values as JSON so the
    durable operational context remains readable and deterministic.
    """
    if isinstance(value, Mapping):
        return "\n".join(
            f"{key}: {_procedure_text(item)}"
            for key, item in value.items()
            if str(key).strip() and item is not None
        )
    if isinstance(value, list):
        return "\n".join(f"{index + 1}. {_procedure_text(item)}" for index, item in enumerate(value))
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value or "").strip()


def _merge_candidate_surface(candidate: dict[str, Any], surface: Mapping[str, Any]) -> dict[str, Any]:
    proposed = candidate["proposed_skill"]
    fields = {
        "claim": surface.get("transferable_abstractions") or surface.get("source_invariant") or surface.get("capability"),
        "scope": surface.get("applicability"),
        "operational_procedure": surface.get("operational_procedure") or surface.get("procedure"),
        "preconditions": surface.get("preconditions"),
        "limitations": surface.get("limitations") or surface.get("non_applicability"),
        "learning_action": surface.get("learning_action"),
        "evidence_path": surface.get("evidence_path"),
        "uncertainties": surface.get("uncertainties"),
    }
    for field, value in fields.items():
        if value is not None:
            proposed[field] = value
    candidate["qwen_revisions"] = [*candidate.get("qwen_revisions", []), dict(surface)]
    return candidate


def _sync_skill_from_candidate(skill: dict[str, Any], proposed: Mapping[str, Any], target_capability: str) -> None:
    claim = proposed.get("claim") or target_capability
    if isinstance(claim, list):
        claim = "; ".join(str(item) for item in claim)
    skill.update({
        "kind": str(proposed.get("kind") or skill.get("kind") or "procedure"),
        "claim": str(claim),
        "applicability": _as_list(proposed.get("scope") or [target_capability]),
        "operational_procedure": _procedure_text(proposed.get("operational_procedure")),
        "preconditions": _as_list(proposed.get("preconditions")),
        "limitations": _as_list(proposed.get("limitations")),
    })


def learn_from_source(
    *,
    engine: LearningEngine,
    learner: Learner,
    source_title: str,
    source_text: str,
    intent_key: str,
    target_capability: str,
    grader: TrustedEvaluator,
    reviewer: Mapping[str, Any] | Reviewer,
    fresh_learner: Learner | None = None,
    control_learner: Learner | None = None,
    baseline_task: str = "Perform the target capability on a concrete held-out task and return the result.",
    practice_task: str = "Apply the learned capability to the held-out practice task and return the result.",
    runtime_task: str = "Apply the learned capability to the held-out runtime task and return the result.",
    negative_task: str = "Decide whether the learned capability applies to this unrelated task. Return exactly one label: applicable or not-applicable.",
    skill_kind: str = "procedure",
    max_practice_remediations: int = 1,
    curriculum_id: str | None = None,
    source_locator: str | None = None,
    source_id: str | None = None,
    source_content_hash: str | None = None,
    material_content_hash: str | None = None,
    qualification_tasks: Mapping[str, str] | None = None,
    independent_design: Mapping[str, Any] | None = None,
    baseline_preflight_id: str | None = None,
) -> dict[str, Any]:
    """Run one bounded source-to-durable-skill acquisition cycle.

    The learner supplies both the candidate surface and every behavioral answer.
    The engine owns persistence, trusted grading, qualification, promotion, and
    runtime discovery. No answer key is placed in learner prompts.
    """
    if not source_title.strip() or not source_text.strip() or not intent_key.strip() or not target_capability.strip():
        raise LearningError("source title, source text, intent key, and target capability are required")
    if qualification_tasks is not None:
        required_tasks = ("baseline", "practice", "runtime", "negative")
        if any(not isinstance(qualification_tasks.get(key), str) or not qualification_tasks[key].strip() for key in required_tasks):
            raise LearningError("qualification_tasks must provide non-empty baseline, practice, runtime, and negative tasks")
    if independent_design is not None:
        if independent_design.get("schema") != "rex-learning-independent-test-design-v1":
            raise LearningError("independent_design must be a bound independent test design")
        if "answer_key" in independent_design or "expected_case_results" in independent_design:
            raise LearningError("independent design cannot contain answer scaffolding")
        phases = ("baseline", "practice", "posttest", "retest", "negative", "fresh", "control")
        phase_cases = independent_design.get("phase_cases")
        phase_tasks = independent_design.get("phase_tasks")
        if (not isinstance(phase_cases, Mapping) or not isinstance(phase_tasks, Mapping)
                or set(phase_cases) != set(phases) or set(phase_tasks) != set(phases)
                or any(not isinstance(phase_cases.get(key), str) or not phase_cases[key].strip() for key in phases)
                or any(not isinstance(phase_tasks.get(key), str) or not phase_tasks[key].strip() for key in phases)
                or len(set(phase_cases.values())) != len(phases)):
            raise LearningError("independent design must provide unique case IDs and tasks for every phase")
    if grader.evaluator_id not in engine.trusted_evaluators:
        raise LearningError("grader must be registered as a trusted evaluator")
    if fresh_learner is None:
        raise LearningError("fresh_learner is required to prove fresh-context reuse")
    control_learner_was_supplied = control_learner is not None
    if control_learner is None:
        control_learner = fresh_learner
    curriculum = engine.store.read("curricula", curriculum_id) if curriculum_id else engine.create_curriculum(source_title, target_capability)
    if curriculum_id and curriculum.get("status") != "active":
        raise LearningError("shared acquisition curriculum must be active")
    material_hash = hashlib.sha256(source_text.encode()).hexdigest()
    if material_content_hash is not None and material_content_hash != material_hash:
        raise LearningError("acquisition material content hash mismatch")
    source_hash = source_content_hash or material_hash
    if not isinstance(source_hash, str) or not source_hash.strip():
        raise LearningError("source content hash is required")
    if source_id is not None:
        source = engine.store.read("sources", source_id)
        if source.get("curriculum_id") != curriculum["id"]:
            raise LearningError("shared acquisition source/curriculum mismatch")
        if source.get("content_hash") != source_hash:
            raise LearningError("shared acquisition source content mismatch")
    else:
        source = engine.add_source(curriculum["id"], source_title, source_locator or f"memory://{source_hash}", content_hash=source_hash)
    note = engine.record_study_note(curriculum["id"], source["id"], {"kind": "instructional_source", "claim": source_text, "conditions": [source["id"]]})
    if skill_kind not in {"declarative", "procedure", "workflow"}:
        raise LearningError("skill_kind must be declarative, procedure, or workflow")
    intent = {"key": intent_key, "target_capability": target_capability, "skill_kind": skill_kind}
    case = {"id": intent_key, "source": {"title": source_title, "text": source_text, "source_refs": [source["id"], note["id"]]}, "intent": intent}
    provenance = {"provider": learner.provider, "session_id": learner.session_id, "role": "candidate_generator"}

    # Bind and validate an independent design before any learner call. A
    # malformed frozen design is a protocol failure, not a learner outcome.
    bound_design: dict[str, Any] | None = None
    if independent_design is not None and "phase_cases" in independent_design:
        request = {
            "schema": "rex-learning-test-design-request-v1", "status": "awaiting_independent_design",
            "discovery_provenance": {"provider": learner.provider, "session_id": learner.session_id},
            "proposal_id": str(independent_design.get("proposal_id", intent_key)), "intent_key": intent_key,
            "curriculum_id": curriculum["id"], "curriculum_intent_id": str(independent_design.get("curriculum_intent_id", intent_key)),
            "source_id": source["id"], "source_refs": [source["id"], note["id"]], "source_hash": source_hash,
            "target_capability": target_capability, "procedure": str(independent_design.get("procedure", "")), "skill_kind": skill_kind,
        }
        canonical = dict(independent_design)
        canonical.update({key: request[key] for key in ("proposal_id", "intent_key", "curriculum_id", "curriculum_intent_id", "source_id", "source_refs", "source_hash", "target_capability", "skill_kind")})
        canonical["cases"] = list(cast(Mapping[str, str], independent_design["phase_cases"]).values())
        canonical["discovery_provenance"] = request["discovery_provenance"]
        bound_design = bind_independent_competence_test_design(request=request, design=canonical, designer_provenance=cast(Mapping[str, Any], independent_design["designer_provenance"]))

    if not baseline_task.strip():
        raise LearningError("baseline_task is required")
    if not isinstance(max_practice_remediations, int) or isinstance(max_practice_remediations, bool) or max_practice_remediations < 0:
        raise LearningError("max_practice_remediations must be a non-negative integer")
    tasks = qualification_tasks or {
        "baseline": baseline_task,
        "practice": practice_task,
        "runtime": runtime_task,
        "negative": negative_task,
    }
    baseline_case = independent_design.get("phase_cases", {}).get("baseline", "pre") if independent_design else "pre"
    baseline_prompt = independent_design.get("phase_tasks", {}).get("baseline", tasks["baseline"]) if independent_design else tasks["baseline"]
    baseline_record: dict[str, Any] | None = None
    pre_attempt: dict[str, Any] | None = None
    pre_eval: dict[str, Any] | None = None
    if baseline_preflight_id is not None:
        baseline_record, skill, pre_attempt, pre_eval = _reuse_baseline_preflight(
            engine=engine,
            baseline_preflight_id=baseline_preflight_id,
            curriculum_id=curriculum["id"],
            source_id=source["id"],
            source_hash=source_hash,
            intent_key=intent_key,
            target_capability=target_capability,
        )
        baseline_case = str(pre_attempt.get("responses", [{}])[0].get("case", baseline_case))
        baseline_surface = {"responses": list(pre_attempt["responses"])}
    else:
        skill = engine.create_skill_hypothesis(curriculum["id"], {
            "key": intent_key,
            "kind": skill_kind,
            "claim": target_capability,
            "applicability": [target_capability],
            "operational_procedure": "Pending learner acquisition.",
        })
        baseline_surface = {} if bound_design is not None else learner.answer(task=f"pre|case={baseline_case}|{baseline_prompt}", material="")

    def execute(phase: str, case_id: str, result: Mapping[str, Any], *, source_free: bool, novel_transfer: bool, revision: str = "") -> tuple[dict[str, Any], dict[str, Any]]:
        if bound_design is not None:
            design_phase = {"pretest": "baseline", "practice": "practice", "posttest": "posttest", "retest": "retest", "negative": "negative", "edge": "fresh", "adversarial": "control"}[phase]
            actual_case = str(bound_design["phase_cases"][design_phase])
            test = define_test_from_independent_design(engine=engine, curriculum_id=curriculum["id"], skill_id=skill["id"], design=bound_design, phase=phase)
            execution = execute_bound_independent_test(engine=engine, test=test, learner=control_learner if phase == "adversarial" else (fresh_learner if phase in {"posttest", "retest", "negative", "edge"} else learner), task=f"{phase}|case={actual_case}|{bound_design['phase_tasks'][design_phase]}", material=source_text if phase == "practice" else "", revision=revision or ("" if phase in {"pretest", "practice", "negative"} else _procedure_text(skill.get("operational_procedure"))), task_id="fresh-runtime-task" if phase == "edge" else f"{phase}-{actual_case}")
            return execution["attempt"], execution["evaluation"]
        spec = _test_spec(grader, phase, case_id, source_free=source_free, novel_transfer=novel_transfer)
        if independent_design is not None:
            spec["independent_design_lineage"] = {
                key: independent_design[key]
                for key in ("proposal_id", "source_hash", "target_capability", "skill_kind")
                if key in independent_design
            }
            spec["independent_design_provenance"] = {
                "designer": dict(independent_design.get("designer_provenance", {})),
                "design_schema": independent_design.get("schema"),
            }
        test = engine.define_test(curriculum["id"], skill["id"], spec)
        attempt = engine.record_attempt(test["id"], {"responses": [_response(result, case_id)]}, task_id=f"{phase}-{case_id}")
        return attempt, engine.evaluate_trusted_attempt(attempt["id"])

    if baseline_preflight_id is None:
        pre_attempt, pre_eval = execute("pretest", "pre", baseline_surface, source_free=True, novel_transfer=False)
        pre_test = engine.store.read("tests", pre_attempt["test_id"])
        baseline_preflight = _persist_baseline_preflight(
            engine=engine, curriculum_id=curriculum["id"], source_id=source["id"],
            source_hash=source_hash, intent_key=intent_key,
            target_capability=target_capability, skill_id=skill["id"],
            test=pre_test, attempt=pre_attempt, evaluation=pre_eval,
        )
    else:
        assert baseline_record is not None
        baseline_preflight = baseline_record
    assert pre_attempt is not None and pre_eval is not None
    if pre_eval.get("score") == 1.0:
        raise CeilingBaselineError(
            "trusted baseline preflight shows no measurable headroom",
            details={"code": CeilingBaselineError.code, "pre_score": pre_eval.get("score"), "baseline_preflight_id": baseline_preflight["id"]},
        )
    control_eval: dict[str, Any] | None = None
    if control_learner_was_supplied:
        if bound_design is not None:
            _, control_eval = execute("adversarial", "control", {}, source_free=True, novel_transfer=True)
        else:
            control = control_learner.answer(task=f"control|case=control|{tasks['runtime']}", material="")
            control_test = engine.define_test(curriculum["id"], skill["id"], _test_spec(grader, "adversarial", "control", source_free=True, novel_transfer=True))
            control_attempt = engine.record_attempt(control_test["id"], {"responses": [_response(control, "control")]}, task_id="fresh-control-task")
            control_eval = engine.evaluate_trusted_attempt(control_attempt["id"])
    if control_learner_was_supplied and control_eval is not None and control_eval.get("score") == 1.0:
        raise NonDiscriminatingControlError(
            "matched control has no measurable acquisition headroom",
            details={"code": NonDiscriminatingControlError.code, "control_score": control_eval.get("score"), "control_evaluation_id": control_eval.get("id")},
        )
    candidate_surface = learner.answer(
        task=(
            "candidate|case=candidate|identify only the target capability below and return an actionable, "
            "source-grounded procedure or principle with applicability, preconditions, limitations, and "
            "verification method. Do not broaden it to neighboring topics from the source. "
            f"TARGET CAPABILITY: {target_capability} "
            "Return JSON with a responses list containing exactly one response with case candidate, plus the procedure field."
        ),
        material=source_text,
    )
    candidate = build_candidate_skill(case=case, qwen_surface=_candidate_surface(candidate_surface), provenance=provenance)
    candidate["revision_history"] = []
    candidate["proposed_skill"]["kind"] = skill_kind
    engine.store.write("candidate_skills", candidate["candidate_id"], candidate)
    proposed = candidate["proposed_skill"]
    _sync_skill_from_candidate(skill, proposed, target_capability)
    engine.store.write("skills", skill["id"], skill)
    practice_surface = {} if bound_design is not None else learner.answer(task=f"practice|case=practice|{tasks['practice']}", material=source_text)
    practice_attempt, practice_eval = execute("practice", "practice", practice_surface, source_free=False, novel_transfer=True)
    practice_attempts = [practice_attempt]
    practice_evaluations = [practice_eval]
    revision = ""
    remediation_count = 0
    while not practice_eval["passed"] and remediation_count < max_practice_remediations:
        remediation_count += 1
        revision = _practice_revision_feedback(
            practice_eval,
            remediation_count,
            skill_kind=skill_kind,
            target_capability=target_capability,
        )
        revised_surface = _candidate_surface(learner.answer(
            task="candidate|case=candidate|revise the candidate procedure after the recorded practice failure; return JSON with a responses list containing exactly one response with case candidate, plus the corrected actionable procedure.",
            material=source_text,
            revision=revision,
        ))
        candidate["revision_history"].append({
            "kind": "practice_failure_revision",
            "remediation_number": remediation_count,
            "failed_attempt_id": practice_attempt["id"],
            "failed_evaluation_id": practice_eval["id"],
            "instruction": revision,
            "surface": dict(revised_surface),
        })
        _merge_candidate_surface(candidate, revised_surface)
        proposed = candidate["proposed_skill"]
        _sync_skill_from_candidate(skill, proposed, target_capability)
        engine.store.write("candidate_skills", candidate["candidate_id"], candidate)
        engine.store.write("skills", skill["id"], skill)
        practice_surface = {} if bound_design is not None else learner.answer(task=f"practice|case=practice|Retry: {tasks['practice']}", material=source_text, revision=revision)
        practice_attempt, practice_eval = execute("practice", "practice", practice_surface, source_free=False, novel_transfer=True, revision=revision)
        practice_attempts.append(practice_attempt)
        practice_evaluations.append(practice_eval)
    if not practice_eval["passed"]:
        raise LearningError(f"learner failed bounded practice after {max_practice_remediations} remediation attempt(s)")

    procedure = _procedure_text(proposed.get("operational_procedure"))
    post_result = {} if bound_design is not None else fresh_learner.answer(task=f"posttest|case=post|{tasks['runtime']}", material="", revision=procedure)
    _, post_eval = execute("posttest", "post", post_result, source_free=True, novel_transfer=True)
    retest_result = {} if bound_design is not None else fresh_learner.answer(task=f"retest|case=retest|{tasks['runtime']}", material="", revision=procedure)
    _, retest_eval = execute("retest", "retest", retest_result, source_free=True, novel_transfer=True)
    negative_result = {} if bound_design is not None else fresh_learner.answer(task=f"negative|case=negative|{tasks['negative']}", material="", revision=procedure)
    _, negative_eval = execute("negative", "negative", negative_result, source_free=True, novel_transfer=True)

    if not control_learner_was_supplied:
        if bound_design is not None:
            _, control_eval = execute("adversarial", "control", {}, source_free=True, novel_transfer=True)
        else:
            control = control_learner.answer(task=f"control|case=control|{tasks['runtime']}", material="")
            control_test = engine.define_test(curriculum["id"], skill["id"], _test_spec(grader, "adversarial", "control", source_free=True, novel_transfer=True))
            control_attempt = engine.record_attempt(control_test["id"], {"responses": [_response(control, "control")]}, task_id="fresh-control-task")
            control_eval = engine.evaluate_trusted_attempt(control_attempt["id"])

    review_input = {
        "candidate": candidate,
        "baseline": {"attempt": pre_attempt, "evaluation": pre_eval},
        "practice": {"attempt": practice_attempt, "evaluation": practice_eval, "attempts": practice_attempts, "evaluations": practice_evaluations},
        "posttest": {"evaluation": post_eval},
        "retest": {"evaluation": retest_eval},
        "negative": {"evaluation": negative_eval},
        "control": {"evaluation": control_eval},
        "source": case["source"],
    }
    claim = proposed.get("claim") or target_capability
    if isinstance(claim, list):
        claim = "; ".join(str(item) for item in claim)
    decision, reviewer_provenance = _invoke_reviewer(reviewer, candidate, review_input, source["id"], str(claim))
    reviewed = attach_review(candidate, decision, reviewer=reviewer_provenance)
    if decision.get("decision") == "DEFER":
        run_id = "acquisition-" + hashlib.sha256((source_hash + candidate["candidate_id"]).encode()).hexdigest()[:16]
        engine.store.write("candidate_skills", candidate["candidate_id"], reviewed)
        result = {
            "schema": "rex-learning-acquisition-run-v1",
            "run_id": run_id,
            "curriculum": curriculum,
            "source": source,
            "candidate": reviewed,
            "practice": {"attempt": practice_attempt, "evaluation": practice_eval, "attempts": practice_attempts, "evaluations": practice_evaluations, "failed_attempt": practice_attempts[0] if revision else None, "failed_evaluation": practice_evaluations[0] if revision else None, "revision": revision},
            "promotion": {"state": "deferred", "reason": "explicit reviewer judgment required"},
            "evidence": {"baseline": pre_eval, "posttest": post_eval, "retest": retest_eval, "negative": negative_eval, "control": control_eval},
            "artifacts": {"source_hash": source_hash, "provider": learner.provider, "learner_session_id": learner.session_id, "reviewer": reviewer_provenance},
        }
        engine.store.write("acquisition_runs", run_id, result)
        return result

    promoted = engine.promote_candidate_skill_from_records(
        reviewed,
        curriculum_id=curriculum["id"],
        pretest_attempt_id=pre_attempt["id"],
        pretest_evaluation_id=pre_eval["id"],
        posttest_evaluation_id=post_eval["id"],
        retest_evaluation_id=retest_eval["id"],
        negative_evaluation_id=negative_eval["id"],
        control_evaluation_id=control_eval["id"] if control_learner_was_supplied and control_eval is not None else None,
    )
    evidence = engine.derive_candidate_behavioral_evidence(pretest_attempt_id=pre_attempt["id"], pretest_evaluation_id=pre_eval["id"], posttest_evaluation_id=post_eval["id"], retest_evaluation_id=retest_eval["id"], negative_evaluation_id=negative_eval["id"], control_evaluation_id=control_eval["id"] if control_learner_was_supplied and control_eval is not None else None)

    selected = engine.discover_applicable_skills({"requirements": _as_list(proposed.get("scope") or [target_capability])})
    if not selected or selected[0]["skill_id"] != promoted["id"]:
        raise LearningError("promoted skill was not discovered for its applicability")
    if bound_design is not None:
        fresh_attempt, fresh_eval = execute("edge", "fresh", {}, source_free=True, novel_transfer=True)
    else:
        fresh_result = fresh_learner.answer(task=f"fresh|case=fresh|{tasks['runtime']}", material="", revision=selected[0]["operational_context"]["procedure"])
        fresh_test = engine.define_test(curriculum["id"], promoted["id"], _test_spec(grader, "edge", "fresh", source_free=True, novel_transfer=True))
        fresh_attempt = engine.record_attempt(fresh_test["id"], {"responses": [_response(fresh_result, "fresh")]}, task_id="fresh-runtime-task")
        fresh_eval = engine.evaluate_trusted_attempt(fresh_attempt["id"])
    application = engine.apply_skill(selected[0]["key"], "fresh-runtime-task", selected[0]["operational_context"]["procedure"], outcome={"passed": fresh_eval["passed"]}, curriculum_id=curriculum["id"], selection_reason=selected[0]["why_selected"], operational_context=selected[0]["operational_context"], execution_ref=fresh_attempt["id"], evaluator_ref=fresh_eval["id"])
    application = engine.attach_application_evidence(application["id"], execution_ref=fresh_attempt["id"], evaluation_id=fresh_eval["id"], outcome={"passed": fresh_eval["passed"]})

    negative_selected = bool(engine.discover_applicable_skills({"requirements": ["unrelated capability"]}))
    result = {
        "schema": "rex-learning-acquisition-run-v1",
        "run_id": "acquisition-" + hashlib.sha256((source_hash + candidate["candidate_id"]).encode()).hexdigest()[:16],
        "curriculum": curriculum,
        "source": source,
        "candidate": reviewed,
        "practice": {"attempt": practice_attempt, "evaluation": practice_eval, "attempts": practice_attempts, "evaluations": practice_evaluations, "failed_attempt": practice_attempts[0] if revision else None, "failed_evaluation": practice_evaluations[0] if revision else None, "revision": revision},
        "promotion": promoted,
        "evidence": evidence,
        "fresh_context": {"selected_skill_id": selected[0]["skill_id"], "application": application, "evaluation": fresh_eval},
        "control": control_eval,
        "negative_applicability": negative_selected,
        "artifacts": {"source_hash": source_hash, "provider": learner.provider, "learner_session_id": learner.session_id, "fresh_learner_session_id": fresh_learner.session_id, "control_learner_session_id": control_learner.session_id, "reviewer": reviewer_provenance},
    }
    engine.store.write("acquisition_runs", result["run_id"], result)
    return result


def learn_multiple_from_source(
    *,
    engine: LearningEngine,
    learner: Learner,
    source_title: str,
    source_text: str,
    curriculum_title: str,
    capabilities: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Acquire multiple independently qualified skills from one source.

    Each capability supplies its own intent key, target, trusted grader,
    reviewer, and fresh learner. The shared source/curriculum is only the
    instructional context; skill identity, tests, evidence, and promotion
    remain separate. Adding a capability therefore adds data and evaluation,
    not a branch in the acquisition algorithm.
    """
    if not capabilities:
        raise LearningError("at least one capability is required")
    curriculum = engine.create_curriculum(curriculum_title, source_title)
    runs: list[dict[str, Any]] = []
    for capability in capabilities:
        if not isinstance(capability, Mapping):
            raise LearningError("capability specifications must be objects")
        required = ("intent_key", "target_capability", "grader", "reviewer", "fresh_learner")
        missing = next((field for field in required if field not in capability), None)
        if missing:
            raise LearningError(f"capability is missing {missing}")
        runs.append(learn_from_source(
            engine=engine,
            learner=capability.get("learner", learner),
            source_title=source_title,
            source_text=source_text,
            intent_key=str(capability["intent_key"]),
            target_capability=str(capability["target_capability"]),
            grader=capability["grader"],
            reviewer=capability["reviewer"],
            fresh_learner=capability["fresh_learner"],
            control_learner=capability.get("control_learner"),
            baseline_task=str(capability.get("baseline_task", "Perform the target capability on a concrete held-out task and return the result.")),
            practice_task=str(capability.get("practice_task", "Apply the learned capability to the held-out practice task and return the result.")),
            runtime_task=str(capability.get("runtime_task", "Apply the learned capability to the held-out runtime task and return the result.")),
            negative_task=str(capability.get("negative_task", "Decide whether the learned capability applies to this unrelated task. Return exactly one label: applicable or not-applicable.")),
            skill_kind=str(capability.get("skill_kind", "procedure")),
            max_practice_remediations=int(capability.get("max_practice_remediations", 1)),
            curriculum_id=curriculum["id"],
            source_locator=str(capability.get("source_locator", "")) or None,
        ))
    return {
        "schema": "rex-learning-multi-skill-acquisition-v1",
        "curriculum": engine.store.read("curricula", curriculum["id"]),
        "source": runs[0]["source"],
        "skills": [run["promotion"] for run in runs],
        "runs": runs,
    }
