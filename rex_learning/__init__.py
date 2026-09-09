"""Evidence-gated, inspectable skill acquisition for Rex Learning V1."""

from .engine import LearningEngine, LearningError
from .evaluator import CallbackEvaluator, ContractArtifactEvaluator, ExecutableArtifactGrader, ExecutableGrader, ExecutableProcessGrader, ExecutableProjectContract, ExecutableProjectEvaluator, EvaluatorInstabilityError, SemanticChecklistGrader, StableTrustedEvaluator, StructuredBehaviorGrader, TrustedEvaluator
from .learner import HostLLMLearner, Learner, LearnerError, OpenAICompatibleLearner
from .learner_contract import LearnerContractError, validate_learner_response
from .ingestion import ingest_epub, ingest_source
from .store import LearningStore
from .deep_study import DeepStudyEngine, STUDY_STATES
from .cognitive import CognitiveContextPolicy, CognitiveProvider, CognitiveResult, CognitiveRouter, CognitiveRequest, ProviderPolicy
from .instructional_abstraction import CurriculumIntent, InstructionalAbstraction, SourceContext
from .provider_boundary import ProviderAbstractionNormalization, normalize_provider_abstraction
from .jobs import JOB_STATES, LearningJobStore, run_learning_job
from .staged_abstraction import EpistemicLedger, StagedAbstractionPipeline, StagedAbstractionResult, validate_staged_abstraction
from .candidate_skill import (
    CANDIDATE_STATES,
    REVIEW_DECISIONS,
    attach_review,
    build_candidate_skill,
    promotion_boundary,
    validate_review_decision,
)
from .lightweight_abstraction import (
    CRITIC_OPERATION,
    INTEGRATED_OPERATION,
    IntegrityResult,
    apply_targeted_critic,
    assess_risks,
    normalize_integrated_answer,
    validate_integrated_answer,
)
from .acquisition import learn_from_source, learn_multiple_from_source
from .autonomous_study import autonomous_study
from .preflight import prepare_independent_designs, validate_pilot_preflight
from .entrypoint import LearningRuntime, run_learning
from .experiment_integrity import CardinalityError, freeze_hidden_artifact, require_exact_cardinality, verify_hidden_artifact
from .capability_discovery import bind_independent_competence_test_design, build_competence_test_requests, build_learning_treatments, define_test_from_independent_design, discover_capabilities, discover_capabilities_from_units, derive_independent_behavioral_evidence, execute_bound_independent_test, execute_learning_treatments, integrate_existing_skill_plan, normalize_independent_test_design, plan_existing_skill_integrations, preserve_conflict_plan, requalify_existing_skill_plan, select_learning_proposal_keys, select_learning_treatments

__all__ = [
    "CognitiveContextPolicy", "CognitiveProvider", "CognitiveRequest", "CognitiveResult", "CognitiveRouter", "LearningRuntime", "run_learning", "autonomous_study", "prepare_independent_designs", "validate_pilot_preflight", "ingest_epub", "ingest_source",
    "CardinalityError", "freeze_hidden_artifact", "require_exact_cardinality", "verify_hidden_artifact",
    "CurriculumIntent", "DeepStudyEngine", "EpistemicLedger", "ContractArtifactEvaluator", "ExecutableGrader", "ExecutableProcessGrader", "ExecutableProjectContract", "ExecutableProjectEvaluator",
 "ExecutableArtifactGrader", "EvaluatorInstabilityError", "StableTrustedEvaluator", "InstructionalAbstraction",
    "Learner", "HostLLMLearner", "LearnerContractError", "LearnerError", "LearningEngine", "LearningError", "LearningStore", "OpenAICompatibleLearner",
    "ProviderAbstractionNormalization", "ProviderPolicy", "SourceContext", "STUDY_STATES", "JOB_STATES", "LearningJobStore", "run_learning_job",
    "StagedAbstractionPipeline", "StagedAbstractionResult", "TrustedEvaluator", "normalize_provider_abstraction",
    "validate_staged_abstraction", "validate_learner_response", "CRITIC_OPERATION", "INTEGRATED_OPERATION", "IntegrityResult",
    "apply_targeted_critic", "assess_risks", "normalize_integrated_answer", "validate_integrated_answer",
    "CANDIDATE_STATES", "REVIEW_DECISIONS", "attach_review", "build_candidate_skill",
    "promotion_boundary", "validate_review_decision", "learn_from_source", "learn_multiple_from_source", "discover_capabilities", "discover_capabilities_from_units", "build_learning_treatments", "build_competence_test_requests", "bind_independent_competence_test_design", "define_test_from_independent_design", "derive_independent_behavioral_evidence", "execute_bound_independent_test", "select_learning_proposal_keys", "select_learning_treatments", "execute_learning_treatments", "plan_existing_skill_integrations", "integrate_existing_skill_plan", "requalify_existing_skill_plan", "preserve_conflict_plan",
]
