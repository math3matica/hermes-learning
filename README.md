# Hermes Learning

Standalone, provider-neutral learning orchestration for Hermes Agent. This is
**not model training or fine-tuning**: it studies authorized source material,
extracts structured candidate knowledge/procedures, validates evidence, and
offers demonstrated artifacts for later Hermes use.

This is a third-party/open-source project for the Hermes Agent ecosystem, not an
official Nous Research component.

## Pipeline

```text
source material -> structural analysis -> knowledge/skill extraction
-> prerequisites, methods, examples -> structured durable artifacts
-> validation/deduplication -> bounded retrieval/use by Hermes
```

The host owns LLM inference and credentials. The package is provider-neutral and
does not select a model from job input.

## Features and boundaries

- Durable, idempotent job lifecycle: enqueue, worker claim, completion/failure,
  retry, and restart-safe state.
- Provenance-bearing sources, study notes, knowledge, candidates, tests,
  evaluations, skill versions, and application records.
- Fail-closed promotion gates; a summary, retrieval result, model confidence,
  or caller-supplied `passed=true` is not competence evidence.
- Thin Hermes plugin using host-owned `ctx.llm`; generated state belongs under
  the active profile (`state_dir`, `HERMES_REX_LEARNING_HOME`, or
  `$HERMES_HOME/rex-learning`).

The repository contains code, tests, plugin metadata, and synthetic examples.
Private source books, provider responses, generated artifacts/state, model
weights, and Hermes profile data are intentionally excluded from Git.

## Install

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'
pytest
```

The durable filesystem store defaults to `$HERMES_HOME/rex-learning`, or can be selected with `HERMES_REX_LEARNING_HOME`. A host context's `state_dir`/`learning_state_dir` takes precedence.

## Hermes plugin

The project-local plugin is `.hermes/plugins/rex-learning/`. Enable project plugins using the host's normal opt-in mechanism (`HERMES_ENABLE_PROJECT_PLUGINS=1`). The plugin is deliberately thin: Hermes owns routing and `ctx.llm`; Rex Learning owns durable records and orchestration.

- `run <source> <objective>` only durably enqueues a job.
- `worker <job-id>` claims and executes one queued job.
- `status` reports inspectable state counts.

This separation is intentional. There is no hidden synchronous execution route. Any host scheduler or post-call worker may invoke the explicit worker operation.

## Synthetic example

An authorized Markdown source might state: “For this fixture, validate the
configuration before deploying; do not deploy when validation fails.” A worker
can extract a candidate with a prerequisite, procedure, and negative
applicability case. The validator stores source references and test evidence;
only an independently evaluated held-out result can promote a version for
bounded requirement-based retrieval. This example is illustrative and does
not claim that a live model learned it.

## Security and privacy

Do not commit model credentials, host state, evidence roots, source books, generated artifacts, or provider responses. Keep state outside the checkout where possible. The plugin never accepts provider/model selection from job input and fails closed when `ctx.llm` is unavailable. Source paths are treated as job data and must be authorized by the host before execution.

See `docs/standalone.md` and `examples/config.example.toml`.

## Requirements and verification

Python 3.11+ is required. Runtime dependencies are Python's standard library;
`pytest>=8` is only the test extra. Hermes Agent is required for the plugin
surface, and a host-provided callable `ctx.llm` is required for live worker
execution. No GPU, Android device, or particular provider is required for the
package tests.

The install command above, followed by `pytest`, verifies packaging and the
provider-neutral deterministic lifecycle. It does not verify provider
cognition, broad real-world transfer, or voice/hardware behavior. See
`MANUAL_ACCEPTANCE.md` for live-provider and integration gates.

## License

MIT. See `LICENSE`.
