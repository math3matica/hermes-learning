---
title: Rex Learning orchestration
status: active
---

# Rex Learning

`run` durably enqueues a job under the host-owned LearningStore. A separate
`worker <job-id>` operation claims and executes one job through `ctx.llm`; job
state is persisted as queued, running, completed, or failed. If the host
provides a callable `ctx.subagent_lifecycle`, the worker emits bounded start,
failure, and completion notifications through that callback. No provider or
model is selected by this plugin.

`rex-learning` is a thin Hermes integration around the provider-neutral
`rex_learning` package. It intentionally does not select a provider, model,
agent, or personal filesystem path.

## Surfaces

- `/rex-learning status` reports profile-local job/catalog counts.
- `/rex-learning run <source> <objective>` durably enqueues a source-to-study job.
- `/rex-learning worker <job-id>` explicitly claims and executes one queued job.
- The native CLI command `rex-learning` exposes the same operations.

When available, the host's `ctx.llm` is used for bounded JSON generation. The
host remains responsible for routing and credentials. If it is unavailable,
the job fails closed with `host_llm_unavailable`.

State is owned by Hermes: a context `state_dir` is preferred, then
`HERMES_REX_LEARNING_HOME`, then `HERMES_HOME/rex-learning`. Generated
candidates remain subject to the existing evidence-gated validation and
promotion boundaries; plugin registration or a successful job transport is not
learning evidence.
