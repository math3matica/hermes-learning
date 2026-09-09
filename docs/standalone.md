# Standalone repository contract

## Boundaries

This repository contains the complete `rex_learning` package, its project-local Hermes plugin, tests, and package-local scripts. It is installable from any working directory. No runtime module resolves a path relative to the original Hermes checkout, and no test requires a particular user or `PYTHONPATH`.

## Job lifecycle

`run` is the durable enqueue boundary: it validates source/objective, computes a deterministic identity, and atomically writes `queued` state. It never invokes the model. A host-owned worker calls `worker <job-id>`, which claims `queued`/retryable `failed`, invokes the learning executor through the host's callable `ctx.llm`, and atomically persists `completed` or `failed`. Repeating a completed worker call is idempotent and does not invoke the executor again.

This is the canonical asynchronous contract. There is no implicit synchronous compatibility path because that would make a command response perform expensive model work and would break durable ownership. Hosts that historically expected a synchronous return must explicitly enqueue and then call the worker operation; this compatibility is visible and auditable.

## Plugin contract

The plugin has no Hermes-private imports. It registers the `rex-learning` command, optional CLI command, and skill through the public context methods. State is resolved from host context first, then explicit environment configuration, then the standard Hermes home. `ctx.llm` is the only model boundary; provider and model identifiers are not accepted from user/job arguments.

## Source and security hygiene

Source paths are persisted as supplied job data. A host integration must authorize paths before enqueueing or executing them; the package does not broaden filesystem access. Keep credentials and model state in host-managed locations, outside version control. Generated evidence is append-only and should be stored outside the repository.
