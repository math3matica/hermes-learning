# Rex Learning

Standalone, provider-neutral learning orchestration for Hermes Agent. The package is independently installable and does not import Hermes internals, a source checkout, a username-specific path, or `PYTHONPATH`.

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

## Security and privacy

Do not commit model credentials, host state, evidence roots, source books, generated artifacts, or provider responses. Keep state outside the checkout where possible. The plugin never accepts provider/model selection from job input and fails closed when `ctx.llm` is unavailable. Source paths are treated as job data and must be authorized by the host before execution.

See `docs/standalone.md` and `examples/config.example.toml`.

## License

MIT. See `LICENSE`.
