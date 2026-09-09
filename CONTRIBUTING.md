# Contributing

This is a provider-neutral learning package and thin Hermes Agent plugin.
Keep host model routing, durable state, provenance, and evidence gates explicit;
do not add provider credentials, private source material, generated state, or
model artifacts.

## Development

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q
python -m compileall -q rex_learning .hermes/plugins/rex-learning
```

Add deterministic tests for behavior changes, including malformed and negative
inputs. Provider-backed and physical Hermes/voice checks are separate from CI.
Use `git diff --check` before submitting a change. The project is licensed
under MIT; see `LICENSE`.