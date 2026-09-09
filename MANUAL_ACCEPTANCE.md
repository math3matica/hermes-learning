# Manual acceptance checklist

The following gates require a live host provider or Hermes integration and are
not established by the deterministic test suite:

- [ ] Configure a live provider through host-owned `ctx.llm` without putting
      credentials in the repository.
- [ ] Ingest synthetic or authorized source material and verify source hash and
      provenance.
- [ ] Exercise enqueue, worker claim, completion/failure, retry, and restart.
- [ ] Verify a generated candidate/skill passes the trusted evidence gates;
      model self-report and artifact presence are insufficient.
- [ ] Verify requirement-based retrieval and actual use by Hermes separately.
- [ ] Restart from the same private state root and verify persistence/idempotency.
- [ ] Confirm generated state, provider output, source material, and evidence
      remain outside Git.

Do not claim model cognition, broad transfer, or physical voice learning from
fixture tests, retrieval context, or a successful worker exit.