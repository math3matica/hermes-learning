from __future__ import annotations

import pytest

from rex_learning.jobs import LearningJobStore, run_learning_job
from rex_learning.store import LearningStore


def test_learning_job_is_durable_and_idempotent(tmp_path):
    jobs = LearningJobStore(LearningStore(tmp_path))
    first = jobs.enqueue(source="book.epub", objective="learn testing")
    assert jobs.enqueue(source="book.epub", objective="learn testing")["id"] == first["id"]
    events = []
    completed = run_learning_job(
        jobs,
        first["id"],
        lambda job: {"source": job["source"], "ok": True},
        lifecycle=lambda name, payload: events.append((name, payload)),
    )
    assert completed["state"] == "completed"
    assert completed["result"]["ok"] is True
    assert [name for name, _ in events] == ["learning_job_started", "learning_job_completed"]
    assert run_learning_job(jobs, first["id"], lambda _: pytest.fail("completed job reran"))["state"] == "completed"


def test_learning_job_failure_is_persisted(tmp_path):
    jobs = LearningJobStore(LearningStore(tmp_path))
    job = jobs.enqueue(source="book.epub", objective="learn testing")
    with pytest.raises(RuntimeError, match="boom"):
        run_learning_job(jobs, job["id"], lambda _: (_ for _ in ()).throw(RuntimeError("boom")))
    assert jobs.get(job["id"])["state"] == "failed"
    assert "RuntimeError: boom" in jobs.get(job["id"])["error"]
