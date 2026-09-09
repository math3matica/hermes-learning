from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Mapping
from typing import Any

from .store import LearningStore


JOB_STATES = ("queued", "running", "completed", "failed")


def _job_id(source: str, objective: str) -> str:
    digest = hashlib.sha256(f"{source}\0{objective}".encode("utf-8")).hexdigest()[:24]
    return f"learning-job-{digest}"


class LearningJobStore:
    """Durable, idempotent job boundary for host-owned Learning execution."""

    def __init__(self, store: LearningStore):
        self.store = store

    def enqueue(self, *, source: str, objective: str) -> dict[str, Any]:
        if not source.strip() or not objective.strip():
            raise ValueError("source and objective are required")
        job = {
            "schema": "rex-learning-job-v1",
            "id": _job_id(source, objective),
            "source": source,
            "objective": objective,
            "state": "queued",
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        try:
            existing = self.store.read("learning_jobs", job["id"])
        except KeyError:
            return self.store.write("learning_jobs", job["id"], job)
        if any(existing.get(key) != job[key] for key in ("source", "objective")):
            raise ValueError("learning job identity collision")
        return existing

    def get(self, job_id: str) -> dict[str, Any]:
        return self.store.read("learning_jobs", job_id)

    def claim(self, job_id: str) -> dict[str, Any]:
        job = self.get(job_id)
        if job["state"] == "completed":
            return job
        if job["state"] == "running":
            raise RuntimeError("learning job is already running")
        if job["state"] not in {"queued", "failed"}:
            raise ValueError("invalid learning job state")
        job.update({"state": "running", "updated_at": time.time()})
        return self.store.write("learning_jobs", job_id, job)

    def finish(self, job_id: str, *, result: Mapping[str, Any]) -> dict[str, Any]:
        job = self.get(job_id)
        if job["state"] != "running":
            raise RuntimeError("learning job is not running")
        job.update({"state": "completed", "result": dict(result), "updated_at": time.time()})
        return self.store.write("learning_jobs", job_id, job)

    def fail(self, job_id: str, *, error: str) -> dict[str, Any]:
        job = self.get(job_id)
        if job["state"] != "running":
            raise RuntimeError("learning job is not running")
        job.update({"state": "failed", "error": error[:2000], "updated_at": time.time()})
        return self.store.write("learning_jobs", job_id, job)


def run_learning_job(
    jobs: LearningJobStore,
    job_id: str,
    executor: Callable[[dict[str, Any]], Mapping[str, Any]],
    *,
    lifecycle: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Claim and execute one job, preserving terminal state on failure."""
    job = jobs.claim(job_id)
    if job["state"] == "completed":
        return job
    if lifecycle is not None:
        lifecycle("learning_job_started", {"job_id": job_id})
    try:
        result = executor(job)
    except Exception as exc:
        failed = jobs.fail(job_id, error=f"{type(exc).__name__}: {exc}")
        if lifecycle is not None:
            lifecycle("learning_job_failed", {"job_id": job_id, "error": failed["error"]})
        raise
    completed = jobs.finish(job_id, result=result)
    if lifecycle is not None:
        lifecycle("learning_job_completed", {"job_id": job_id})
    return completed
