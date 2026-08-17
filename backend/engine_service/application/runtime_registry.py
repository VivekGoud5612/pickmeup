"""In-memory registry of active Engine runtime workers."""

from __future__ import annotations

from uuid import UUID

from backend.engine_service.domain.entities.runtime_worker import RuntimeWorker


class RuntimeRegistry:
    """Tracks execution-plane state by control-plane run identifier."""

    def __init__(self) -> None:
        self._workers: dict[UUID, RuntimeWorker] = {}

    def add(self, worker: RuntimeWorker) -> None:
        if worker.run_id in self._workers:
            raise ValueError(f"A runtime worker already exists for run {worker.run_id}")
        self._workers[worker.run_id] = worker

    def get(self, run_id: UUID) -> RuntimeWorker:
        try:
            return self._workers[run_id]
        except KeyError as exc:
            raise LookupError(f"No runtime worker exists for run {run_id}") from exc

    def remove(self, run_id: UUID) -> RuntimeWorker | None:
        return self._workers.pop(run_id, None)

    def active_run_ids(self) -> tuple[UUID, ...]:
        return tuple(self._workers)
