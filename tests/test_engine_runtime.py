from __future__ import annotations

import unittest
from uuid import uuid4

from backend.contracts.engine.enums import EngineStatus
from backend.contracts.engine.events.training_events import TrainingStartedEvent
from backend.engine_service.application.runtime_registry import RuntimeRegistry
from backend.engine_service.domain.entities.runtime_worker import RuntimeWorker


class RecordingHandle:
    def __init__(self) -> None:
        self.operations: list[str] = []

    def start(self) -> None:
        self.operations.append("start")

    def pause(self) -> None:
        self.operations.append("pause")

    def resume(self) -> None:
        self.operations.append("resume")

    def stop(self) -> None:
        self.operations.append("stop")


class RuntimeWorkerTests(unittest.TestCase):
    def test_worker_lifecycle_is_execution_plane_state(self) -> None:
        run_id = uuid4()
        handle = RecordingHandle()
        registry = RuntimeRegistry()
        worker = RuntimeWorker(run_id=run_id, execution_handle=handle)
        registry.add(worker)

        worker.start()
        worker.pause()
        worker.resume()
        worker.stop()

        self.assertEqual(handle.operations, ["start", "pause", "resume", "stop"])
        self.assertIs(worker.status, EngineStatus.STOPPED)
        self.assertIs(registry.get(run_id), worker)

    def test_engine_events_carry_control_plane_run_identity(self) -> None:
        run_id = uuid4()
        event = TrainingStartedEvent(run_id=run_id)
        self.assertEqual(event.run_id, run_id)
        with self.assertRaises(AttributeError):
            event.run_id = uuid4()


if __name__ == "__main__":
    unittest.main()
