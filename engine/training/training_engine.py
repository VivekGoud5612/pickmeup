from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import UUID

import torch

from backend.contracts.engine.dto.responses import (
    EngineCheckpointEvaluationResponse,
    EngineCheckpointSavedResponse,
)
from backend.contracts.engine.enums import CheckpointType
from backend.contracts.engine.events.base import EngineEvent
from backend.contracts.engine.events.checkpoint_events import CheckpointCreatedEvent
from backend.contracts.engine.events.evaluation_events import (
    EvaluationCompletedEvent,
    EvaluationStartedEvent,
)
from backend.contracts.engine.events.training_events import (
    TrainingCompletedEvent,
    TrainingFailedEvent,
    TrainingInitializedEvent,
    TrainingPausedEvent,
    TrainingResumedEvent,
    TrainingStartedEvent,
    TrainingStoppedEvent,
)
from backend.contracts.engine.models.performance_metrics import PerformanceMetrics
from backend.contracts.engine.models.training_configuration import TrainingConfiguration
from backend.contracts.engine.models.training_progress import TrainingProgress
from backend.contracts.engine.enums import AgentRole, EngineStatus


class TrainingEngine:
    """
    Local execution handle for a training workload.

    It deliberately does not invent training results.  The existing MAPPO
    implementation remains in ``engine.simulation.main``; wiring its rich
    runtime telemetry into this service is a separate, explicit adapter task.
    """

    def __init__(self) -> None:
        self._training_configuration: TrainingConfiguration | None = None
        self._run_name: str | None = None
        self._run_id: UUID | None = None
        self._checkpoint_directory: Path | None = None
        self._publisher: Any = None
        self._state = EngineStatus.CREATED
        self._global_step = 0
        self._episode = 0
        self._progress = TrainingProgress(episode=0, step=0)
        self._metrics: PerformanceMetrics | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def initialize(
        self,
        config: TrainingConfiguration,
        run_id: UUID,
        run_name: str,
        checkpoint_directory: Path | None = None,
        publisher: Any = None,
    ) -> None:
        self._training_configuration = config
        self._run_id = run_id
        self._run_name = run_name
        self._publisher = publisher
        self._checkpoint_directory = checkpoint_directory or Path(f"checkpoints/{run_name}")
        self._checkpoint_directory.mkdir(parents=True, exist_ok=True)
        self._state = EngineStatus.INITIALIZED
        self._publish(TrainingInitializedEvent(run_id=run_id))

    def start(self) -> None:
        self._require_state(EngineStatus.INITIALIZED)
        self._state = EngineStatus.RUNNING
        self._publish(TrainingStartedEvent(run_id=self._require_run_id()))

    def pause(self) -> None:
        self._require_state(EngineStatus.RUNNING)
        self._state = EngineStatus.PAUSED
        self._publish(TrainingPausedEvent(run_id=self._require_run_id(), global_step=self._global_step))

    def resume(self) -> None:
        self._require_state(EngineStatus.PAUSED)
        self._state = EngineStatus.RUNNING
        self._publish(TrainingResumedEvent(run_id=self._require_run_id(), global_step=self._global_step))

    def stop(self) -> None:
        if self._state not in (EngineStatus.RUNNING, EngineStatus.PAUSED):
            raise RuntimeError("Training needs to be running or paused to stop")
        self._state = EngineStatus.STOPPED
        self._publish(TrainingStoppedEvent(run_id=self._require_run_id(), global_step=self._global_step))
        self._publish(
            TrainingCompletedEvent(
                run_id=self._require_run_id(),
                global_step=self._global_step,
                total_episodes=self._episode,
            )
        )

    def save_checkpoint(
        self,
        checkpoint_path: Path,
        checkpoint_type: CheckpointType = CheckpointType.PERIODIC,
    ) -> None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "global_step": self._global_step,
                "episode": self._episode,
                "run_name": self._run_name,
                "progress": {
                    "episode": self._progress.episode,
                    "step": self._progress.step,
                    "total_episodes": self._progress.total_episodes,
                    "total_steps": self._progress.total_steps,
                },
            },
            checkpoint_path,
        )
        self._publish(
            CheckpointCreatedEvent(
                checkpoint_path=checkpoint_path,
                checkpoint_type=checkpoint_type,
                global_step=self._global_step,
            )
        )

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self._device)
        self._global_step = int(checkpoint.get("global_step", 0))
        self._episode = int(checkpoint.get("episode", 0))
        progress = checkpoint.get("progress")
        if isinstance(progress, dict):
            self._progress = TrainingProgress(
                episode=int(progress.get("episode", 0)),
                step=int(progress.get("step", 0)),
                total_episodes=progress.get("total_episodes"),
                total_steps=progress.get("total_steps"),
            )

    def evaluate(self, checkpoint_path: Path) -> PerformanceMetrics:
        self._publish(EvaluationStartedEvent(checkpoint_path=checkpoint_path))
        metrics = self.get_metrics()
        if metrics is None:
            raise NotImplementedError(
                "Evaluation metrics require a telemetry adapter for the training workload"
            )
        self._publish(
            EvaluationCompletedEvent(
                checkpoint_path=checkpoint_path,
                average_reward=metrics.average_reward,
                win_rate=metrics.win_rate,
                episode_length=metrics.episode_length,
                actor_losses=metrics.actor_losses,
                critic_losses=metrics.critic_losses,
                entropy=metrics.entropy,
                explained_variance=metrics.explained_variance,
            )
        )
        return metrics

    def get_metrics(self, checkpoint_path: Path | None = None) -> PerformanceMetrics | None:
        """Return actual workload telemetry once an executor has supplied it."""
        return self._metrics

    def record_metrics(self, metrics: PerformanceMetrics) -> None:
        """Called by a concrete workload/telemetry adapter with measured data."""
        self._metrics = metrics

    @property
    def state(self) -> EngineStatus:
        return self._state

    @property
    def checkpoint_directory(self) -> Path:
        if self._checkpoint_directory is None:
            raise RuntimeError("Engine has not been initialized")
        return self._checkpoint_directory

    @property
    def global_step(self) -> int:
        return self._global_step

    @property
    def progress(self):
        return self._progress

    def _publish(self, event: EngineEvent) -> None:
        if self._publisher is not None:
            self._publisher.publish(event)

    def _require_state(self, state: EngineStatus) -> None:
        if self._state != state:
            raise RuntimeError(f"Engine must be in {state} state")

    def _require_run_id(self) -> UUID:
        if self._run_id is None:
            raise RuntimeError("Engine has not been initialized")
        return self._run_id
