from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)

from sqlalchemy.orm import (
    Mapped,
    mapped_column,
)

from backend.training_service.infrastructure.database.base import Base

from backend.training_service.domain.enums import (
    CheckpointType,
)


class TrainingCheckpointModel(Base):

    __tablename__ = "training_checkpoints"

    # ------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------

    id: Mapped[UUID] = mapped_column(primary_key=True)

    training_run_id: Mapped[UUID] = mapped_column(
        ForeignKey("training_runs.id"),
        nullable=False,
    )

    # ------------------------------------------------------------
    # Progress
    # ------------------------------------------------------------

    step: Mapped[int] = mapped_column(Integer, nullable=False)

    # ------------------------------------------------------------
    # Hyperparameters Snapshot
    # ------------------------------------------------------------

    learning_rate: Mapped[float] = mapped_column(Float, nullable=False)

    total_timesteps: Mapped[int] = mapped_column(Integer, nullable=False)

    num_envs: Mapped[int] = mapped_column(Integer, nullable=False)

    gamma: Mapped[float] = mapped_column(Float, nullable=False)

    gae_lambda: Mapped[float] = mapped_column(Float, nullable=False)

    clip_range: Mapped[float] = mapped_column(Float, nullable=False)

    entropy_coeff: Mapped[float] = mapped_column(Float, nullable=False)

    batch_size: Mapped[int] = mapped_column(Integer, nullable=False)

    rollout_length: Mapped[int] = mapped_column(Integer, nullable=False)

    ppo_epochs: Mapped[int] = mapped_column(Integer, nullable=False)

    max_grad_norm: Mapped[float] = mapped_column(Float, nullable=False)

    checkpoint_save_interval: Mapped[int] = mapped_column(Integer, nullable=False)

    # ------------------------------------------------------------
    # Reward Weights Snapshot
    # ------------------------------------------------------------

    dealer_damage: Mapped[float] = mapped_column(Float, nullable=False)

    tank_damage: Mapped[float] = mapped_column(Float, nullable=False)

    healer_healing: Mapped[float] = mapped_column(Float, nullable=False)

    boss_damage: Mapped[float] = mapped_column(Float, nullable=False)

    tank_block: Mapped[float] = mapped_column(Float, nullable=False)

    healer_damage: Mapped[float] = mapped_column(Float, nullable=False)

    boss_healing: Mapped[float] = mapped_column(Float, nullable=False)

    death_penalty: Mapped[float] = mapped_column(Float, nullable=False)

    step_penalty: Mapped[float] = mapped_column(Float, nullable=False)

    victory_bonus: Mapped[float] = mapped_column(Float, nullable=False)

    invalid_action_penalty: Mapped[float] = mapped_column(Float, nullable=False)

    # ------------------------------------------------------------
    # Curriculum Snapshot
    # ------------------------------------------------------------

    boss_hp: Mapped[int] = mapped_column(Integer, nullable=False)

    spawn_radius: Mapped[int] = mapped_column(Integer, nullable=False)

    max_steps: Mapped[int] = mapped_column(Integer, nullable=False)

    difficulty_level: Mapped[int] = mapped_column(Integer, nullable=False)

    reward_scale: Mapped[float] = mapped_column(Float, nullable=False)

    grid_size: Mapped[int] = mapped_column(Integer, nullable=False)

    # ------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------

    checkpoint_type: Mapped[CheckpointType] = mapped_column(
        Enum(CheckpointType),
        nullable=False,
    )

    file_path: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
    )

    is_best: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    description: Mapped[str] = mapped_column(
        Text,
        default="",
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )