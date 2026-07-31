from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import (
    DateTime,
    Enum,
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
    TrainingAlgorithm,
    TrainingStatus,
)


class TrainingRunModel(Base):
    """
    Database representation of a TrainingRun entity.
    """

    __tablename__ = "training_runs"

    id: Mapped[UUID] = mapped_column(
        primary_key=True,
    )

    name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    algorithm: Mapped[TrainingAlgorithm] = mapped_column(
        Enum(TrainingAlgorithm),
        nullable=False,
    )

    status: Mapped[TrainingStatus] = mapped_column(
        Enum(TrainingStatus),
        nullable=False,
    )

    # Flattened TrainingProgress value object
    episode: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    step: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    configuration_id: Mapped[UUID] = mapped_column(
        ForeignKey("training_configurations.id"),
        nullable=False,
    )

    latest_checkpoint_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("training_checkpoints.id"),
        nullable=True,
    )

    best_checkpoint_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("training_checkpoints.id"),
        nullable=True,
    )

    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    notes: Mapped[str] = mapped_column(
        Text,
        default="",
        nullable=False,
    )