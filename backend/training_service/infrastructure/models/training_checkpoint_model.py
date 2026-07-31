from __future__ import annotations

from datetime import datetime
from pathlib import Path
from uuid import UUID

from sqlalchemy import (
    Boolean,
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
    CheckpointType,
)


class TrainingCheckpointModel(Base):
    """
    Database representation of a TrainingCheckpoint.
    """

    __tablename__ = "training_checkpoints"

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    id: Mapped[UUID] = mapped_column(
        primary_key=True,
    )

    training_run_id: Mapped[UUID] = mapped_column(
        ForeignKey("training_runs.id"),
        nullable=False,
    )

    configuration_id: Mapped[UUID] = mapped_column(
        ForeignKey("training_configurations.id"),
        nullable=False,
    )

    # ------------------------------------------------------------------
    # Progress
    # ------------------------------------------------------------------

    episode: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    step: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    # ------------------------------------------------------------------
    # Checkpoint Metadata
    # ------------------------------------------------------------------

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