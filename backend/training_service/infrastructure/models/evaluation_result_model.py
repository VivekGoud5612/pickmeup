from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import (
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    JSON,
    Text,
)

from sqlalchemy.orm import (
    Mapped,
    mapped_column,
)

from backend.training_service.infrastructure.database.base import Base

from backend.training_service.domain.enums import (
    EvaluationStatus,
)


class EvaluationResultModel(Base):
    """
    Database representation of an evaluation result.
    """

    __tablename__ = "evaluation_results"

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    id: Mapped[UUID] = mapped_column(
        primary_key=True,
    )

    checkpoint_id: Mapped[UUID] = mapped_column(
        ForeignKey("training_checkpoints.id"),
        nullable=False,
    )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    status: Mapped[EvaluationStatus] = mapped_column(
        Enum(EvaluationStatus),
        nullable=False,
    )

    num_episodes: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    notes: Mapped[str] = mapped_column(
        Text,
        default="",
        nullable=False,
    )

    # ------------------------------------------------------------------
    # Performance Metrics
    # ------------------------------------------------------------------

    average_reward: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    actor_losses: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
    )

    critic_losses: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
    )

    entropy: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    explained_variance: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    win_rate: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    episode_length: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    # ------------------------------------------------------------------
    # Timestamps
    # ------------------------------------------------------------------

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )