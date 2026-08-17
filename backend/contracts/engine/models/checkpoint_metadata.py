from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path


@dataclass(slots=True, frozen=True, kw_only=True)
class CheckpointMetadata:
    """
    Metadata for a saved checkpoint.
    """

    path: Path
    size_bytes: int
    episodes: int
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


# Compatibility for existing callers while the corrected contract name is adopted.
CheckpointMetaData = CheckpointMetadata
