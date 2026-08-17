"""Control-plane description of conditions under which a run executes."""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID, uuid4


@dataclass(slots=True, frozen=True, kw_only=True)
class Scenario:
    name: str
    configuration: dict[str, object]
    id: UUID = field(default_factory=uuid4)
    random_seed: int | None = None
