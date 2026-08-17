"""Control-plane research intent; execution is delegated to Engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import UUID, uuid4


@dataclass(slots=True, kw_only=True)
class Experiment:
    research_question: str
    configuration_version: str
    id: UUID = field(default_factory=uuid4)
    scenario_ids: tuple[UUID, ...] = ()
    run_ids: tuple[UUID, ...] = ()
    reproducibility_metadata: dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def attach_run(self, run_id: UUID) -> None:
        if run_id not in self.run_ids:
            self.run_ids += (run_id,)
