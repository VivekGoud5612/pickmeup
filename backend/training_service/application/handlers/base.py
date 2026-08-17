from __future__ import annotations

from abc import ABC, abstractmethod

from backend.contracts.engine.events.base import EngineEvent


class EngineEventHandler(ABC):
    @abstractmethod
    def handle(self, event: EngineEvent) -> None:
        raise NotImplementedError
