from __future__ import annotations

from collections import defaultdict
from typing import Protocol

from backend.contracts.engine.events.base import EngineEvent
from backend.engine_gateway.application.contracts.engine_event_publisher import (
    EngineEventPublisher,
)

class EventHandler(Protocol):
    def handle(self, event: EngineEvent) -> None: ...


class LocalEngineEventPublisher(EngineEventPublisher):
    """
    In-memory publisher used by the local engine adapter.
    """

    def __init__(self) -> None:
        self._handlers: dict[type[EngineEvent], list[EventHandler]] = defaultdict(list)

    def publish(self, event: EngineEvent) -> None:
        for handler in self._handlers.get(type(event), []):
            handler.handle(event)

    def register_handler(self, event_type: type[EngineEvent], handler: EventHandler) -> None:
        self._handlers[event_type].append(handler)
