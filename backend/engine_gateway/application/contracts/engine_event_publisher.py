from __future__ import annotations 
from abc import abstractmethod, ABC 

from backend.engine_gateway.application.events.base import EngineEvent


class EngineEventPublisher(ABC):    
    """
    Contract responsible for publishing engine events.

    Concrete implementations may publish events in-memory,
    through Redis, Kafka, RabbitMQ, etc.
    """
    @abstractmethod 
    def publish(self, event : EngineEvent) -> None:
        raise NotImplementedError()
        