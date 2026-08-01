from __future__ import annotations 
from abc import ABC, abstractmethod 

from backend.engine_gateway.application.events.base import EngineEvent 


class EngineEventHandler(ABC):
    """
    A contract for engine handlers, Just one single function
    Taking in events and writing that in infra layer
    """

    @abstractmethod
    def handle(self, event : EngineEvent) -> None:
        ...  ### Not a contract but a skeleton for now

        