from __future__ import annotations 

from backend.engine_gateway.application.contracts.engine_event_publisher import EngineEventPublisher 

from backend.engine_gateway.application.events.base import EngineEvent 
from backend.engine_gateway.application.handlers.base import EngineEventHandler


class LocalEngineEventPublisher(EngineEventPublisher):
    """
    The infrastructure object of that contract. The local engine publisher
    """
    
    def __init__(self) -> None:
        self._handler : dict[type[EngineEvent], list[EngineEventHandler]] = defaultdict(list)   ### Due to the fact that there are many types of events inheriting from base... We create the key for each type like training, checkpoint, evaluation or such...


    def publish(self, event : EngineEvent) -> None:
        """
        For each handler publush the data, according to that
        In the format of events, written in events cntract
        """
        
        for handler in self._handlers[type(event)]:
            handler.handle(event)   ### This handle is taken care by the handler

        
    def register_handle(self, event_type : type[EngineEvent], handler : EngineEventHandler) -> None:
        """
        THe type becomes the key and the handler becomes the value..
        For each type of event, we have different handlers
        """
        self._handlers[event_type].append(handler)  ## So the handler handles everything for each type.