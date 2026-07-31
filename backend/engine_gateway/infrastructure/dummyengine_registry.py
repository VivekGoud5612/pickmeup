from backend.engine_gateway.infrastructure.clients.dummy_engine_client import DummyEngineClient


class DummyEngineRegistry:

    def __init__(self):
        self._client = DummyEngineClient()

    def create(self, *args, **kwargs):
        return self._client

    def get(self, *args, **kwargs):
        return self._client

    def remove(self, *args, **kwargs):
        pass

    def exists(self, *args, **kwargs):
        return True

    def running_training_runs(self):
        return []