


class LocalEngineClient(EngineCLient):
    def __init__(self, engine):
        self._engine = engine

    def start(self):
        self._engine.start()

    def pause(self):
        self._engine.pause()

    def resume(self):
        self._engine.resume()

    def stop(self):
        self._engine.stop()

        