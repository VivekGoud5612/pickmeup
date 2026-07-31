class DummyEngineClient:

    def start(self, *args, **kwargs):
        pass

    def pause(self, *args, **kwargs):
        pass

    def resume(self, *args, **kwargs):
        pass

    def stop(self, *args, **kwargs):
        pass

    def save_checkpoint(self, *args, **kwargs):
        return None

    def load_checkpoint(self, *args, **kwargs):
        pass

    def delete_checkpoint(self, *args, **kwargs):
        pass

    def evaluate_checkpoint(self, *args, **kwargs):
        return None

    def get_metrics(self, *args, **kwargs):
        return None