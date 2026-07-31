


class TrainingService :
    def __init__(self, respository, validator, environment_factory, checkpoint_manager):
        
        self._repository = repository
        self._validator = validator
        self._environment_factory = environment_factory
        self._checkpoint_manager = checkpoint_manager

    def start(self, request):
        
        self.validator.validate_start(request)

        training = self._repository.create_training(request)

        training_run = self._environment_factory.create_run(training)

        training_run.start()

        return training


    def pause():
        pass

    def resume():
        pass

    def stop():
        pass