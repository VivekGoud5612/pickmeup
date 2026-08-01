from unittest.mock import MagicMock, patch

from backend.training_service.infrastructure.repositories.sql_training_run_repository import (
    SQLTrainingRunRepository,
)


@patch(
    "backend.training_service.infrastructure.repositories.sql_training_run_repository.TrainingRunPersistenceMapper.to_model"
)
def test_save_training_run(mock_mapper):

    session = MagicMock()

    repo = SQLTrainingRunRepository(session)

    entity = MagicMock()

    model = MagicMock()

    mock_mapper.return_value = model

    repo.save(entity)

    mock_mapper.assert_called_once_with(entity)

    session.add.assert_called_once_with(model)

    session.commit.assert_called_once()