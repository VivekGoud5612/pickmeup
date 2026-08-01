from unittest.mock import MagicMock, patch

from uuid import uuid4

from backend.training_service.infrastructure.repositories.sql_training_run_repository import (
    SQLTrainingRunRepository,
)

from backend.training_service.infrastructure.models.training_run_model import (
    TrainingRunModel,
)


@patch(
    "backend.training_service.infrastructure.repositories.sql_training_run_repository.TrainingRunPersistenceMapper.to_entity"
)
def test_get_training_run(mock_mapper):

    session = MagicMock()

    repo = SQLTrainingRunRepository(session)

    run_id = uuid4()

    model = MagicMock(spec=TrainingRunModel)

    entity = MagicMock()

    session.get.return_value = model

    mock_mapper.return_value = entity

    returned = repo.get_by_id(run_id)

    session.get.assert_called_once_with(
        TrainingRunModel,
        run_id,
    )

    mock_mapper.assert_called_once_with(model)

    assert returned == entity