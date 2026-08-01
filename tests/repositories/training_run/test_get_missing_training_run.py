import pytest

from unittest.mock import MagicMock

from uuid import uuid4

from backend.training_service.infrastructure.repositories.sql_training_run_repository import (
    SQLTrainingRunRepository,
)

def test_get_missing_training_run():

    session = MagicMock()

    session.get.return_value = None

    repo = SQLTrainingRunRepository(session)

    with pytest.raises(ValueError):
        repo.get_by_id(uuid4())