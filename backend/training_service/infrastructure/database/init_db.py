from backend.training_service.infrastructure.database.base import Base
from backend.training_service.infrastructure.database.engine import engine

# Registers all models
import backend.training_service.infrastructure.models


def init_database() -> None:
    Base.metadata.create_all(bind=engine)