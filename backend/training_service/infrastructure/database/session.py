from sqlalchemy.orm import sessionmaker

from backend.training_service.infrastructure.database.engine import engine

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)