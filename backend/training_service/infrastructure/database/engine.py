from sqlalchemy import create_engine

DATABASE_URL = (
    "postgresql+psycopg://postgres:password@postgres:5432/training_db"
)

engine = create_engine(
    DATABASE_URL,
    echo=False,
    future=True,
    pool_pre_ping=True,
)