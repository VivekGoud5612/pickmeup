from __future__ import annotations

from fastapi import FastAPI

from backend.training_service.infrastructure.database.init_db import (
    init_database,
)

from backend.training_service.app.routes.training_router import (
    router as training_router,
)

from backend.training_service.app.routes.checkpoint_router import (
    router as checkpoint_router,
)

from backend.training_service.app.routes.replay_router import (
    router as replay_router,
)

# Future
# from backend.training_service.app.routes.configuration_router import (
#     router as configuration_router,
# )


app = FastAPI(
    title="Training Service",
    version="1.0.0",
    description="Training Service for Robotics Research Platform",
)


@app.on_event("startup")
def startup() -> None:
    """
    Initialize database schema.

    NOTE:
    Development only.
    Replace with Alembic migrations in production.
    """
    init_database()



#Routers
app.include_router(training_router)
app.include_router(checkpoint_router)
app.include_router(replay_router)
# app.include_router(configuration_router)


@app.get("/")
def root():
    return {
        "message": "PickMeUp Training Service is running",
    }