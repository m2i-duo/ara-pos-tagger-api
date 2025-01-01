from fastapi import FastAPI
from app.routers import pos
from app.config.api_config import APIConfig

app = FastAPI(
    title=APIConfig.TITLE,
    version=APIConfig.VERSION,
    description=APIConfig.DESCRIPTION
)


# Add a base URL prefix for all routes
app.include_router(pos.router, prefix="/api/v1")