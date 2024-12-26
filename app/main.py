from fastapi import FastAPI
from app.routers import pos
from app.config.api_config import APIConfig

app = FastAPI(
    title=APIConfig.TITLE,
    version=APIConfig.VERSION,
    description=APIConfig.DESCRIPTION
)

app.include_router(pos.router)