import tensorflow as tf
from fastapi import FastAPI
from app.routers import pos
from app.config.api_config import APIConfig

# Enable eager execution
tf.config.experimental_run_functions_eagerly(True)
app = FastAPI(
    title=APIConfig.TITLE,
    version=APIConfig.VERSION,
    description=APIConfig.DESCRIPTION
)

# Add a base URL prefix for all routes
app.include_router(pos.router, prefix="/api/v1")