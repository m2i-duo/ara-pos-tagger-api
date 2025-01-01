from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import pos
from app.config.api_config import APIConfig
import tensorflow as tf

# Enable eager execution
tf.config.experimental_run_functions_eagerly(True)

app = FastAPI(
    title=APIConfig.TITLE,
    version=APIConfig.VERSION,
    description=APIConfig.DESCRIPTION
)

# Add CORS middleware to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Add a base URL prefix for all routes
app.include_router(pos.router, prefix="/api/v1")