from typing import List

import tensorflow as tf
from fastapi import APIRouter
from app.schemas.pos import POSTagRequest, POSTagResponse
from app.services.keras_service import KerasPOSTaggerService
from app.services.pos_service import HmmPOSTagService

# Force eager execution globally
tf.compat.v1.enable_eager_execution()
print("Eager execution (global):", tf.executing_eagerly())  # Should print True

router = APIRouter()

# Initialize services
kerasService = KerasPOSTaggerService()
hmmService = HmmPOSTagService()

@router.post("/pos-tag/keras", response_model=List[POSTagResponse])
async def pos_tag_keras(request: POSTagRequest):
    return kerasService.tag(request.text)

@router.post("/pos-tag/hmm", response_model=List[POSTagResponse])
async def pos_tag_hmm(request: POSTagRequest):
    return hmmService.tag(request.text)

@router.get("/debug")
async def debug():
    return {"eager_execution": tf.executing_eagerly()}
