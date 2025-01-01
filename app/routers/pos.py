from typing import List

from fastapi import APIRouter, Query
from app.schemas.pos import POSTagRequest, POSTagResponse
from app.services.pos_service import POSTagService

router = APIRouter()

@router.post("/pos-tag/keras", response_model=List[POSTagResponse])
async def pos_tag_keras(request: POSTagRequest):
    service = POSTagService("keras")
    return service.tag(request.text)

@router.post("/pos-tag/hmm", response_model=List[POSTagResponse])
async def pos_tag_hmm(request: POSTagRequest):
    service = POSTagService("hmm")
    return service.tag(request.text)

@router.post("/pos-tag/custom", response_model=List[POSTagResponse])
async def pos_tag_custom(request: POSTagRequest):
    service = POSTagService("custom")
    return service.tag(request.text)

@router.post("/pos-tag/performance", response_model=POSTagResponse)
async def pos_tag_performance(request: POSTagRequest, model_type: str = Query("keras")):
    service = POSTagService(model_type)
    # Implement performance check logic here
    return {"performance": "performance metrics"}