from pydantic import BaseModel

class POSTagRequest(BaseModel):
    text: str

class POSTagResponse(BaseModel):
    tagged_text: str