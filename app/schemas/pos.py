from pydantic import BaseModel

class POSTagRequest(BaseModel):
    text: str

class POSTagResponse(BaseModel):
    word: str
    tag: str
    arabic_tag: str
    french_tag: str
    english_tag: str