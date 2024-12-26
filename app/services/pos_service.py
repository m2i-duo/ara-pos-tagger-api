from app.models.keras_tagger import KerasPOSTagger
from app.models.hmm_tagger import HMMPOSTagger
from app.models.custom_tagger import CustomPOSTagger
from app.utils.preprocessing import preprocess_text

class POSTagService:
    def __init__(self, model_type: str):
        if model_type == "keras":
            self.tagger = KerasPOSTagger()
        elif model_type == "hmm":
            self.tagger = HMMPOSTagger()
        elif model_type == "custom":
            self.tagger = CustomPOSTagger()
        else:
            raise ValueError("Invalid model type")

    def tag(self, text: str):
        preprocessed_text = preprocess_text(text)
        tagged_text = self.tagger.tag(preprocessed_text)
        return {"tagged_text": tagged_text}