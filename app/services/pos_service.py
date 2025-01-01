from app.models.bi_LSTM.keras_biLSTM_tagger import KerasPOSTagger
from app.models.hmm_pos_tagger.hmm_tagger import HMMPOSTagger
from app.models.custom_tagger import CustomPOSTagger
from app.utils.mapper import map_tags

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
        tagged_text = self.tagger.tag(text)
        response = map_tags(tagged_text)
        return response

# # test
# pos = POSTagService("keras")
# print(pos.tag("المنظمة العربية للتربية والثقافة والعلوم هي منظمة متخصصة"))
# pos = POSTagService("hmm")
