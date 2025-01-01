from app.config.data_config import MODEL_PATH, WORD_INDEX_PATH, POS_TO_NUMERICAL_PATH, MAX_SEQUENCE_LENGTH
from app.models.bi_LSTM.predict import POSPredictor
class KerasPOSTagger:
    def __init__(self):
        self.predictor = POSPredictor(
            model_path=MODEL_PATH,
            word_index_path=WORD_INDEX_PATH,
            pos_to_numerical_path=POS_TO_NUMERICAL_PATH,
            max_seq_length=MAX_SEQUENCE_LENGTH
        )

    def tag(self, sentence):
        predicted_tags = self.predictor.predict(sentence)
        return self.predictor.postprocess_output(sentence, predicted_tags)
