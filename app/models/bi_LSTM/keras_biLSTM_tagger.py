import tensorflow as tf
from app.config.data_config import MODEL_PATH, WORD_INDEX_PATH, POS_TO_NUMERICAL_PATH, MAX_SEQUENCE_LENGTH
from app.models.bi_LSTM.predict import POSPredictor
from app.models.bi_LSTM.predict import tag
tf.config.run_functions_eagerly(True)  # Enable eager execution globally

class KerasPOSTagger:
    def __init__(self):
        self.predictor = POSPredictor(
            model_path=MODEL_PATH,
            word_index_path=WORD_INDEX_PATH,
            pos_to_numerical_path=POS_TO_NUMERICAL_PATH,
            max_seq_length=MAX_SEQUENCE_LENGTH
        )

    def tag(self, sentence):
        result = tag(sentence)
        print(result)
        return result

if __name__ == "__main__":
    tagger = KerasPOSTagger()
    sentence = "أنا ذاهب إلى المدرسة"
    tagger.tag(sentence)

