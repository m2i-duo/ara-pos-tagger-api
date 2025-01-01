from app.models.bi_LSTM.keras_biLSTM_tagger import KerasPOSTagger
from app.utils.mapper import map_tags
import tensorflow as tf
tf.config.run_functions_eagerly(True)

class KerasPOSTaggerService:
    def __init__(self):
        self.tagger = KerasPOSTagger()
        print("loading keras model")

    def tag(self, text: str):
        tagged_text = self.tagger.tag(text)
        response = map_tags(tagged_text)
        print(response)
        return response

def tag(text: str):
    serv = KerasPOSTaggerService()
    tagged_text = serv.tag(text)
    return tagged_text

if __name__ == "__main__":
    service = KerasPOSTaggerService()
    sentence = "أنا ذاهب إلى المدرسة"
    print(service.tag(sentence))