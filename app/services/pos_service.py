import tensorflow as tf
from app.models.hmm_pos_tagger.hmm_tagger import HMMPOSTagger
from app.utils.mapper import map_tags

tf.config.run_functions_eagerly(True)  # Enable eager execution globally

class HmmPOSTagService:
    def __init__(self):
        self.tagger = HMMPOSTagger()

    def tag(self, text: str):
        tagged_text = self.tagger.tag(text)
        response = map_tags(tagged_text)
        return response

if __name__ == "__main__":
    service = HmmPOSTagService()
    sentence = "أنا ذاهب إلى المدرسة"
    print(service.tag(sentence))
