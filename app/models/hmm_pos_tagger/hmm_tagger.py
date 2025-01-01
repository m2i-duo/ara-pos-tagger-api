from app.config.model_config import ModelConfig
from app.models.hmm_pos_tagger.hmm_model import test_model
class HMMPOSTagger:
    def __init__(self):
        # Load your HMM + Viterbi model here
        self.batch_size = ModelConfig.BATCH_SIZE
        self.epochs = ModelConfig.EPOCHS
        self.learning_rate = ModelConfig.LEARNING_RATE
        self.embedding_dim = ModelConfig.EMBEDDING_DIM
        self.hidden_units = ModelConfig.HIDDEN_UNITS
        self.dropout_rate = ModelConfig.DROPOUT_RATE
        self.recurrent_dropout_rate = ModelConfig.RECURRENT_DROPOUT_RATE
        self.optimizer = ModelConfig.OPTIMIZER
        self.loss = ModelConfig.LOSS
        self.metrics = ModelConfig.METRICS

    def tag(self, text: str):
        tagged_text = test_model(text)
        return tagged_text