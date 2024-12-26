from app.config.model_config import ModelConfig

class KerasPOSTagger:
    def __init__(self):
        # Load your Keras model here
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
        # Implement the tagging logic using Keras model here
        return "tagged text with Keras"