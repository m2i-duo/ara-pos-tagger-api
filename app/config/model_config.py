class ModelConfig:
    BATCH_SIZE = 32
    EPOCHS = 20
    LEARNING_RATE = 0.001
    EMBEDDING_DIM = 128
    HIDDEN_UNITS = 256
    DROPOUT_RATE = 0.5
    RECURRENT_DROPOUT_RATE = 0.5
    OPTIMIZER = 'adam'
    LOSS = 'categorical_crossentropy'
    METRICS = ['accuracy']
    # Add other model-specific configurations here
