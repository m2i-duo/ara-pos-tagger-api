import numpy as np
from model import BiLSTMModel
from app.config.data_config import X_TRAIN_PATH, Y_TRAIN_PATH, X_VAL_PATH, Y_VAL_PATH, WORD_INDEX_PATH, MODEL_PATH

# Load preprocessed data
X_train = np.load(X_TRAIN_PATH)
y_train = np.load(Y_TRAIN_PATH)
X_val = np.load(X_VAL_PATH)
y_val = np.load(Y_VAL_PATH)

# Load additional metadata
word_index = np.load(WORD_INDEX_PATH, allow_pickle=True).item()
vocab_size = len(word_index) + 1
max_seq_length = X_train.shape[1]

# Initialize and train the model
bilstm_model = BiLSTMModel(vocab_size=vocab_size, max_seq_length=max_seq_length)
bilstm_model.build_model()
bilstm_model.train(X_train, y_train, X_val, y_val, epochs=10, batch_size=32)

# Save the trained model
bilstm_model.save_model(MODEL_PATH)
