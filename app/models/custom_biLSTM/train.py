import os.path
import numpy as np
from model import BiLSTM
from app.config.data_config import X_TRAIN_PATH, Y_TRAIN_PATH, X_VAL_PATH, Y_VAL_PATH, CHECKPOINTS_PATH

# Load preprocessed data
X_train = np.load(X_TRAIN_PATH)
y_train = np.load(Y_TRAIN_PATH)
X_val = np.load(X_VAL_PATH)
y_val = np.load(Y_VAL_PATH)

input_dim = 50
hidden_dim = 128
output_dim = 100
batch_size = 32  # Define the batch size

bi_lstm = BiLSTM(input_dim, hidden_dim, output_dim, batch_size)
checkpoint_path = os.path.join(CHECKPOINTS_PATH, "bi_lstm_checkpoint.pkl")

bi_lstm.train(X_train, y_train, X_val, y_val, epochs=10, learning_rate=0.001, checkpoint_path=checkpoint_path)