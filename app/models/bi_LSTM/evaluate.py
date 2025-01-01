import numpy as np
from model import BiLSTMModel
from app.config.data_config import MODEL_PATH, X_TEST_PATH, Y_TEST_PATH
# Load test data
X_test = np.load(X_TEST_PATH)
y_test = np.load(Y_TEST_PATH)

# Initialize and load the trained model
bilstm_model = BiLSTMModel(vocab_size=None, max_seq_length=None)  # Placeholder; loading the model doesn't require vocab_size/max_seq_length
bilstm_model.load_model(MODEL_PATH)

# Evaluate the model
bilstm_model.evaluate(X_test, y_test)
