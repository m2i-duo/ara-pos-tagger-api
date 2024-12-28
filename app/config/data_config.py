import os

# Get the absolute path to the directory containing the config file
CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the project root as the parent of the config directory
PROJECT_ROOT = os.path.dirname(CONFIG_DIR)

# Define directories for data and models relative to the project root
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(MODEL_DIR,"data", "processed")

# Maximum sequence length for padding
MAX_SEQUENCE_LENGTH = 100

# Test size for data splitting (20% test data)
TEST_SIZE = 0.2

# Validation size for data splitting (10% of the training set)
VALIDATION_SIZE = 0.1

# Define file paths
X_TRAIN_PATH = os.path.join(DATA_DIR, "X_train.npy")
Y_TRAIN_PATH = os.path.join(DATA_DIR, "y_train.npy")
X_VAL_PATH = os.path.join(DATA_DIR, "X_val.npy")
Y_VAL_PATH = os.path.join(DATA_DIR, "y_val.npy")
X_TEST_PATH = os.path.join(DATA_DIR, "X_test.npy")
Y_TEST_PATH = os.path.join(DATA_DIR, "y_test.npy")
WORD_INDEX_PATH = os.path.join(DATA_DIR, "word_index.npy")
POS_TO_NUMERICAL_PATH = os.path.join(DATA_DIR, "pos_to_numerical.npy")

MODEL_PATH = os.path.join(MODEL_DIR + "/saved_models", "pos_tagger_bilstm.keras")
DATASET_PATH = os.path.join(MODEL_DIR, "data", "raw", "arabic_pos_data.txt")
CHECKPOINTS_PATH = os.path.join(MODEL_DIR, "saved_models", "checkpoints")