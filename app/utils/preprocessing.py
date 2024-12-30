import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow.compat.v1 as tf

tf.compat.v1.disable_v2_behavior()

# Configuration imports
from app.config.data_config import MAX_SEQUENCE_LENGTH, TEST_SIZE, VALIDATION_SIZE, DATASET_PATH, X_TEST_PATH, X_TRAIN_PATH, X_VAL_PATH, Y_TEST_PATH, Y_TRAIN_PATH, Y_VAL_PATH, PROC_DATA_DIR

input_file_path = DATASET_PATH # Path to the input file

def extract_data_from_txt(txt_file):
    with open(txt_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    sentences = []
    pos_tags = []

    current_sentence = []
    current_pos_tags = []

    for line in lines:
        if line.startswith("III"):
            sentence = ' '.join(current_sentence)
            sentences.append(sentence.strip())

            pos_tags.append(current_pos_tags)

            current_sentence = []
            current_pos_tags = []
        else:
            parts = line.split('\t')
            if len(parts) < 2:
                continue  # Skip lines that do not have at least two parts
            word = parts[0]
            current_sentence.append(word)

            pos = parts[1].strip()
            current_pos_tags.append(pos)

    return sentences, pos_tags

def tokenize_text(sentences):
    tokenizer = Tokenizer(oov_token='<OOV>')
    tokenizer.fit_on_texts(sentences)
    return tokenizer

def map_pos_to_numerical(pos_tags):
    pos_to_numerical = {}
    numerical_pos_tags = []

    for tags in pos_tags:
        for tag in tags:
            if tag not in pos_to_numerical:
                pos_to_numerical[tag] = len(pos_to_numerical)

    numerical_pos_tags = [[pos_to_numerical[tag] for tag in tags] for tags in pos_tags]

    return pos_to_numerical, numerical_pos_tags

def preprocess_data(input_file_path):
    sentences, pos_tags = extract_data_from_txt(input_file_path)

    tokenizer = tokenize_text(sentences)
    word_index = tokenizer.word_index
    vocab_size = len(word_index) + 1  # Add 1 for padding token
    print("------------ vocab_size", vocab_size)
    pos_to_numerical, numerical_pos_tags = map_pos_to_numerical(pos_tags)

    sequences = tokenizer.texts_to_sequences(sentences)

    # Pad sequences to ensure uniform input size
    padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

    # Pad POS tags to match the sequences
    padded_pos_tags = pad_sequences(numerical_pos_tags, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

    # Split data into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(padded_sequences, padded_pos_tags, test_size=TEST_SIZE, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=VALIDATION_SIZE / (1 - TEST_SIZE), random_state=42)

    # Créer les répertoires pour les fichiers de sortie
    os.makedirs(os.path.dirname(X_TRAIN_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(Y_TRAIN_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(X_VAL_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(Y_VAL_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(X_TEST_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(Y_TEST_PATH), exist_ok=True)

    # Save the processed data
    np.save(X_TRAIN_PATH, X_train)
    np.save(Y_TRAIN_PATH, y_train)
    np.save(X_VAL_PATH, X_val)
    np.save(Y_VAL_PATH, y_val)
    np.save(X_TEST_PATH, X_test)
    np.save(Y_TEST_PATH, y_test)
    np.save(os.path.join(PROC_DATA_DIR, "word_index.npy"), word_index)
    np.save(os.path.join(PROC_DATA_DIR, "pos_to_numerical.npy"), pos_to_numerical)

    print("Preprocessing complete!")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Sample sentences: {sentences[:4]}")
    print(f"Sample POS tags: {pos_tags[:4]}")

# Example usage
if __name__ == "__main__":
    # preprocess_data(input_file_path)
    sentences, pos_tags = extract_data_from_txt(input_file_path)
    unique_tags_set = set([tag for tags in pos_tags for tag in tags])
    print(" tags size ", len(unique_tags_set))
    print(unique_tags_set)
