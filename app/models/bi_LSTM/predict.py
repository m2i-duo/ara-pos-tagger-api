import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from typing import List, Tuple, Dict
from app.config.data_config import MODEL_PATH, WORD_INDEX_PATH, POS_TO_NUMERICAL_PATH, MAX_SEQUENCE_LENGTH

class POSPredictor:
    def __init__(self, model_path: str, word_index_path: str, pos_to_numerical_path: str, max_seq_length: int):
        self.max_seq_length = max_seq_length
        try:
            self.model = load_model(model_path)
            self.word_index: Dict[str, int] = np.load(word_index_path, allow_pickle=True).item()
            index_to_pos_dict = {v: k for k, v in np.load(pos_to_numerical_path, allow_pickle=True).item().items()}
            self.index_to_pos_tensor = tf.constant(list(index_to_pos_dict.values()), dtype=tf.string)

        except FileNotFoundError as e:
            raise FileNotFoundError(f"Error loading model or data files: {e}")

    def preprocess_input(self, sentence: str) -> np.ndarray:
        words = sentence.split()
        sequence = [self.word_index.get(word, self.word_index.get('<OOV>')) for word in words]
        return pad_sequences([sequence], maxlen=self.max_seq_length, padding='post')

    @tf.function
    def predict(self, sentence: str) -> List[str]:
        padded_sequence = self.preprocess_input(sentence)
        predictions = self.model(padded_sequence, training=False)
        predicted_indices = tf.argmax(predictions[0], axis=-1)
        predicted_tags_tensor = tf.gather(self.index_to_pos_tensor, predicted_indices)
        return [tag.decode('utf-8') for tag in predicted_tags_tensor.numpy()]


    def postprocess_output(self, sentence: str, predicted_tags: List[str]) -> List[Tuple[str, str]]:
        words = sentence.split()
        return list(zip(words, predicted_tags[:len(words)]))

def tag(sentence: str) -> List[Tuple[str, str]]:
    try:
        predictor = POSPredictor(
            model_path=MODEL_PATH,
            word_index_path=WORD_INDEX_PATH,
            pos_to_numerical_path=POS_TO_NUMERICAL_PATH,
            max_seq_length=MAX_SEQUENCE_LENGTH
        )
        predicted_tags = predictor.predict(sentence.strip())
        return predictor.postprocess_output(sentence, predicted_tags)
    except Exception as e:
        print(f"An error occurred during tagging: {e}") # Handle and log the exception.
        return [] # Return empty list in case of error.

