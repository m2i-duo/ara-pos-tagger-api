import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from app.config.data_config import (
    MODEL_PATH,
    WORD_INDEX_PATH,
    MAX_SEQUENCE_LENGTH,
    POS_TO_NUMERICAL_PATH,
)


class POSPredictor:
    def __init__(self, model_path, word_index_path, pos_to_numerical_path, max_seq_length):
        # Load the trained model
        self.model = load_model(model_path)
        self.max_seq_length = max_seq_length

        # Load word index and POS mapping
        self.word_index = np.load(word_index_path, allow_pickle=True).item()
        self.index_to_word = {v: k for k, v in self.word_index.items()}

        self.pos_to_numerical = np.load(pos_to_numerical_path, allow_pickle=True).item()
        self.numerical_to_pos = {v: k for k, v in self.pos_to_numerical.items()}

    def preprocess_input(self, sentence):
        """
        Convert a sentence into a sequence of indices and pad it.
        """
        words = sentence.split()
        sequence = [self.word_index.get(word, self.word_index.get('<OOV>')) for word in words]
        padded_sequence = pad_sequences([sequence], maxlen=self.max_seq_length, padding='post')
        return padded_sequence

    def predict(self, sentence):
        """
        Predict the POS tags for a given sentence.
        """
        padded_sequence = self.preprocess_input(sentence)
        predictions = self.model.predict(padded_sequence)

        # Convert predictions to POS tags
        predicted_tags = [self.numerical_to_pos[np.argmax(tag)] for tag in predictions[0]]
        return predicted_tags

    def postprocess_output(self, sentence, predicted_tags):
        """
        Combine the sentence and the predicted tags for visualization.
        """
        words = sentence.split()
        return list(zip(words, predicted_tags[:len(words)]))


# Example Usage
if __name__ == "__main__":
    sentence = input("Enter a sentence in Arabic: ").strip()

    # Paths from config
    predictor = POSPredictor(
        model_path=MODEL_PATH,
        word_index_path=WORD_INDEX_PATH,
        pos_to_numerical_path=POS_TO_NUMERICAL_PATH,
        max_seq_length=MAX_SEQUENCE_LENGTH
    )

    predicted_tags = predictor.predict(sentence)
    result = predictor.postprocess_output(sentence, predicted_tags)

    print("Predicted POS tags:")
    for word, tag in result:
        print(f"{word} -> {tag}")
