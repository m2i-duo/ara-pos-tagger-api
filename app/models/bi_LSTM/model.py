import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional


class BiLSTMModel:
    def __init__(self, vocab_size, max_seq_length, embedding_dim=100, lstm_units=100):
        """
        Initialize the Bi-LSTM model with the given parameters.

        Args:
            vocab_size (int): Size of the vocabulary.
            max_seq_length (int): Maximum length of input sequences.
            embedding_dim (int): Dimension of word embeddings. Default is 100.
            lstm_units (int): Number of units in the LSTM layer. Default is 100.
        """
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.model = None

    def build_model(self):
        """
        Build and compile the Bi-LSTM model.
        """
        self.model = Sequential()
        self.model.add(Embedding(input_dim=self.vocab_size,
                                 output_dim=self.embedding_dim,
                                 input_length=self.max_seq_length))
        self.model.add(Bidirectional(LSTM(units=self.lstm_units, return_sequences=True)))
        self.model.add(Dense(units=self.vocab_size, activation='softmax'))
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])
        print("Bi-LSTM Model successfully built!")

    def train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
        """
        Train the Bi-LSTM model.

        Args:
            X_train (numpy.ndarray): Training input data.
            y_train (numpy.ndarray): Training target data.
            X_val (numpy.ndarray): Validation input data.
            y_val (numpy.ndarray): Validation target data.
            epochs (int): Number of training epochs. Default is 10.
            batch_size (int): Batch size for training. Default is 32.
        """
        if self.model is None:
            raise ValueError("Model has not been built. Call 'build_model()' first.")

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size
        )
        print("Model training completed!")
        return history

    def evaluate(self, X_test, y_test):
        """
        Evaluate the trained Bi-LSTM model.

        Args:
            X_test (numpy.ndarray): Test input data.
            y_test (numpy.ndarray): Test target data.
        """
        if self.model is None:
            raise ValueError("Model has not been built. Call 'build_model()' first.")

        test_loss, test_accuracy = self.model.evaluate(X_test, y_test)
        print(f"Test Loss: {test_loss}")
        print(f"Test Accuracy: {test_accuracy}")
        return test_loss, test_accuracy

    def save_model(self, model_path):
        """
        Save the trained model to a file.

        Args:
            model_path (str): Path to save the model.
        """
        if self.model is None:
            raise ValueError("Model has not been built. Call 'build_model()' first.")

        self.model.save(model_path)
        print(f"Model saved to {model_path}!")

    def load_model(self, model_path):
        """
        Load a pre-trained model from a file.

        Args:
            model_path (str): Path to the saved model.
        """
        from tensorflow.keras.models import load_model
        self.model = load_model(model_path)
        print(f"Model loaded from {model_path}!")
