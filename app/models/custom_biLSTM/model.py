import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from transformers import TFBertModel, BertTokenizer
from tensorflow.keras.optimizers import Adam

# Configuration
MAX_SEQUENCE_LENGTH = 100
BERT_MODEL_NAME = "bert-base-multilingual-cased"  # Change to a specific model if needed
LEARNING_RATE = 5e-5

# Load preprocessed data
X_train = np.load('/kaggle/working/proc_data/X_train.npy')
X_val = np.load('/kaggle/working/proc_data/X_val.npy')
X_test = np.load('/kaggle/working/proc_data/X_test.npy')
y_train = np.load('/kaggle/working/proc_data/y_train.npy')
y_val = np.load('/kaggle/working/proc_data/y_val.npy')
y_test = np.load('/kaggle/working/proc_data/y_test.npy')
unique_tags = np.load('/kaggle/working/proc_data/labels.npy', allow_pickle=True)

# Number of labels
num_labels = len(unique_tags)

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

# Define the model
def build_bert_pos_model():
    # Input layers
    input_ids = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name="input_ids")
    attention_mask = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name="attention_mask")

    # BERT model
    bert_model = TFBertModel.from_pretrained(BERT_MODEL_NAME)

    # Pass inputs to BERT
    bert_output = bert_model([input_ids, attention_mask], training=False)[0]  # Sequence output

    # Add a dropout and dense layer
    dropout = Dropout(0.1)(bert_output)
    logits = Dense(num_labels, activation="softmax")(dropout)

    # Build and compile the model
    model = Model(inputs=[input_ids, attention_mask], outputs=logits)
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

# Prepare data for BERT
def prepare_data_for_bert(padded_sequences, padded_labels):
    attention_masks = np.where(padded_sequences != 0, 1, 0)  # Mask non-padding tokens
    return padded_sequences, attention_masks, padded_labels

# Prepare training and validation data
X_train_ids, X_train_masks, y_train = prepare_data_for_bert(X_train, y_train)
X_val_ids, X_val_masks, y_val = prepare_data_for_bert(X_val, y_val)

# Build the model
model = build_bert_pos_model()

# Train the model
BATCH_SIZE = 32
EPOCHS = 3

model.fit(
    {"input_ids": X_train_ids, "attention_mask": X_train_masks},
    y_train,
    validation_data=({"input_ids": X_val_ids, "attention_mask": X_val_masks}, y_val),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS
)

# Evaluate the model
X_test_ids, X_test_masks, y_test = prepare_data_for_bert(X_test, y_test)
loss, accuracy = model.evaluate({"input_ids": X_test_ids, "attention_mask": X_test_masks}, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")
