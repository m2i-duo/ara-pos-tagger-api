# Step 2: Import Necessary Libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Lambda
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from transformers import TFBertForTokenClassification, BertTokenizer
from app.config.data_config import X_TEST_PATH, Y_TEST_PATH, X_VAL_PATH, Y_VAL_PATH, X_TRAIN_PATH, Y_TRAIN_PATH, WORD_INDEX_PATH,MODEL_DIR

BERT_MODEL_NAME = "bert-base-multilingual-cased"

# Step 3: Define Configurations
MAX_SEQUENCE_LENGTH = 100
BATCH_SIZE = 32
EPOCHS = 10
OUTPUT_DIR = MODEL_DIR

X_TRAIN_PATH = os.path.join(X_TRAIN_PATH)
Y_TRAIN_PATH = os.path.join(Y_TRAIN_PATH)
X_VAL_PATH = os.path.join(X_VAL_PATH)
Y_VAL_PATH = os.path.join(Y_VAL_PATH)
WORD_INDEX_PATH = os.path.join(WORD_INDEX_PATH)
MODEL_PATH = os.path.join(OUTPUT_DIR, 'bert_pos_tagger.keras')

# Step 4: Check GPU Availability
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
if tf.config.list_physical_devices('GPU'):
    print(tf.config.list_physical_devices('GPU'))
else:
    print("GPU not available. Check if it is enabled in notebook settings.")


# Step 5: Load Data
print("Loading data...")
X_train = np.load(X_TRAIN_PATH)
y_train = np.load(Y_TRAIN_PATH)
X_val = np.load(X_VAL_PATH)
y_val = np.load(Y_VAL_PATH)
word_index = np.load(WORD_INDEX_PATH, allow_pickle=True).item()
vocab_size = len(word_index) + 1

print(f"Vocabulary size: {vocab_size}")
print(f"Training data shape: {X_train.shape}")
print(f"Validation data shape: {X_val.shape}")

# Step 6: Load BERT Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def encode_data(sentences, labels, tokenizer, max_seq_length):
    # Check if sentences are already tokenized
    if isinstance(sentences[0], str):  # If the input is raw text
        input_ids, attention_masks, label_ids = [], [], []
        for sentence, label in zip(sentences, labels):
            # Tokenize the sentences
            tokenized = tokenizer(
                sentence,
                truncation=True,
                padding='max_length',
                max_length=max_seq_length,
                return_tensors='tf'
            )
            input_ids.append(tokenized['input_ids'][0].numpy())
            attention_masks.append(tokenized['attention_mask'][0].numpy())
            label_ids.append(label)
        return np.array(input_ids), np.array(attention_masks), np.array(label_ids)
    else:  # If sentences are already tokenized (indices)
        input_ids, attention_masks, label_ids = [], [], []
        for sentence, label in zip(sentences, labels):
            input_ids.append(sentence)  # Assume sentences are already tokenized
            attention_masks.append([1] * len(sentence))  # Create attention mask
            label_ids.append(label)
        return np.array(input_ids), np.array(attention_masks), np.array(label_ids)


# Define the model
# Define the model
def build_bert_model(max_seq_length):
    # Load the BERT model
    bert_model = TFBertForTokenClassification.from_pretrained(BERT_MODEL_NAME, num_labels=2)  # Assuming binary classification

    # Define the input layers
    input_ids = Input(shape=(max_seq_length,), dtype=tf.int32, name="input_ids")
    attention_mask = Input(shape=(max_seq_length,), dtype=tf.int32, name="attention_mask")

    # Pass the inputs to BERT model
    bert_output = bert_model.bert(input_ids, attention_mask=attention_mask)[0]

    # Apply Dropout
    x = Dropout(0.3)(bert_output)

    # Apply a Dense layer for classification
    x = Dense(2, activation='softmax')(x)

    # Define the model
    model = Model(inputs=[input_ids, attention_mask], outputs=x)

    return model


# Build and Compile the Model
model = build_bert_model(MAX_SEQUENCE_LENGTH)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5), loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# Step 9: Add Callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    filepath=MODEL_PATH,
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

# Step 10: Train the Model
print("Training model...")
history = model.fit(
    x={'input_ids': X_train, 'attention_mask': train_attention_masks},
    y=y_train,
    validation_data=(
        {'input_ids': X_val, 'attention_mask': val_attention_masks}, y_val
    ),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stopping, checkpoint]
)

# Step 11: Save the Final Model
model.save(MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

# Step 12: Visualize Training Performance
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'training_curves.png'))
plt.show()

# Step 13: Save Training History
import json

history_path = os.path.join(OUTPUT_DIR, 'training_history.json')
with open(history_path, 'w') as f:
    json.dump(history.history, f)
print(f"Training history saved to {history_path}")
