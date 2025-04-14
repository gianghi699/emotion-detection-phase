import pandas as pd
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Emotion mapping
emotion_map = {
    0: "Angry",
    3: "Happy",
    4: "Sad",
    6: "Neutral"
}
emotion_labels = list(emotion_map.values())
label_to_index = {v: i for i, v in enumerate(emotion_labels)}

# Load dataset
df = pd.read_csv("fer2013.csv")

# Filter only the 4 selected emotions
df = df[df['emotion'].isin(emotion_map.keys())]

# Map to new labels (0 to 3)
df['emotion'] = df['emotion'].map(emotion_map)
df['emotion'] = df['emotion'].map(label_to_index)

# Process images
pixels = df['pixels'].tolist()
faces = []
for pixel_sequence in pixels:
    face = np.array([int(p) for p in pixel_sequence.split()], dtype=np.uint8).reshape(48, 48)
    faces.append(face)

faces = np.array(faces)
faces = np.expand_dims(faces, -1)  # Add channel dimension
faces = faces / 255.0              # Normalize pixel values

# One-hot encode labels
labels = to_categorical(df['emotion'], num_classes=4)

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(faces, labels, test_size=0.2, random_state=42)

# Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=30, batch_size=64, validation_data=(X_val, y_val))

# Save model
model.save("emotion_model_4classes.h5")
print("Model saved as emotion_model_4classes.h5")

# Optional: Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.title('Loss')

plt.tight_layout()
plt.show()
