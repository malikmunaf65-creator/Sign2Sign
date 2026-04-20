# improved_train_model.py

import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATA_DIR = 'gesture_data'

# Load gestures
gestures = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
print(f"Training on gestures: {gestures}")

features = []
labels = []

# Load data
for gesture in gestures:
    gesture_dir = os.path.join(DATA_DIR, gesture)
    for file in os.listdir(gesture_dir):
        if file.endswith('.npy'):
            data = np.load(os.path.join(gesture_dir, file))
            
            if data.shape == (64, 64, 3):
                features.append(data / 255.0)
                labels.append(gesture)
            else:
                print(f"Skipping {file} with shape {data.shape}")

if not features:
    print("❌ No valid training data found!")
    exit()

X = np.array(features)
y = np.array(labels)

print(f"✅ Loaded {len(X)} samples")

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded
)

# Data augmentation (IMPROVED)
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True
)

# Improved CNN model (stronger)
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Flatten(),

    Dense(256, activation='relu'),
    Dropout(0.5),

    Dense(128, activation='relu'),
    Dropout(0.3),

    Dense(len(gestures), activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.0005),  # slower learning = better accuracy
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ONLY learning rate reduction (NO early stopping)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=5,
    min_lr=0.00001
)

print("🧠 Training model for FULL 100 epochs...")

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32, shuffle=True),
    epochs=100,
    validation_data=(X_test, y_test),
    callbacks=[reduce_lr],
    verbose=1
)

# Evaluate model
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Final Test Accuracy: {test_accuracy * 100:.2f}%")

# Save model and labels
model.save('gesture_model.keras')
np.save('label_map.npy', le.classes_)

print("✅ Model trained and saved successfully!")