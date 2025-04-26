# cnn_accident_detection.py

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from keras.utils import plot_model
from pathlib import Path
from tensorflow.keras.utils import plot_model


# ========== CONFIGURATION ==========
# Set project base path
base_path = Path(__file__).parent

# Input directories
input_data_dir = base_path / "data" / "input"
train_dir = input_data_dir / "train"
test_dir = input_data_dir / "test"
val_dir = input_data_dir / "val"

# Output directories for saving frames (if needed later)
output_data_dir = base_path / "data" / "output"
accident_frames_dir = output_data_dir / "accident_frames"
non_accident_frames_dir = output_data_dir / "non_accident_frames"

# Model directory
model_dir = base_path / "model"
model_dir.mkdir(exist_ok=True)

# Basic parameters
batch_size = 100
img_height = 250
img_width = 250

# ========== DATA LOADING ==========
training_ds = tf.keras.preprocessing.image_dataset_from_directory(
    str(train_dir),
    seed=101,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

testing_ds = tf.keras.preprocessing.image_dataset_from_directory(
    str(test_dir),
    seed=101,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

validation_ds = tf.keras.preprocessing.image_dataset_from_directory(
    str(val_dir),
    seed=101,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

class_names = training_ds.class_names

# Performance config
AUTOTUNE = tf.data.AUTOTUNE
training_ds = training_ds.cache().prefetch(buffer_size=AUTOTUNE)
testing_ds = testing_ds.cache().prefetch(buffer_size=AUTOTUNE)
validation_ds = validation_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ========== MODEL DEFINITION ==========
img_shape = (img_height, img_width, 3)

base_model = tf.keras.applications.MobileNetV2(input_shape=img_shape,
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    layers.Conv2D(32, 3, activation='relu'),
    layers.Conv2D(64, 3, activation='relu'),
    layers.Conv2D(128, 3, activation='relu'),
    layers.Flatten(),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# ========== MODEL TRAINING ==========
history = model.fit(training_ds, validation_data=validation_ds, epochs=1)

# ========== PLOTTING TRAINING RESULTS ==========
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Metrics')
plt.xlabel('Epochs')
plt.ylabel('Metric Value')
plt.legend()
plt.grid(True)
plt.show()

# ========== MODEL EVALUATION ==========
AccuracyVector = []
plt.figure(figsize=(20, 20))
for images, labels in testing_ds.take(1):
    predictions = model.predict(images)
    predlabel = []
    prdlbl = []

    for mem in predictions:
        predlabel.append(class_names[np.argmax(mem)])
        prdlbl.append(np.argmax(mem))

    AccuracyVector = np.array(prdlbl) == labels.numpy()

    for i in range(min(20, len(images))):
        ax = plt.subplot(5, 4, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(f'Pred: {predlabel[i]}, Actl: {class_names[labels[i]]}')
        plt.axis('off')

plt.show()

# Calculate Confusion Matrix
truePositive = trueNegative = falsePositive = falseNegative = 0
for i in range(len(AccuracyVector)):
    if predlabel[i] == class_names[labels[i]] and predlabel[i] == 'Accident':
        truePositive += 1
    elif predlabel[i] == class_names[labels[i]] and predlabel[i] == 'Non Accident':
        trueNegative += 1
    elif predlabel[i] == 'Non Accident' and class_names[labels[i]] == 'Accident':
        falseNegative += 1
    else:
        falsePositive += 1

print(f"True Positives: {truePositive}")
print(f"True Negatives: {trueNegative}")
print(f"False Positives: {falsePositive}")
print(f"False Negatives: {falseNegative}")

# ========== SAVE MODEL ==========
# Define model_dir if not already
model_dir = Path("../model")
model_dir.mkdir(parents=True, exist_ok=True)

# ========== SAVE MODEL ==========
# Recommended: Save in native Keras format (.keras)
model.save(model_dir / "accident_detection_model.keras")

print("✅ Model saved in native Keras format (.keras)")

# ========== PLOT MODEL STRUCTURE ==========
try:
    plot_model(model, to_file=str(model_dir / "model_plot.png"), show_shapes=True, show_layer_names=True)
    print("✅ Model structure plot saved.")
except ImportError:
    print("⚠️ Skipping model plot. Install 'graphviz' and 'pydot' to enable plotting. Run:")
    print("   pip install graphviz pydot")

# ========== OPTIONAL: Convert to TFLite ==========
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open(model_dir / "tf_lite_model.tflite", 'wb') as f:
    f.write(tflite_model)

print("✅ TFLite model saved.")
