#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, f1_score


x = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=int)

y= np.array([
    [1, 0],
    [0, 1],
    [0, 1],
    [0, 1]
], dtype=int)


# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(4, input_dim=2, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])



model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy')
model.fit(x, y, epochs=1000, verbose=1)

# Make predictions
yhat = model.predict(x)

# Convert the probabilities to class labels
yhat_class = np.argmax(yhat, axis=1)
y_class = np.argmax(y, axis=1)

# Metrics
cm = confusion_matrix(y_class, yhat_class)
precision = precision_score(y_class, yhat_class)
accuracy = accuracy_score(y_class, yhat_class)
f1 = f1_score(y_class, yhat_class)

# Output metrics
print("Confusion Matrix:")
print(cm)
print(f"\nPrecision: {precision}")
print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1}")