#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

# Step 1: Load and prepare the Iris dataset
iris = datasets.load_iris()
X = iris.data  # Features
y = iris.target  # Labels

# One-hot encode the labels
y = np.eye(3)[y]  # Convert labels to one-hot encoding

# Split data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, input_dim=4, activation='relu'),  # Hidden layer with 64 neurons
    tf.keras.layers.Dense(3, activation='softmax')  # Output layer with 3 neurons for 3 classes
])

# Compile the model
model.compile(optimizer='adam', 
              loss='categorical_crossentropy',  # Suitable for multiclass classification
              metrics=['accuracy'])

# Step 3: Train the model
model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1)

# Step 4: Make predictions
y_pred = model.predict(X_test)

# Convert predictions from probabilities to class labels
y_pred_class = np.argmax(y_pred, axis=1)
y_test_class = np.argmax(y_test, axis=1)

# Step 5: Compute confusion matrix and metrics
cm = confusion_matrix(y_test_class, y_pred_class)
print("Confusion Matrix:")
print(cm)

# Compute precision, recall, f1 score (for multiclass)
precision = precision_score(y_test_class, y_pred_class, average='weighted')
recall = recall_score(y_test_class, y_pred_class, average='weighted')
f1 = f1_score(y_test_class, y_pred_class, average='weighted')
accuracy = accuracy_score(y_test_class, y_pred_class)

# Output metrics
print(f"\nPrecision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"Accuracy: {accuracy}")
