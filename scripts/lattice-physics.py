#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

def get_data(path):
    data = np.loadtxt(path, delimiter=',', comments="#")
    X = data[:, :-1]
    y = data[:, -1]
    return X, y

X_train, y_train = get_data('data/lattice-physics+(pwr+fuel+assembly+neutronics+simulation+results)/raw.data')
X_test, y_test = get_data('data/lattice-physics+(pwr+fuel+assembly+neutronics+simulation+results)/test.data')

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build a simple feedforward neural network
model = tf.keras.Sequential()

# Input layer
model.add(tf.keras.layers.InputLayer(shape=(X_train.shape[1],)))

# Hidden layers
model.add(tf.keras.layers.Dense(64, activation='relu'))  # First hidden layer with 64 units
model.add(tf.keras.layers.Dense(32, activation='relu'))  # Second hidden layer with 32 units

# Output layer (adjust the number of neurons and activation for your problem)
model.add(tf.keras.layers.Dense(1, activation='linear'))  # Assuming regression, use 'softmax' or 'sigmoid' for classification

# Model summary
model.summary()

# Compile the model
model.compile(optimizer='sgd', loss='mse', metrics=['mae'])  # Using Mean Squared Error for regression

# Train the model
#history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
model.fit(X_train, y_train, epochs=50)

# Evaluate the model performance on the test set
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")