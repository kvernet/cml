# Machine Learning in C

I have decided to quickly implement a CML (C Machine Learning) library to test the mathematical fundamentals behind this technology. The code may contain bugs, but the main goal was to understand how neural networks work in practice.

I ran the library on several datasets from [https://archive.ics.uci.edu/](https://archive.ics.uci.edu/), including the Iris dataset, Heart Disease dataset, Wine Quality dataset, and the Lattice-Physics (PWR fuel assembly neutronics simulation results) dataset. I also implemented some simple TensorFlow scripts to compare results and performance.

I also implemented three gate logics (AND, OR, XOR) from the C library and TensorFlow to check if the library works.

I implemented gradient descent as my only optimizer and two loss functions: Mean Squared Error (MSE) for regression, and Softmax Cross-Entropy for multiclass classification. To evaluate the performance of a model, I implemented several metrics for both regression (Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), R-Squared (R²), Adjusted R-Squared (aR²), Mean Absolute Percentage Error (MAPE), Symmetric Mean Absolute Percentage Error (sMAPE), Huber Loss (HLoss), Explained Variance Score (EVS), and Median Absolute Error (MedAE)), and classification (confusion matrix, precision, accuracy, and F1-score).

## Note
Sometimes, I increase the learning rate at each epoch to accelerate convergence to the minimum of the loss/cost function. Indeed, as the model approaches the minimum, the gradient tends to zero, and the step size becomes smaller. By increasing the learning rate, I aim to maintain a larger step size. However, one must use a variable learning rate with caution, as it could overshoot the minimum of the loss function.

## Compile
To compile the library, run the following command in a terminal:
```
make
```

## Examples
To compile the examples, run the following command in a terminal:
```
make examples
```

## TODO
- Implement a Pseudo-Random Number Generator (PRNG) using the Mersenne Twister, for instance.
- Use parallelism to improve performance in matrix calculations.
- Implement more optimizers, such as ADAM.
- Etc.