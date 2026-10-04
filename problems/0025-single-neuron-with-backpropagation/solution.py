import numpy as np

def train_neuron(features: np.ndarray, labels: np.ndarray,initial_weights: np.ndarray, initial_bias: float,learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
    # Convert inputs to numpy arrays
    X = np.array(features, dtype=float)
    y = np.array(labels, dtype=float)
    weights = np.array(initial_weights, dtype=float)
    bias = float(initial_bias)

    m, n = X.shape
    mse_values = []

    # Sigmoid activation
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    for _ in range(epochs):
        # Forward pass
        z = X.dot(weights) + bias
        preds = sigmoid(z)

        # Compute MSE before update
        mse = np.mean((preds - y) ** 2)
        mse_values.append(round(mse, 4))

        # Gradients
        dL_dp = 2 * (preds - y) / m
        dp_dz = preds * (1 - preds)
        dL_dz = dL_dp * dp_dz

        grad_w = X.T.dot(dL_dz)
        grad_b = np.sum(dL_dz)

        # Update parameters
        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b

    # Round results
    updated_weights = np.round(weights, 4)
    updated_bias = round(bias, 4)

    return updated_weights, updated_bias, mse_values
