import numpy as np

def mini_batch_gd_step(X: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: float, batch_indices: list, lr: float) -> np.ndarray:
    """
    Perform one mini-batch gradient descent update step for linear regression with MSE loss.
    Returns a 1D array of length D+1: updated weights followed by updated bias.
    """
    X_b = X[batch_indices]
    y_b = y[batch_indices]

    m = len(batch_indices)
    predictions = X_b @ weights + bias
    errors = predictions - y_b

    grad_W = (2.0/m) * (X_b.T @ errors)
    grad_b = (2.0/m) * np.sum(errors)

    updated_weights = weights - lr * grad_W
    updated_bias = bias - lr * grad_b
    return  np.concatenate([updated_weights, np.array([updated_bias])], axis=0)
