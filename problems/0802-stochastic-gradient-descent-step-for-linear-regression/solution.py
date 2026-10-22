import numpy as np

def sgd_update(X: np.ndarray, y: np.ndarray, weights: np.ndarray, learning_rate: float, n_iter: int) -> list:
    """
    Perform n_iter steps of stochastic gradient descent on a linear regression
    model with MSE loss, cycling through samples in order.

    Returns the final weight vector as a Python list.
    """
    n_samples = X.shape[0]
    w = np.array(weights,dtype = float)
    for i in range(n_iter):
        idx = i % n_samples
        x_i = X[idx]
        y_i = y[idx]

        y_pred = np.dot(x_i,w)
        error = y_pred - y_i

        grad =2.0 * error * x_i
        w -= learning_rate * grad
    return w.tolist() 
