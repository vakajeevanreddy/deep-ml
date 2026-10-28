import numpy as np

def lr_range_finder(X, y, w0, a, b, n_steps):
    """
    Sweep learning rate from 10**a to 10**b over n_steps full-batch GD updates
    on a linear regression model (no bias). Return the list of MSE losses after
    each update.
    """
    n = X.shape[0]
    w = w0.astype(float).copy()
    loses = []
    for k in range(n_steps):
        if n_steps == 1:
            lr = 10 ** a 
        else:
            lr = 10 ** (a + (b - a) * k / (n_steps - 1))
        grad = (2.0 / n) * (X.T @ (X @ w - y))
        w = w - lr * grad 
        loss = float(np.mean((X @ w - y)**2))
        loses.append(loss)
    return loses