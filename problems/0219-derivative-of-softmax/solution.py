import numpy as np

def softmax_derivative(x: list[float]) -> list[list[float]]:
    """
    Compute the Jacobian matrix of the softmax function.
    
    Args:
        x: Input vector of real numbers
        
    Returns:
        Jacobian matrix J where J[i][j] = d(softmax_i)/d(x_j)
    """
    x = np.array(x, dtype=float)
    e_x = np.exp(x - np.max(x))
    s = e_x / e_x.sum()
    s_col = s.reshape(-1, 1)
    J = np.diagflat(s) - np.dot(s_col, s_col.T)
    return J.tolist()
