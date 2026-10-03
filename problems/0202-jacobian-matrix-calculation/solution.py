import numpy as np

def jacobian_matrix(f, x: list[float], h: float = 1e-5) -> list[list[float]]:
    """
    Compute the Jacobian matrix using numerical differentiation.
    
    Args:
        f: Function that takes a list and returns a list
        x: Point at which to evaluate the Jacobian
        h: Step size for finite differences
    
    Returns:
        Jacobian matrix as list of lists
    """
    x = np.array(x, dtype=float)
    fx = np.array(f(x))
    m = len(fx)
    n = len(x)
    J = np.zeros((m, n))
    
    for j in range(n):
        x_perturbed = x.copy()
        x_perturbed[j] += h 
        f_perturbed = np.array(f(x_perturbed))
        J[:, j] = (f_perturbed - fx) / h 
        
    return J.tolist()