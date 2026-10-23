import numpy as np

def cholesky_decomposition(A):
    """
    Perform Cholesky decomposition on a symmetric positive-definite matrix.
    
    Args:
        A: A symmetric positive-definite matrix (2D list or numpy array)
    
    Returns:
        L: Lower triangular matrix such that A = L @ L.T as a 2D list,
            or -1 if decomposition is not possible
    """
    if A is None:
        return -1
        
    # Convert input to a numpy array for easier shape checking and math
    try:
        A = np.array(A, dtype=float)
    except:
        return -1
        
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        return -1
        
    n = A.shape[0]
    L = np.zeros((n, n))
    
    try:
        for i in range(n):
            for j in range(i + 1):
                # Sum of previous products for the dot product term
                s = sum(L[i][k] * L[j][k] for k in range(j))
                
                if i == j:
                    # Diagonal elements
                    val = A[i][i] - s
                    if val <= 0:
                        return -1
                    L[i][j] = np.sqrt(val)
                else:
                    # Off-diagonal elements
                    if L[j][j] == 0:
                        return -1
                    L[i][j] = (A[i][j] - s) / L[j][j]
                    
        # Return as a 2D list to match the specification
        return L.tolist()
        
    except Exception:
        return -1