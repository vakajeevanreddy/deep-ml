import numpy as np

def check_positive_definite(matrix: list) -> dict:
    """
    Check if a matrix is positive definite and compute its eigenvalues.
    
    Args:
        matrix: A 2D list representing a square matrix
        
    Returns:
        dict with 'is_positive_definite' (bool) and 'eigenvalues' (list of floats sorted ascending)
    """
    # Your code here
    mat = np.array(matrix,dtype = float)
    if mat.shape[0] != mat.shape[1]:
        raise  ValueError("Matrix must be square")
    eigenvalues = np.linalg.eigvals(mat)
    eigenvalues_sorted = sorted([round(float(ev), 4)for ev in eigenvalues])
    is_positive_definite = all(ev > 0 for ev in eigenvalues_sorted)
    return {
        "is_positive_definite" : is_positive_definite,
        "eigenvalues" : eigenvalues_sorted
    }