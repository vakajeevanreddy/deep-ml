import numpy as np

def is_linearly_independent(vectors: list[list[float]]) -> bool:
    """
    Check if a set of vectors is linearly independent.
    
    Args:
        vectors: List of vectors, where each vector is a list of floats.
                 All vectors must have the same dimension.
        
    Returns:
        True if vectors are linearly independent, False otherwise.
    """
    # Your code here
    if not vectors:
        return True
    A = np.array(vectors,dtype = float)
    rank = np.linalg.matrix_rank(A)
    return rank == A.shape[0]
    