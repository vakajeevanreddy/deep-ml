import numpy as np

def newton_schulz(M, num_iters: int, a: float, b: float, c: float):
    """
    Apply Newton-Schulz iterations to approximately orthogonalize M.
    Returns the resulting matrix as a nested list of floats.
    """
    # Your code here
    arr = np.array(M,dtype = float)
    frob_norm = np.linalg.norm(arr)
    if frob_norm == 0:
        return  arr.tolist()
    MK = arr/frob_norm
    for _ in range(num_iters):
        X = MK @ MK.T
        X_MK = X @ MK
        X_X_MK = X @ X_MK
        MK = a * MK + b * X_MK + c * X_X_MK
    return  MK.tolist()
