import numpy as np

def compute_norm(arr: np.ndarray, norm_type: str) -> float:
    """
    Compute the specified norm of the input array.

    'l1', 'l2' and 'linf' are entrywise norms (sum, Euclidean, max).
    'frobenius' is a matrix norm and must raise a ValueError if arr is not 2D.
    """
    if norm_type == "l1":
        return float(np.sum(np.abs(arr)))
    elif norm_type == "l2":
        return float(np.sqrt(np.sum(arr**2)))
    elif norm_type == "linf":
        return float(np.max(np.abs(arr)))
    elif norm_type == "frobenius":
        if arr.ndim != 2:
            raise ValueError("Frobenius norm is only defined for 2D matrices.")
        return float(np.linalg.norm(arr, ord='fro'))
    else:
        raise ValueError("Invalid norm_type. Choose from 'l1', 'l2', 'linf', 'frobenius'.")
