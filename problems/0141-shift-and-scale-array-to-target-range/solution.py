import numpy as np

def convert_range(values: np.ndarray, c: float, d: float) -> np.ndarray:
    """
    Shift and scale values from their original range [min, max] to a target [c, d] range.
    """
    # Your code here
    x = np.asarray(values,dtype = float)
    a = np.min(x)
    b = np.max(x)
    if a == b:
        return np.full_like(x,c,dtype = float)
    return c + ((d - c) / (b-a))*(x-a)