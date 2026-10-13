import numpy as np

def entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    mask = p > 0
    return -np.sum(p[mask] * np.log(p[mask]))  # natural log

def cross_entropy(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    mask = (p > 0) & (q > 0)
    return -np.sum(p[mask] * np.log(q[mask]))  # natural log

def entropy_and_cross_entropy(P: list[float], Q: list[float]) -> tuple[float, float]:
    p = np.asarray(P, dtype=np.float64)
    q = np.asarray(Q, dtype=np.float64)
    return entropy(p), cross_entropy(p, q)