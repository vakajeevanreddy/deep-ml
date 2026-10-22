import numpy as np
import  math
def orthonormal_basis(vectors: list[list[float]], tol: float = 1e-10) -> list[np.ndarray]:
    # Your code here
    if not vectors:
        return []
        
    basis = []
    for vec in vectors:
        u = np.array(vec, dtype=float)
        for q in basis:
            u = u - np.dot(q,u) * q
        norm = np.linalg.norm(u)
        if norm > tol:
            basis.append(u / norm)
    return  basis