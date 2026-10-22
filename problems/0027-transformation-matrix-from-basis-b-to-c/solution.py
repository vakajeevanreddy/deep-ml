import numpy as np

def transform_basis(B: list[list[float]], C: list[list[float]]) -> list[list[float]]:
    B_mat = np.array(B, dtype=float)
    C_mat = np.array(C, dtype=float)
    p = np.linalg.solve(C_mat, B_mat)
    
    return np.round(p, 4).tolist()