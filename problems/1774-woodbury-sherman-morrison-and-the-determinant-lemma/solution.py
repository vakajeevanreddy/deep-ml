import numpy as np

def woodbury_solve(sigma_diag, X, B):
    sigma_diag = np.asarray(sigma_diag, dtype = float)
    X = np.asarray(X, dtype = float)
    B = np.asarray(B, dtype = float)
    N, D = X.shape
    is_1D = (B.ndim == 1)
    if is_1D:
        B = B[:, np.newaxis]
    sigma_inv_b = B / sigma_diag[:, np.newaxis]
    sigma_inv_x = X / sigma_diag[:, np.newaxis]
    MP = np.eye(D) + X.T @ sigma_inv_x
    Z = X.T @ sigma_inv_b
    W = np.linalg.solve(MP, Z)
    result = sigma_inv_b - sigma_inv_x @ W  # Fixed: combined into a single line with subtraction
    if is_1D:
        return result.squeeze(axis = 1)
    return result

def sherman_morrison(A_inv, u, v):
    A_inv = np.asarray(A_inv, dtype = float)
    u = np.asarray(u, dtype = float)
    v = np.asarray(v, dtype = float)
    A_inv_u = A_inv @ u 
    v_A_inv = v @ A_inv
    scalar_denom = 1.0 + np.dot(v, A_inv_u)  
    if np.isclose(scalar_denom, 0.0) or scalar_denom == 0.0:
        raise ValueError("The update is singular: 1 + v^T @ A^-1 @ u is zero.")
    correction = np.outer(A_inv_u, v_A_inv) / scalar_denom
    return A_inv - correction  
def logdet_rank_one_update(logdet_A, A_inv, u, v):
    A_inv = np.asarray(A_inv, dtype = float)
    u = np.asarray(u, dtype = float)
    v = np.asarray(v, dtype = float)
    scalar_term = 1.0 + np.dot(v, A_inv @ u)
    if scalar_term <= 0:
        raise ValueError("The updated determinant is not positive")
    return logdet_A + np.log(scalar_term)