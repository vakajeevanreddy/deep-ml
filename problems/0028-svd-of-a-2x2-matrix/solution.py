import numpy as np

def svd_2x2(A: np.ndarray) -> tuple:
    A = np.asarray(A, dtype=float)
    if A.shape != (2, 2):
        raise ValueError("Input matrix must be of shape (2, 2).")
    
    B = A.T @ A 
    b11, b12 = B[0, 0], B[0, 1]
    b21, b22 = B[1, 0], B[1, 1]
    
    trace = b11 + b22
    det = b11 * b22 - b12 * b12
    disc = max(0.0, trace ** 2 - 4 * det)
    
    lam1 = (trace + np.sqrt(disc)) / 2.0
    lam2 = (trace - np.sqrt(disc)) / 2.0
    
    lam1 = max(0.0, lam1)
    lam2 = max(0.0, lam2)
    
    swap = False
    if lam1 < lam2:
        lam1, lam2 = lam2, lam1
        swap = True
        
    s1 = np.sqrt(lam1)
    s2 = np.sqrt(lam2)
    s = np.array([s1, s2])
    
    def eigen_vector(lam):
        if abs(b12) > 1e-12:
            v = np.array([b12, lam - b11])
        else:
            if abs(b11 - lam) < abs(b22 - lam):
                v = np.array([1.0, 0.0])
            else:
                v = np.array([0.0, 1.0])
        norm = np.sqrt(v[0] ** 2 + v[1] ** 2)
        if norm > 1e-12:
            v = v / norm
        return v 
        
    v1 = eigen_vector(lam1)
    v2 = eigen_vector(lam2)
    
    if abs(lam1 - lam2) < 1e-12:
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
    else:
        v2 = v2 - np.dot(v2, v1) * v1
        norm_v2 = np.sqrt(v2[0] ** 2 + v2[1] ** 2)
        if norm_v2 > 1e-12:
            v2 = v2 / norm_v2
            
    if swap:
        v1, v2 = v2, v1
        
    V = np.stack([v1, v2], axis=0)
    
    eps = 1e-12
    if s1 > eps:
        u1 = (A @ v1) / s1
    else:
        u1 = np.array([1.0, 0.0])
        
    if s2 > eps:
        u2 = (A @ v2) / s2
    else:
        u2 = np.array([-u1[1], u1[0]])
        
    U = np.column_stack([u1, u2])
    
    return U, s, V