import numpy as np

def gauss_seidel(A, b, n, x_ini=None):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    
    if x_ini is None:
        x = np.zeros_like(b, dtype=float)
    else:
        x = np.array(x_ini, dtype=float)
        
    for _ in range(n):
        for i in range(len(b)):
            s = sum(A[i, j] * x[j] for j in range(len(b)) if j != i)
            x[i] = (b[i] - s) / A[i, i]
            
    return x