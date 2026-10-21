import numpy as np

def cramers_rule(A, b):
    # Your code here
    A = np.array(A,dtype = float)
    b = np.array(b,dtype = float)

    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix A must be square.")
    det_A = np.linalg.det(A)
    if np.isclose(det_A,0.0):
        return -1
    n = len(b)
    x = np.zeros(n,dtype = float)
    for i in range(n):
        A_i = np.copy(A)
        A_i[:,i] = b
        x[i] = np.linalg.det(A_i)/det_A
    return x