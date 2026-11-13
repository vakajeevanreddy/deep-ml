import numpy as np

def min_norm_solution(A, b):
    M = A @ A.T
    y = np.linalg.solve(M,b)
    return A.T @ y

def min_norm_via_kkt(A, b):
    # return (x, lam)
    n = A.shape[1]
    m = A.shape[0]
    K = np.zeros((n + m, n + m))
    K[:n,:n] = 2 * np.eye(n)
    K[:n,n:] = A.T
    K[n:, :n] = A 
    rhs = np.zeros(n + m)
    rhs[n:] = b 

    sol = np.linalg.solve(K,rhs)
    X = sol[:n]
    lam = sol[n:]
    return X,lam

def min_norm_from_particular(A, x_p):
    M = A @ A.T
    rhs = A @ x_p
    y = np.linalg.solve(M,rhs)
    return A.T @ y
