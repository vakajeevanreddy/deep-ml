import numpy as np

def forward_sub(L, b):
    n = L.shape[0]
    y = np.zeros_like(b,dtype = np.float64)
    for i in range(n):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i]))/L[i,i]
    return y
def back_sub(U, y):
    n = U.shape[0]
    X = np.zeros_like(y,dtype = np.float64)
    for i in range(n - 1 ,-1,-1):
        X[i] = (y[i] - np.dot(U[i, i+1:], X[i+1:]))/U[i,i]
    return X
def chol_solve(A, b):
    L = np.linalg.cholesky(A)
    y = forward_sub(L,b)
    X = back_sub(L.T,y)
    return X
    

def logdet_chol(A):
    L = np.linalg.cholesky(A)
    diag_L = np.diagonal(L)
    return 2.0 * np.sum(np.log(diag_L))
def mvn_sample(mu, Sigma, eps):
    L = np.linalg.cholesky(Sigma)
    return eps @ L.T + mu
