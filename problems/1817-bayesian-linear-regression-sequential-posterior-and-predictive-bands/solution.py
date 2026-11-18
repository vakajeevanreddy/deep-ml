import numpy as np

def posterior(X, y, m0, S0, sigma2):
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)
    m0 = np.asarray(m0).reshape(-1)
    S0 = np.asarray(S0)
    S0_inv = np.linalg.inv(S0)
    precesion_N = S0_inv + (1.0 / sigma2) * (X.T @ X)
    SN = np.linalg.inv(precesion_N)
    precision_weighted_mean = S0_inv @ m0 + (1.0 / sigma2) * (X.T @ y)
    mN = SN @ precision_weighted_mean
    return mN,SN 
def sequential_posterior(X, y, m0, S0, sigma2):
    # return (mN, SN, means)
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)
    m0 = np.asarray(m0).reshape(-1)
    S0 = np.asarray(S0)
    N,D = X.shape
    means = np.zeros((N,D))
    current_m = m0.copy()
    current_s = S0.copy()
    for n in range(N):
        X_N = X[n:n+1,:]
        y_N = y[n:n+1]
        current_m,current_s = posterior(X_N,y_N,current_m,current_s,sigma2)
        means[n] = current_m
    return current_m,current_s,means


def predictive(X_test, mN, SN, sigma2):
    # return (mean, var)
    X_test = np.asarray(X_test)
    mN = np.asarray(mN).reshape(-1)
    SN = np.asarray(SN)
    mean = X_test @ mN
    var = sigma2 + np.sum((X_test @ SN)* X_test, axis = 1)
    return mean,var
