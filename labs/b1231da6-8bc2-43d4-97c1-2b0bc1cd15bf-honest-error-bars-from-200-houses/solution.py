# --- cell 1 ---
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV

def fit_predict(X_train, y_train, X_test):
    X_tr = X_train.copy()
    X_te = X_test.copy()
    combined_columns = [0, 2, 3, 4, 5]
    X_tr[:, combined_columns] = np.log1p(X_tr[:, combined_columns])
    X_te[:, combined_columns] = np.log1p(X_te[:, combined_columns])
    
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    
    ym = np.mean(y_train)
    ys = np.std(y_train)
    if ys < 1e-18:
        ys = 1.0
    y_tr_s = (y_train - ym) / ys
    phi = np.hstack([np.ones((len(X_tr_s), 1)), X_tr_s])
    phi_t = np.hstack([np.ones((len(X_te_s), 1)), X_te_s])
    rcv = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
    rcv.fit(phi[:, 1:], y_tr_s)
    lam = rcv.alpha_
    N, D = phi.shape
    A = phi.T @ phi + lam * np.eye(D)
    invA = np.linalg.inv(A)
    W = invA @ (phi.T @ y_tr_s)
    
    res = y_tr_s - phi @ W
    sigma2 = np.sum(res ** 2) / max(1, N - D)
    if sigma2 < 1e-4:
        sigma2 = 1e-4
    mean_s = phi_t @ W
    vars_s = sigma2 * (1.0 + np.sum((phi_t @ invA) * phi_t, axis=1))
    mean = mean_s * ys + ym
    var = vars_s * (ys ** 2)
    var = np.maximum(var, 1e-4)

    return mean, var