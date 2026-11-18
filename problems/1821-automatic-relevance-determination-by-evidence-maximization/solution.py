import numpy as np

def ard_fit(X, y, sigma2, n_iter, alpha_cap=1e8):
    N, D = X.shape
    y = np.asarray(y).flatten()
    alpha = np.ones(D)
    log_ev = []
    for _ in range(n_iter):
        A = np.diag(alpha)
        XTX_sigma2 = (X.T @ X) / sigma2
        sigma_inv = A + XTX_sigma2
        XTy_sigma2 = (X.T @ y) / sigma2
        try:
            L = np.linalg.cholesky(sigma_inv)
            log_det_sigma_inv = 2.0 * np.sum(np.log(np.diag(L)))
            z = np.linalg.solve(L, XTy_sigma2)
            m = np.linalg.solve(L.T, z)
        except np.linalg.LinAlgError:
            sigma_inv += 1e-6 * np.eye(D)
            L = np.linalg.cholesky(sigma_inv)
            log_det_sigma_inv = 2.0 * np.sum(np.log(np.diag(L)))
            z = np.linalg.solve(L, XTy_sigma2)
            m = np.linalg.solve(L.T, z)
        residual = y - X @ m
        quad_term = (1.0 / sigma2) * np.sum(residual ** 2) + np.sum(alpha * (m ** 2))
        log_evidence = -0.5 * (
            N * np.log(2 * np.pi) + N * np.log(sigma2) + log_det_sigma_inv - np.sum(np.log(alpha)) + quad_term
        )
        log_ev.append(log_evidence)
        Sigma = np.linalg.inv(sigma_inv)
        diag_sigma = np.diag(Sigma)
        gamma = 1.0 - alpha * diag_sigma
        m_sq = m ** 2
        alpha = np.where(m_sq > 1e-16, gamma / m_sq, alpha_cap)
        alpha = np.clip(alpha, 0.0, alpha_cap)
    A = np.diag(alpha)
    sigma_inv = A + (X.T @ X) / sigma2
    XTy_sigma2 = (X.T @ y) / sigma2
    try:
        L = np.linalg.cholesky(sigma_inv)
        z = np.linalg.solve(L, XTy_sigma2)
        m = np.linalg.solve(L.T, z)
    except np.linalg.LinAlgError:
        sigma_inv += 1e-6 * np.eye(D)
        L = np.linalg.cholesky(sigma_inv)
        z = np.linalg.solve(L, XTy_sigma2)
        m = np.linalg.solve(L.T, z)
    return m, alpha, log_ev