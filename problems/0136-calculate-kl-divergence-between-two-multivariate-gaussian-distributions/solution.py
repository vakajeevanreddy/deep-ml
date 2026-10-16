import numpy as np

def multivariate_kl_divergence(mu_p: np.ndarray, Cov_p: np.ndarray, mu_q: np.ndarray, Cov_q: np.ndarray) -> float:
    """
    Computes the KL divergence between two multivariate Gaussian distributions.
    
    Parameters:
    mu_p: mean vector of the first distribution
    Cov_p: covariance matrix of the first distribution
    mu_q: mean vector of the second distribution
    Cov_q: covariance matrix of the second distribution

    Returns:
    KL divergence as a float
    """
    # Your code here
    mu_p = np.atleast_1d(mu_p)
    mu_q = np.atleast_1d(mu_q)
    Cov_p = np.atleast_2d(Cov_p)
    Cov_q = np.atleast_2d(Cov_q)

    d = mu_p.shape[0]

    inv_cov_q = np.linalg.inv(Cov_q)
    det_cov_q = np.linalg.det(Cov_q)
    det_cov_p = np.linalg.det(Cov_p)

    trace_term = np.trace(inv_cov_q @ Cov_p)

    diff = mu_q - mu_p
    mahalnobis = diff.T @ inv_cov_q @ diff
    log_det_term = np.log(det_cov_q / det_cov_p)

    kl = 0.5 * (log_det_term - d + trace_term + mahalnobis)
    return float(kl)