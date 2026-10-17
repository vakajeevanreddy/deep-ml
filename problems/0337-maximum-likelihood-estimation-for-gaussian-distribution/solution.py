import numpy as np

def gaussian_mle(data: np.ndarray) -> tuple:
    """
    Compute Maximum Likelihood Estimates for Gaussian distribution parameters.
    
    Args:
        data: 1D numpy array of observations
        
    Returns:
        Tuple of (mean_mle, variance_mle)
    """
    # Your code here
    mu_hat = np.mean(data)
    sigma_2_hat = np.mean((data - mu_hat) ** 2)
    return mu_hat,sigma_2_hat