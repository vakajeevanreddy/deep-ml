import numpy as np

def map_estimate_bernoulli(observations: list, alpha: float, beta: float) -> float:
    """
    Compute the Maximum A Posteriori (MAP) estimate for a Bernoulli parameter.
    
    Args:
        observations: List of binary observations (0s and 1s)
        alpha: Alpha parameter of Beta prior (>= 1)
        beta: Beta parameter of Beta prior (>= 1)
    
    Returns:
        MAP estimate of the probability parameter, rounded to 4 decimal places
    """
    success = np.sum(observations)
    failures = len(observations) - success
    alpha_post = alpha + success
    beta_post = beta + failures
    if alpha_post > 1 and beta_post > 1:
        theta_map = (alpha_post - 1) / (alpha_post + beta_post - 2)
    elif alpha_post > 1 and beta_post > 1:
        theta_map = 0.0
    elif alpha_post > 1 and beta_post <= 1:
        theta_map = 1.0
    else:
        theta_map = 0.0 if success == 0 else 1
    return round(theta_map,4)