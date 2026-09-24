import numpy as np
from collections import Counter
def descriptive_statistics(data: list | np.ndarray) -> dict:
    """
    Calculate various descriptive statistics metrics for a given dataset.
    
    Args:
        data: List or numpy array of numerical values
    
    Returns:
        Dictionary containing mean, median, mode, variance, standard deviation,
        percentiles (25th, 50th, 75th), and interquartile range (IQR)
    """
    # Your code here
    arr = np.array(data,dtype = float)
    mean_val = np.mean(arr)
    median_val = np.median(arr)
    counts = Counter(arr)
    max_freq = max(counts.values())
    mode_cand = [val for val,freq in counts.items() if freq == max_freq]
    mode_val = min(mode_cand)
    variance_val = np.var(arr)
    std_val = np.std(arr)

    p25 = np.percentile(arr,25)
    p50 = np.percentile(arr,50)
    p75 = np.percentile(arr,75)

    iQR = p75-p25
    return {
        "mean" : mean_val,
        "median" : median_val,
        "mode" : mode_val,
        "variance" : variance_val,
        "standard_deviation" : std_val,
        "25th_percentile" : p25,
        "50th_percentile" : p50,
        "75th_percentile" : p75,
        "interquartile_range" : iQR
    }