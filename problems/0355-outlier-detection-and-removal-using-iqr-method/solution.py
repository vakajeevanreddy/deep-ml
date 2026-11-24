import numpy as np

def detect_outliers_iqr(data: list[float], k: float = 1.5) -> dict:
    """
    Detect and remove outliers using the IQR method.
    
    Args:
        data: List of numerical values
        k: IQR multiplier for determining outlier bounds (default 1.5)
    
    Returns:
        Dictionary with 'cleaned_data', 'outlier_indices', 'lower_bound', 'upper_bound'
    """
    # Your code here
    if not data:
        return {
            "cleaned_data" : [],
            "outlier_indices" : [],
            "lower_bound" : 0.0,
            "upper_bound" : 0.0
        }
    arr = np.asarray(data,dtype = float)
    q1,q3 = np.percentile(arr,[25,75])
    iqr = q3 - q1
    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr

    outlier_mask = (arr < lower_bound) | (arr > upper_bound)
    outlier_indices = np.flatnonzero(outlier_mask).astype(int).tolist()
    cleaned_data = [round(float(x), 4) for x in arr[~outlier_mask]]
    return {
        "cleaned_data" : cleaned_data,
        "outlier_indices" : outlier_indices,
        "lower_bound" : round(float(lower_bound),4),
        "upper_bound" : round(float(upper_bound),4 )
    }