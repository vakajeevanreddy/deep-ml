import numpy as np


def find_leaky_features(X: np.ndarray, y: np.ndarray, threshold: float) -> list:
    """Column indices whose |correlation| with y is >= threshold.

    Args:
        X (np.ndarray): (n, p) feature matrix.
        y (np.ndarray): (n,) target.
        threshold (float): absolute-correlation cutoff.

    Returns:
        list[int]: ascending column indices.
    """
    # Your code here
    X_arr = np.asarray(X, dtype = float)
    y_arr = np.asarray(y,dtype = float)
    if X_arr.ndim != 2:
        raise ValueError("X must be a 2D array-like structure.")
    if X_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    if X_arr.shape[0] < 2:
        return []
    y_centered = y_arr - np.mean(y_arr)
    y_norm_sq = np.dot(y_centered, y_centered)
    if y_norm_sq == 0.0:
        return []
    leakey_indices = []
    n_cols = X_arr.shape[1]
    for col_idx in range(n_cols):
        col = X_arr[:,col_idx]
        col_centered = col - np.mean(col)
        col_norm_sq = np.dot(col_centered,col_centered)
        if col_norm_sq == 0.0:
            corr = 0
        else:
            denom = np.sqrt(col_norm_sq * y_norm_sq)
            corr = float(np.dot(col_centered,y_centered)/ denom) if denom > 0.0 else 0.0
            if np.isnan(corr):
                corr = 0.0
        if abs(corr) >= threshold:
            leakey_indices.append(int(col_idx))
    return sorted(leakey_indices)

def is_affine_function_of(col: np.ndarray, y: np.ndarray) -> bool:
    """True when col == a + b * y to within a residual of 1e-9."""
    # Your code here
    col_arr = np.asarray(col, dtype = float).ravel()
    y_arr = np.asarray(y,dtype = float).ravel()
    if col_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("col and y must have the same length.")
    if col_arr.shape[0] == 0:
        return False
    design_matrix = np.column_stack([np.ones_like(y_arr),y_arr])
    params,_,_,_ = np.linalg.lstsq(design_matrix,col_arr,rcond = None)
    predictions = design_matrix @ params
    rss = float(np.sum((col_arr - predictions) ** 2))
    return bool(rss < 1e-9)
