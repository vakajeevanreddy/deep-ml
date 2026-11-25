import numpy as np

def column_transformer(X, transformers):
    """
    Apply per-column-group transformations and horizontally concatenate the results.

    Args:
        X: 2D array-like of shape (n_samples, n_features)
        transformers: list of (name, kind, columns) tuples

    Returns:
        2D list of floats with transformed and concatenated columns.
    """
    X = np.asarray(X, dtype = float)
    results_block = []
    for name,kind,columns in transformers:
        if kind == "drop":
            continue
        block = X[:,columns]
        if kind == "passthrough":
            results_block.append(block)
        elif kind == "standard":
            mean = block.mean(axis = 0)
            std = block.std(axis = 0)
            transformed = np.zeros_like(block)
            nonzero_std = std != 0
            transformed[:, nonzero_std] = (
                block[:,nonzero_std] - mean[nonzero_std]) / std[nonzero_std]
    
            results_block.append(transformed)
        elif kind == "minmax":
            min_vals = block.min(axis = 0)
            max_vals = block.max(axis = 0)
            ranges = max_vals - min_vals
            transformed = np.zeros_like(block)
            nonzero_range = ranges != 0
            transformed[:,nonzero_range] = (
                block[: , nonzero_range] - min_vals[nonzero_range]
            ) / ranges[nonzero_range]
            results_block.append(transformed)
        else:
            raise ValueError(f"Unknown transformer kind: {kind}")
    if not results_block:
        return np.empty((X.shape[0], 0))
    return np.hstack(results_block)