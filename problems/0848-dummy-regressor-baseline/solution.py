import numpy as np

def dummy_regressor(y_train, n_test, strategy='mean', constant=None, quantile=None):
    """
    Baseline regressor that predicts a constant value derived from y_train.

    Args:
        y_train: 1D array-like of training target values.
        n_test: number of test predictions to return (int >= 0).
        strategy: one of 'mean', 'median', 'quantile', 'constant'.
        constant: required when strategy='constant'.
        quantile: required when strategy='quantile', must be in [0, 1].

    Returns:
        List[float] of length n_test, all equal to the chosen summary value.
    """
    if n_test < 0:
        raise ValueError("n_test must be Positive")
    if n_test == 0:
        return []
    if strategy == "constant":
        if constant is None:
            raise ValueError("constant must be provided when strategy is constant")
        p = float(constant)
        return [p] * n_test
    arr = np.asarray(y_train, dtype = float)
    if arr.size == 0:
        raise ValueError("y_train must not be empty")
    if strategy == "mean":
        p = float(np.mean(arr))
    elif strategy == "median":
        p = float(np.median(arr))
    elif strategy == "quantile":
        if quantile is None  or not (0.0 <= quantile <= 1.0):
            raise ValueError("quantile must be a number in [0, 1]")
        p = float(np.quantile(arr, quantile))
    else:
        raise ValueError(f"Unsupported strategy: {strategy!r}")
    return [p] * n_test