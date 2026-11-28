import numpy as np


def causal_zscore(x: np.ndarray, window: int) -> np.ndarray:
    """Rolling z-score using only the trailing `window` values at each point.

    Args:
        x (np.ndarray): 1-D input series.
        window (int): lookback length, inclusive of the current point.

    Returns:
        np.ndarray: same length as x; the first window-1 entries are np.nan.
    """
    # Your code here
    arr = np.asarray(x, dtype=float)
    n = len(arr)
    out = np.full(n, np.nan, dtype=float)
    if window <= 0 or window > n:
        return out

    if window == 1:
        out[:] = 0.0
        return out

    windows = np.lib.stride_tricks.sliding_window_view(arr, window)
    means = np.mean(windows, axis=1)
    std = np.std(windows, axis=1, ddof=0)
    curr_val = arr[window - 1 :]

    zero_std = np.isclose(std, 0.0, atol=1e-12, rtol=0.0)
    equal_to_mean = np.isclose(curr_val, means, atol=1e-12, rtol=1e-12)

    with np.errstate(divide="ignore", invalid="ignore"):
        z_vals = np.where(
            ~zero_std,
            (curr_val - means) / std,
            np.where(equal_to_mean, 0.0, np.nan),
        )

    out[window - 1 :] = z_vals
    return out


def has_lookahead(feature_fn, x: np.ndarray) -> bool:
    """Return True if feature_fn uses information from the future."""
    # Your code here
    base_input = np.array(x,copy = True)
    perturb_input = np.array(x,dtype = float , copy = True)
    perturb_input[-1] += 1000.0

    base = np.asarray(feature_fn(base_input),dtype = float)
    new = np.asarray(feature_fn(perturb_input),dtype = float)
    if len(base) <= 1 or len(new) <= 1:
        return False
    return not bool(np.allclose(base[:-1],new[:-1],equal_nan = True))
