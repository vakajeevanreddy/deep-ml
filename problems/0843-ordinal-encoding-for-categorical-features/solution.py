import numpy as np

def ordinal_encode(X, categories, unknown_value = -1):
    """
    Encode categorical features as integer codes using a provided ordering.

    Args:
        X: 2D list/array of shape (n_samples, n_features) with string values.
        categories: list of length n_features; categories[j] is an ordered
                    list of valid categories for column j.

    Returns:
        np.ndarray of shape (n_samples, n_features), dtype int.
        Unknown categories are encoded as -1.
    """
    if len(X) == 0:
        return np.empty((0, len(categories)), dtype=int)
    rank_maps = [
        {category: idx for idx, category in enumerate(col_categories)}
        for col_categories in categories
    ]
    encoded = [
        [rank_maps[col_idx].get(val, unknown_value) for col_idx, val in enumerate(row)]
        for row in X
    ]
    return np.array(encoded, dtype=int)