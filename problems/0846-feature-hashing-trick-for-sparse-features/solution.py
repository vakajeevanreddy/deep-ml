import hashlib

def feature_hash(features, n_features):
    """
    Apply the (signed) feature hashing trick.

    Args:
        features: dict mapping feature name (str) -> value (float).
        n_features: int, size of the output vector.

    Returns:
        list of length n_features.
    """
    if not isinstance(n_features,int) or  n_features <= 0:
        raise ValueError("n_features must be a positive integer")
    vector = [0.0]*n_features
    for name,value in features.items():
        digest = hashlib.md5(name.encode("utf-8")).hexdigest()
        bucket_int = int(digest[:8],16)
        index = bucket_int % n_features
        sign_int = int(digest[8:16],16)
        sign = 1.0 if sign_int % 2 == 0 else -1.0
        vector[index] += sign * float(value)
    return vector