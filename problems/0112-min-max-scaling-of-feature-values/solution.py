def min_max(x: list[float]) -> list[float]:
    """
    Perform Min-Max normalization to scale values to [0, 1].
    
    Args:
        x: A list of numerical values
    
    Returns:
        A new list with values normalized to [0, 1]
    """
    # Your code here
    if not x:
        return []
    min_val = min(x)
    max_val = max(x)
    if min_val == max_val:
        return [0.0] * len(x)
    return  [(x - min_val) / (max_val - min_val)for x in x]