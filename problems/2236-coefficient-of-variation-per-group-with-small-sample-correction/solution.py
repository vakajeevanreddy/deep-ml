import math 
from collections import defaultdict
def grouped_cv(values, groups):
    """Return {group: bias-corrected coefficient of variation or None}."""
    if len(values) != len(groups):
        raise ValueError("Values and groups must have the same length")
    if not values:
        return {}
    buckets = defaultdict(list)
    for value,group in zip(values,groups):
        buckets[group].append(value)
    result = {}
    for group,group_values in buckets.items():
        n = len(group_values)
        if n < 2 :
            result[group] = None
            continue
        mean = math.fsum(group_values) / n
        if mean == 0.0:
            result[group] = None
            continue
        squared_diff = ((x - mean) ** 2 for x in group_values)
        sample_std = math.sqrt(math.fsum(squared_diff)/(n - 1))
        correction = 1.0 + 1.0 / (4.0 * n)
        result[group] = correction  * sample_std / abs(mean)
    return result