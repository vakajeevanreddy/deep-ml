import numpy as np
from collections import Counter
import math
def dummy_classifier(y_train, n_test, strategy, constant=None):
    """
    Produce baseline predictions of length n_test using the given strategy.
    Returns a Python list of predicted labels.
    """
    if n_test < 0:
        raise ValueError("n_test must be non-negative")
    if n_test == 0:
        return []
    if strategy == "constant":
        if constant is None:
            raise ValueError("constant must be provided when strategy is constant")
        return [constant] * n_test
    counts = Counter(y_train)
    sorted_class = sorted(counts.keys())
    if strategy == "most_frequent":
        best_class = min(sorted_class, key = lambda c : (-counts[c], c))
        return [best_class] * n_test
    if strategy == "uniform":
        k = len(sorted_class)
        return [sorted_class[i % k] for i  in range(n_test)]
    if strategy == "stratified":
        total = len(y_train)
        allocations = {}
        fractions = {}
        for c in sorted_class:
            exact = n_test * (counts[c] / total)
            base = math.floor(exact)
            allocations[c] = base
            fractions[c] = exact - base
        remainder = n_test - sum(allocations.values())
        if remainder > 0:
            ranked_classes = sorted(sorted_class, key = lambda c : (-fractions[c],c))
            for c in ranked_classes[:remainder]:
                allocations[c] += 1
        predictions = []
        for c in sorted_class:
            predictions.extend([c] * allocations[c])
        return predictions
    raise ValueError(f"Unsupported strategy: {strategy!r}")