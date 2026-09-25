from collections import Counter
def empirical_pmf(samples):
    """
    Given an iterable of integer samples, return a list of (value, probability)
    pairs sorted by value ascending.
    """
    # TODO: Implement the function
    total = len(samples)
    counts = Counter(samples)
    pmf = [(value,counts[value]/total) for value in sorted(counts)]
    return  pmf