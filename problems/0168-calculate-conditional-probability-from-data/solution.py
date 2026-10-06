def conditional_probability(data, x, y):
    """
    Returns the probability P(Y=y|X=x) from list of (X, Y) pairs.
    Args:
      data: List of (X, Y) tuples
      x: value of X to condition on
      y: value of Y to check
    Returns:
      float: conditional probability, rounded to 4 decimal places
    """
    # Your code here
    x_rows = [pair for pair in data if pair[0] == x]
    if not x_rows:
      return 0.0
    y_count = sum(1 for pair in x_rows if pair[1] == y)
    return y_count/len(x_rows)