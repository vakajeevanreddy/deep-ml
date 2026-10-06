import numpy as np

def cross_product(a, b):
    # Your code here
    if len(a) != 3 or len(b) != 3:
        raise  ValueError("Both vectors must be 3-dimensional")
    c1 = a[1] * b[2] - a[2] * b[1]
    c2 = a[2] * b[0] - a[0] * b[2]
    c3 = a[0] * b[1] - a[1] * b[0]
    return [c1,c2,c3]