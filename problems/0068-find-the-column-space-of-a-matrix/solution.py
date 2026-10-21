import numpy as np

def matrix_image(A):
    A = np.array(A, dtype=float)
    m, n = A.shape
    M = A.copy()

    pivot_cols = []
    row = 0
    for col in range(n):
        pivot = row
        while pivot < m and abs(M[pivot, col]) < 1e-10:
            pivot += 1
        if pivot == m:
            continue
        M[[row, pivot]] = M[[pivot, row]]
        pivot_val = M[row, col]
        M[row] /= pivot_val
        for i in range(m):
            if i != row:
                factor = M[i, col]
                M[i] -= factor * M[row]
        pivot_cols.append(col) 
        row += 1
        if row == m:
            break
    basis = A[:, pivot_cols].tolist()
    return basis