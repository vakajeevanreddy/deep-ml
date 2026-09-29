from typing import List, Tuple

def matrix_determinant_and_trace(matrix: List[List[float]]) -> Tuple[float, float]:
    """
    Compute the determinant and trace of a square matrix.
    """
    n = len(matrix)
    if any(len(row) != n for row in matrix):
        raise ValueError("Matrix must be square")

    trace = sum(matrix[i][i] for i in range(n))

    def determinant(mat: List[List[float]]) -> float:
        size = len(mat)
        if size == 1:
            return mat[0][0]
        if size == 2:
            return mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]

        det = 0
        for col in range(size):
            minor = [row[:col] + row[col+1:] for row in mat[1:]]
            det += ((-1) ** col) * mat[0][col] * determinant(minor)
        return det

    det = determinant(matrix)
    return det, trace
