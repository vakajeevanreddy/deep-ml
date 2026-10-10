import numpy as np

def lu_decomposition(A: list) -> tuple:
	"""
	Perform LU decomposition on a square matrix using Doolittle's method.
	
	Args:
		A: Square matrix as a list of lists
	
	Returns:
		tuple: (L, U) where L is lower triangular with 1s on diagonal,
		       U is upper triangular, and A = L @ U
	"""
	# Your code here
	A = np.array(A,dtype = float)
	n = A.shape[0]
	L = np.zeros((n,n))
	U = np.zeros((n,n))
	for i in range(n):
		L[i][i] = 1
		for j in range(i,n):
			U[i][j] = A[i][j] - sum(L[i][k] * U[k][j] for k in range(i))
		for j in range(i+1 , n):
			if U[i][i] == 0:
				raise ZeroDivisionError("Zero pivot encountered during LU decomposition")
			L[j][i] = (A[j][i] - sum(L[j][k] * U[k][i] for k in range(i)))/ U[i][i]
	return L,U