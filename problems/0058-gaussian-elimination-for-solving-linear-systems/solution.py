import numpy as np

def gaussian_elimination(A, b):
	"""
	Solves the system Ax = b using Gaussian Elimination with partial pivoting.
    
	:param A: Coefficient matrix
	:param b: Right-hand side vector
	:return: Solution vector x
	"""
	A = A.astype(float)
	b = b.astype(float)
	n = len(b)
	for i in range(n):
		max_row = np.argmax(abs(A[i:,i]))+i
		if A[max_row,i] == 0:
			raise ValueError("Matrix is singular or nearly singular")
		if max_row != i:
			A[[i,max_row]] = A[[max_row,i]]
			b[[i,max_row]] = b[[max_row,i]]
		for j in range(i+1,n):
			factor = A[j,i]/A[i,i]
			A[j,i:] -= factor*A[i,i:]
			b[j] -=  factor * b[i]
	x =np.zeros(n)
	for i in range(n-1,-1,-1):
		if A[i,i] == 0:
			raise ValueError("Matrix is singular or nearely singular")
		x[i] = (b[i]-np.dot(A[i,i+1:],x[i+1:]))/A[i,i]

	return x
