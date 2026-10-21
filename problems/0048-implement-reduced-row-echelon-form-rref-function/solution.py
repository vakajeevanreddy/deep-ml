import numpy as np

def rref(matrix):
	A = np.array(matrix,dtype = float)
	m,n = A.shape
	row = 0
	for col in range(n):
		if row >= m:
			break
		pivot = row
		while  pivot < m and abs(A[pivot,col]) < 1e-10:
			pivot += 1
		if pivot == m:
			continue
		A[[row,pivot]] = A[[pivot,row]]
		pivot_val = A[row,col]
		A[row] /= pivot_val
		for i in range(m):
			if i != row:
				factor = A[i,col]
				A[i] -= factor * A[row]
		row += 1
	A[abs(A) < 1e-10] = 0.0
	return  A.tolist()

