from typing import Callable
import  numpy as np
def compute_hessian(f: Callable[[list[float]], float], point: list[float], h: float = 1e-5) -> list[list[float]]:
	"""
	Compute the Hessian matrix of function f at the given point using finite differences.
	
	Args:
		f: A scalar function that takes a list of floats and returns a float
		point: The point at which to compute the Hessian (list of coordinates)
		h: Step size for finite differences (default: 1e-5)
		
	Returns:
		The Hessian matrix as a list of lists (n x n where n = len(point))
	"""
	# Your code here
	x = np.array(point,dtype = float)
	n = len(x)
	H = np.zeros((n,n))
	f0 = f(x)
	for i in range(n):
		for j in range(n):
			x_ij1 = x.copy()
			x_ij2 = x.copy()
			x_ij3 = x.copy()
			x_ij4 = x.copy()

			x_ij1[i] += h ; x_ij1[j] += h 
			x_ij2[i] += h ; x_ij2[j] -= h 
			x_ij3[i] -= h ; x_ij3[j] += h 
			x_ij4[i] -= h ; x_ij4[j] -= h 

			H[i,j] = (f(x_ij1) - f(x_ij2) - f(x_ij3) + f(x_ij4)) / (4 * h ** 2)
	return H.tolist()