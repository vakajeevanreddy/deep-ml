import numpy as np
def activation_derivatives(x: float) -> dict[str, float]:
	"""
	Compute the derivatives of Sigmoid, Tanh, and ReLU at a given point x.
	
	Args:
		x: Input value
		
	Returns:
		Dictionary with keys 'sigmoid', 'tanh', 'relu' and their derivative values
	"""
	# Your code here
	sigmoid = 1 / (1+np.exp(-x))
	d_sigmoid = sigmoid * (1-sigmoid)

	tanh = np.tanh(x)
	d_tanh = 1 - tanh ** 2

	d_relu = 1.0 if x > 0 else 0.0
	return {
		"sigmoid" : d_sigmoid,
		"tanh" : d_tanh,
		"relu" : d_relu
	}