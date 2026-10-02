import numpy as np

def compute_partial_derivatives(func_name: str, point: tuple[float, ...]) -> tuple[float, ...]:
	"""
	Compute partial derivatives of multivariable functions.
	
	Args:
		func_name: Function identifier
			'poly2d': f(x,y) = x²y + xy²
			'exp_sum': f(x,y) = e^(x+y)
			'product_sin': f(x,y) = x·sin(y)
			'poly3d': f(x,y,z) = x²y + yz²
			'squared_error': f(x,y) = (x-y)²
		point: Point (x, y) or (x, y, z) at which to evaluate
	
	Returns:
		Tuple of partial derivatives (∂f/∂x, ∂f/∂y, ...) at point
	"""
	# Your code here
	if func_name == "poly2d":
		x,y = point
		dfdx = 2 * x * y + y ** 2
		dfdy = x ** 2 + 2 * x * y
		return(dfdx,dfdy)
	elif func_name == "exp_sum":
		x,y = point
		val = np.exp(x + y)
		return (val,val) 
	elif func_name == "product_sin":
		x,y = point
		dfdx = np.sin(y)
		dfdy = x * np.cos(y)
		return (dfdx,dfdy)
	elif func_name == "poly3d":
		x,y,z = point
		dfdx = 2*x*y
		dfdy = x ** 2 + z ** 2
		dfdz = 2 * y *z
		return (dfdx,dfdy,dfdz)
	elif func_name == "squared_error":
		x,y = point
		dfdx = 2 * (x-y)
		dfdy = -2*(x-y)
		return (dfdx,dfdy)
	else:
    		raise ValueError(f"Unknown function name: {func_name}")
