import numpy as np
import  math
def compute_chain_rule_gradient(functions: list[str], x: float) -> float:
	"""
	Compute derivative of composite functions using chain rule.
	
	Args:
		functions: List of function names (applied right to left)
		          Available: 'square', 'sin', 'exp', 'log'
		x: Point at which to evaluate derivative
	
	Returns:
		Derivative value at x
	
	Example:
		['sin', 'square'] represents sin(x²)
		['exp', 'sin', 'square'] represents exp(sin(x²))
	"""
	# Your code here
	def square(u):
		return u ** 2
	def d_square(u):
		return 2 * u

	def sin(u):
		return math.sin(u)
	def d_sin(u):
		return math.cos(u)
	
	def exp(u):
		return math.exp(u)
	def d_exp(u):
		return math.exp(u)
	
	def log(u):
		return math.log(u)
	def d_log(u):
		return 1/u
	func_map = {
		"square" : (square ,d_square),
		"sin" : (sin,d_sin),
		"exp" : (exp,d_exp),
		"log" : (log,d_log)
 	}
	values = [x]
	for f in reversed(functions):
		func,_ = func_map[f]
		values.append(func(values[-1]))
	derivative = 1.0
	for f,val in zip(functions,reversed(values[:-1])):
		_,dfunc = func_map[f]
		derivative  *=  dfunc(val)
	return derivative