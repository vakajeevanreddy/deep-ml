import numpy as np

def gradient_direction_magnitude(gradient: list) -> dict:
	"""
	Calculate the magnitude and direction of a gradient vector.
	
	Args:
		gradient: A list representing the gradient vector
	
	Returns:
		Dictionary containing:
		- magnitude: The L2 norm of the gradient
		- direction: Unit vector in direction of steepest ascent
		- descent_direction: Unit vector in direction of steepest descent
	"""
	# Your code here
	grad = np.array(gradient, dtype = float)
	magnitude = np.linalg.norm(grad)
	if magnitude == 0:
		direction = [0.0 for _ in gradient]
		descent_direction = [0.0 for _ in gradient]
	else:
		direction = [g / magnitude for g in gradient]
		descent_direction = [-g / magnitude for g in gradient]
	return {
		"magnitude" : magnitude,
		"direction" : direction,
		"descent_direction" : descent_direction
	}