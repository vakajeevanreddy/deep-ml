import numpy as np

def cosine_similarity(v1, v2):
	"""
	Calculate the cosine_similarity of two vectors.
	Args:
		vec1 (numpy.ndarray): 1D array representing the first vector.
		vec2 (numpy.ndarray): 1D array representing the second vector.
	Returns:
		The cosine_similarity of the two vectors.
	"""
	# Implement your code here
	v1 = v1.flatten()
	v2 = v2.flatten()

	dot_prod = np.dot(v1,v2)
	norm_v1 = np.linalg.norm(v1)
	norm_v2 = np.linalg.norm(v2)
	if norm_v1 == 0 or norm_v2 == 0:
		raise ValueError("Cosine similarity is undefined for zero vectors.")
	return float(dot_prod / (norm_v1 * norm_v2))