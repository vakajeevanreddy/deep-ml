
def orthogonal_projection(v, L):
	"""
	Compute the orthogonal projection of vector v onto line L.

	:param v: The vector to be projected
	:param L: The line vector defining the direction of projection
	:return: List representing the projection of v onto L
	"""
	dot_vl = sum(vi * li for vi,li in zip(v,L))
	dot_ll = sum(li*li for li in L)
	if dot_ll ==0:
		raise ValueError("Vector L must be non-zero")
	proj = [(dot_vl/dot_ll)*li for li in L]
	return [round(x, 3)for x in proj]