import numpy as np

def suggest_rank(delta_W: np.ndarray, energy_threshold: float) -> int:
	"""
	Return the smallest rank k such that the top-k singular values of delta_W
	capture at least `energy_threshold` of the total squared-singular-value energy.
	"""
	# Your code here
	if not np.any(delta_W):
		return 0
	s = np.linalg.svd(delta_W,compute_uv = False)
	energy = s ** 2
	total_energy = np.sum(energy)
	if total_energy == 0:
		return  0
	cummulative_energy = np.cumsum(energy)
	energy_ratios = cummulative_energy / total_energy
	k = np.searchsorted(energy_ratios , energy_threshold) + 1
	return k