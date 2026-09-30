import numpy as np

def calculate_correlation_matrix(X, Y=None):
	# Your code here
	X = np.asarray(X)
	if Y is None:
		Y = X
	else:
		Y = np.asarray(Y)
	X_centered = X - np.mean(X,axis = 0)
	Y_centered = Y - np.mean(Y, axis=0)

	cov = X_centered.T @ Y_centered / (X.shape[0] - 1)
	X_std = np.std(X,axis = 0, ddof = 1)
	Y_std = np.std(Y,axis = 0 , ddof = 1)
	denom = np.outer(X_std,Y_std)
	corr_matrix = cov/denom
	return  corr_matrix