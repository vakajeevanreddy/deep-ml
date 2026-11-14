import numpy as np
from scipy.optimize import linprog


def lad_fit(X, y):
  X = np.asarray(X, dtype=float)
  y = np.asarray(y, dtype=float)
  N, D = X.shape

  c = np.concatenate([np.zeros(D), np.ones(N)]) 
  I_N = np.eye(N)
  A_ub = np.block([[X, -I_N], [-X, -I_N]])
  b_ub = np.concatenate([y, -y])
  bounds = [(None, None)] * D + [(0.0, None)] * N

  res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
  if not res.success:
    raise RuntimeError(f"Linear program failed to converge: {res.message}")

  return res.x[:D]  


def huber_fit(X, y, delta, n_iter):
  X = np.asarray(X, dtype=float)
  y = np.asarray(y, dtype=float)
  W, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

  for _ in range(n_iter):
    r = y - X @ W 
    abs_r = np.abs(r)

    S = np.ones_like(abs_r)
    mask = abs_r > delta
    S[mask] = delta / abs_r[mask]

    Sqrt_S = np.sqrt(S)
    X_weighted = X * Sqrt_S[:, np.newaxis] 
    y_weighted = y * Sqrt_S  

    W, _, _, _ = np.linalg.lstsq(X_weighted, y_weighted, rcond=None)

  return W  