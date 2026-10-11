import numpy as np

def classify_critical_point(hessian: np.ndarray, tol: float = 1e-10):
    H = np.array(hessian,dtype = float)
    eigenvalues = np.linalg.eigvals(H)
    Postive = all(ev  > tol for ev in eigenvalues)
    Negative = all(ev  < -tol for ev in eigenvalues)
    zero_like = any(abs(ev) <= tol for ev in eigenvalues)
    if zero_like:
        return  None
    elif Postive:
        return -1
    elif Negative:
        return +1
    else:
        return 0