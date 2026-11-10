import numpy as np

def partitioned_inverse(M, k):
    E = M[:k, :k]
    F = M[:k, k:]
    G = M[k:, :k]
    H = M[k:, k:]
    H_inv_G = np.linalg.solve(H,G)
    F_H_inv = np.linalg.solve(H.T , F.T).T
    schur = E - F @ H_inv_G
    schur_inv = np.linalg.inv(schur)

    Top_left = schur_inv
    Top_right = -schur_inv @ F_H_inv
    Bottom_left = -H_inv_G @ schur_inv
    Bottom_right = np.linalg.inv(H) + H_inv_G @ schur_inv @ F_H_inv
    return np.block([[Top_left,Top_right],[Bottom_left,Bottom_right]])

def block_slogdet(M, k):
    E = M[:k,:k]
    F = M[:k,k:]
    G = M[k:,:k]
    H = M[k:, k:]

    H_inv_G = np.linalg.solve(H,G)
    schur = E - F @ H_inv_G
    sign_H , lodget_H = np.linalg.slogdet(H)
    sign_S , lodget_S = np.linalg.slogdet(schur)
    sign = sign_H * sign_S
    loagabsdet = lodget_H +  lodget_S
    return float(sign) , float(loagabsdet)

def bordered_inverse(A_inv, b, c):
    b = np.asarray(b).reshape(-1,1)
    c_val = float(np.atleast_2d(c)[0, 0])
    A_inv_b = A_inv @ b 
    S = c_val - float((b.T @ A_inv_b)[0,0])
    if abs(S) < 1e-12:
        raise ValueError("Schur complement scalar is too close to zero")
    inv_S = 1.0/S 
    Top_left = A_inv + inv_S * (A_inv_b @ A_inv_b.T)
    Top_right = -inv_S * A_inv_b
    Bottom_left = Top_right.T
    Bottom_right = np.array([[inv_S]])
    return np.block([
        [Top_left, Top_right],
        [Bottom_left, Bottom_right]
    ])