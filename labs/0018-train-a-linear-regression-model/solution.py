# --- cell 2 ---
import numpy as np

def train(X, y, W, b):
    """
    Train linear regression weights using Mini-Batch Gradient Descent with Adam.
    
    Args:
        X: numpy array of shape (n_samples, n_features) -- standardized features
        y: numpy array of shape (n_samples,) -- standardized targets
        W: numpy array of shape (n_features,) -- initial random weights
        b: float -- initial bias
    
    Returns:
        W: numpy array of shape (n_features,) -- trained weights
        b: float -- trained bias
    """
    X = np.array(X)
    y = np.array(y)
    n = X.shape[0]
    
    epochs = 100
    batch_size = min(32, n)  # Ensures batch size doesn't exceed dataset size
    lr = 0.01
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    # Initialize first and second moments for W and b
    m_W = np.zeros_like(W)
    v_W = np.zeros_like(W)
    m_b = 0.0
    v_b = 0.0
    
    t = 0  # Global timestep counter for Adam bias correction
    
    for _ in range(epochs):
        # Shuffle indices at the start of every epoch
        indices = np.random.permutation(n)
        
        for start in range(0, n, batch_size):
            batch_idx = indices[start:start + batch_size]
            X_batch = X[batch_idx]
            y_batch = y[batch_idx]
            n_batch = X_batch.shape[0]
            
            t += 1  # Increment timestep on every batch update
            
            # Forward pass: predictions on the mini-batch
            preds = X_batch @ W + b
            errors = preds - y_batch
            
            # Compute gradients on the mini-batch
            grad_W = (2 / n_batch) * (X_batch.T @ errors)
            grad_b = (2 / n_batch) * np.sum(errors)
            
            # --- Update Weights (W) with Adam ---
            m_W = beta1 * m_W + (1 - beta1) * grad_W
            v_W = beta2 * v_W + (1 - beta2) * (grad_W ** 2)
            
            m_W_corrected = m_W / (1 - beta1 ** t)
            v_W_corrected = v_W / (1 - beta2 ** t)
            
            W = W - lr * m_W_corrected / (np.sqrt(v_W_corrected) + eps)
            
            # --- Update Bias (b) with Adam ---
            m_b = beta1 * m_b + (1 - beta1) * grad_b
            v_b = beta2 * v_b + (1 - beta2) * (grad_b ** 2)
            
            m_b_corrected = m_b / (1 - beta1 ** t)
            v_b_corrected = v_b / (1 - beta2 ** t)
            
            b = b - lr * m_b_corrected / (np.sqrt(v_b_corrected) + eps)
            
    return W, b