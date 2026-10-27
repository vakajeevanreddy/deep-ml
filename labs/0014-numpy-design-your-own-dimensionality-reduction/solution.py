# --- cell 1 ---
import numpy as np

class MyReducer:
    """
    Implement your own dimensionality reduction to 10 dimensions.
    
    Your goal: Project high-dimensional data to 10 dimensions while
    preserving structure for classification.
    
    A k-NN classifier will be trained on your reduced data to evaluate quality.
    """
    
    def __init__(self):
        self.n_components = 10
        self.mean_ = None
        self.components_ = None
        self.singular_values_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None      
        
    def fit(self, X):
        """
        Learn the reduction from training data.
        
        Args:
            X: Training data, shape (n_samples, n_features)
        
        Returns:
            self
        """
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape
        
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        U, S, vt = np.linalg.svd(X_centered, full_matrices=False)
        self.singular_values_ = S
        eigenvalues = (S ** 2) / (n_samples - 1)
        self.explained_variance_ = eigenvalues[:self.n_components]
        total_variance = np.sum(eigenvalues)
        if total_variance > 0:
            self.explained_variance_ratio_ = self.explained_variance_ / total_variance
        else:
            self.explained_variance_ratio_ = np.zeros(self.n_components) 
        self.components_ = vt[:self.n_components, :]
        return self
    
    def transform(self, X):
        """
        Apply the learned reduction to data.
        
        Args:
            X: Data to transform, shape (n_samples, n_features)
        
        Returns:
            X_reduced: shape (n_samples, 10)
        """
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("PCA instance is not fitted yet. Call 'fit' before 'transform'")
        X = np.asarray(X, dtype=np.float64)
        X_centered = X - self.mean_
        X_reduced = np.dot(X_centered, self.components_.T)
        return X_reduced

    def fit_transform(self, X):
        """Fit and transform in one step."""
        return self.fit(X).transform(X)