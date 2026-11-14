# --- cell 1 ---
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class MyReducer:
    def __init__(self, n_components=10):
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.n_components, random_state=42)
        
    def _flatten(self, X):
        if X.ndim == 3:
            return X.reshape(X.shape[0], -1)
        return X
        
    def fit(self, X, y=None):
        X_flat = self._flatten(X)
        X_scaled = self.scaler.fit_transform(X_flat.astype(np.float64))
        self.pca.fit(X_scaled)
        return self
        
    def transform(self, X):
        X_flat = self._flatten(X)
        X_scaled = self.scaler.transform(X_flat.astype(np.float64))
        return self.pca.transform(X_scaled)
        
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)