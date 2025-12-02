# src/ocr_project/features/pca_features.py
import numpy as np
from sklearn.decomposition import PCA
from .base import BaseFeatureExtractor

class PCAFeatures(BaseFeatureExtractor):
    """PCA-based feature extraction."""

    def __init__(self, n_components: int = 50):
        self.n_components = n_components
        self.pca: PCA | None = None

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> None:
        """
        Fit PCA on the dataset X.
        X: shape (n_samples, n_features)
        """
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform dataset X into PCA feature space.
        """
        if self.pca is None:
            raise RuntimeError("PCA model is not fitted. Call fit() first.")
        return self.pca.transform(X)

    def fit_transform(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        """Convenience method: fit and transform in one step."""
        self.fit(X, y)
        return self.transform(X)
