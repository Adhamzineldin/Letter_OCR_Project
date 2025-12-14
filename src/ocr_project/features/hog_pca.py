from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

from .base import BaseFeatureExtractor
from .hog_features import HOGFeatureExtractor


class HOGPCAFeatures(BaseFeatureExtractor):
    """
    Feature extractor that combines:
    - raw flattened pixels
    - HOG descriptors computed from the 28x28 images
    and then runs PCA on the concatenated vector.
    """

    def __init__(
        self,
        n_components: int = 100,
        pixels_per_cell: tuple[int, int] = (8, 8),
        cells_per_block: tuple[int, int] = (2, 2),
        orientations: int = 9,
    ):
        self.n_components = n_components
        self.hog = HOGFeatureExtractor(
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            orientations=orientations,
        )
        self.pca: PCA | None = None

    def fit(self, X: np.ndarray, labels: np.ndarray | None = None) -> None:
        """
        X is expected to be (n_samples, 28*28) flattened, normalized images.
        """
        # HOG over each sample
        hog_feats = self.hog.transform(X)
        combined = np.hstack([X, hog_feats])

        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(combined)

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.pca is None:
            raise RuntimeError("PCA model is not fitted. Call fit() first.")

        hog_feats = self.hog.transform(X)
        combined = np.hstack([X, hog_feats])
        return self.pca.transform(combined)










