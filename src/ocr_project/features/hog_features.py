# src/ocr_project/features/hog_features.py
import numpy as np
from skimage.feature import hog
from .base import BaseFeatureExtractor

class HOGFeatureExtractor(BaseFeatureExtractor):
    """Extract HOG features from images."""

    def __init__(self, pixels_per_cell=(8, 8), cells_per_block=(2, 2), orientations=9):
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.orientations = orientations

    def fit(self, images: np.ndarray, labels: np.ndarray = None) -> None:
        # HOG doesn't require fitting
        pass

    def transform(self, images: np.ndarray) -> np.ndarray:
        features = []
        for img in images:
            feat = hog(
                    img,
                    pixels_per_cell=self.pixels_per_cell,
                    cells_per_block=self.cells_per_block,
                    orientations=self.orientations,
                    block_norm='L2-Hys'
            )
            features.append(feat)
        return np.array(features, dtype=np.float32)
