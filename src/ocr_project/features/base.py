# src/ocr_project/features/base.py
from abc import ABC, abstractmethod
import numpy as np

class BaseFeatureExtractor(ABC):
    """Abstract base class for feature extraction."""

    @abstractmethod
    def fit(self, images: np.ndarray, labels: np.ndarray = None) -> None:
        """
        Learn any parameters from the dataset, if needed.
        """
        pass

    @abstractmethod
    def transform(self, images: np.ndarray) -> np.ndarray:
        """
        Extract features from images.
        Returns a 2D array (n_samples, n_features).
        """
        pass

    def fit_transform(self, images: np.ndarray, labels: np.ndarray = None) -> np.ndarray:
        """
        Convenience method to fit and transform in one step.
        """
        self.fit(images, labels)
        return self.transform(images)
