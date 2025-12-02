# src/ocr_project/models/base_model.py
from abc import ABC, abstractmethod
import numpy as np

class BaseModel(ABC):
    """Abstract base class for all models."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model on features X and labels y."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for input features X."""
        pass

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return accuracy of the model."""
        preds = self.predict(X)
        return (preds == y).mean()
