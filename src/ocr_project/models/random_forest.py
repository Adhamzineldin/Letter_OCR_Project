# src/ocr_project/models/random_forest.py
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from .base_model import BaseModel

class RandomForest(BaseModel):
    """Random forest classifier wrapper."""

    def __init__(self, n_estimators: int = 100, max_depth: int = None, random_state: int = 42):
        self.model = RandomForestClassifier(
                n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
