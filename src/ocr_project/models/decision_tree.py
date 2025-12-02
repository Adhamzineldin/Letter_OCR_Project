# src/ocr_project/models/decision_tree.py
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from .base_model import BaseModel

class DecisionTree(BaseModel):
    """Decision tree classifier wrapper."""

    def __init__(self, max_depth: int = None, random_state: int = 42):
        self.model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
