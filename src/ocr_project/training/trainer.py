# src/ocr_project/training/trainer.py
import joblib
from typing import Any
from sklearn.base import ClassifierMixin

class Trainer:
    """Trainer for scikit-learn compatible models."""

    def __init__(self, model: ClassifierMixin):
        self.model = model

    def train(self, X_train, y_train) -> None:
        """
        Train the model.
        """
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test) -> float:
        """
        Evaluate the model accuracy.
        """
        return self.model.score(X_test, y_test)

    def predict(self, X) -> Any:
        """
        Make predictions.
        """
        return self.model.predict(X)

    def save_model(self, path: str) -> None:
        """
        Save the trained model to disk.
        """
        joblib.dump(self.model, path)

    @staticmethod
    def load_model(path: str) -> Any:
        """
        Load a saved model from disk.
        """
        return joblib.load(path)
