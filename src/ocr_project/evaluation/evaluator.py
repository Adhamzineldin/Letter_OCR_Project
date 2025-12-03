# src/ocr_project/evaluation/evaluator.py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class Evaluator:
    """Evaluate model performance."""

    def __init__(self, model):
        self.model = model

    def evaluate(self, X, y):
        """
        Compute accuracy and classification report.
        """
        y_pred = self.model.predict(X)
        acc = accuracy_score(y, y_pred)
        # zero_division=0 avoids noisy warnings when some labels
        # are missing in the evaluation subset.
        report = classification_report(y, y_pred, zero_division=0)
        return acc, report

    def plot_confusion_matrix(self, X, y, figsize=(10, 8)):
        """
        Plot a confusion matrix.
        """
        y_pred = self.model.predict(X)
        cm = confusion_matrix(y, y_pred)
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()
