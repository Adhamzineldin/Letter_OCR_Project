import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np


class Visualizer:
    """
    Utility class for plotting evaluation results.

    This deliberately separates plotting from evaluation logic so that
    notebooks, scripts, and the GUI can all reuse the same helpers.
    """

    @staticmethod
    def plot_confusion_matrix(
        y_true,
        y_pred,
        labels=None,
        figsize=(10, 8),
        title: str = "Confusion Matrix",
        cmap: str = "Blues",
        normalize: bool = False,
    ) -> None:
        """
        Plot a confusion matrix given ground truth and predictions.

        Args:
            y_true: 1D array-like of true labels.
            y_pred: 1D array-like of predicted labels.
            labels: Optional list of label names / indices for axes.
            figsize: Figure size.
            title: Plot title.
            cmap: Matplotlib colormap name.
            normalize: If True, normalize rows to sum to 1.
        """
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        if normalize:
            with np.errstate(all="ignore"):
                cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
                cm = np.nan_to_num(cm)

        plt.figure(figsize=figsize)
        sns.heatmap(
            cm,
            annot=True,
            fmt=".2f" if normalize else "d",
            cmap=cmap,
            xticklabels=labels,
            yticklabels=labels,
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(title)
        plt.tight_layout()
        plt.show()



