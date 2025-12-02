# src/ocr_project/experiments/compare_models.py
from typing import List, Tuple
import matplotlib.pyplot as plt

from ocr_project.training.trainer import Trainer
from ocr_project.evaluation.evaluator import Evaluator
from ocr_project.io.dataset_loader import EmnistLoader
from ocr_project.preprocess.transformer import Transformer


from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

class ModelComparison:
    """Run and compare multiple models on the same dataset."""

    def __init__(self, models: List[Tuple[str, object]]):
        """
        Args:
            models: List of tuples (model_name, sklearn_model_instance)
        """
        self.models = models
        self.results = {}

    def run(
            self,
            X_train,
            y_train,
            X_test,
            y_test
    ):
        """Train and evaluate each model."""
        for name, model in self.models:
            print(f"Training {name}...")
            trainer = Trainer(model)
            trainer.train(X_train, y_train)

            evaluator = Evaluator(trainer.model)
            acc, report = evaluator.evaluate(X_test, y_test)
            print(f"{name} Accuracy: {acc:.4f}")
            print(report)

            self.results[name] = {"accuracy": acc, "report": report}

    def plot_results(self):
        """Plot a comparison of accuracies."""
        names = list(self.results.keys())
        accuracies = [self.results[name]["accuracy"] for name in names]

        plt.figure(figsize=(8, 5))
        plt.bar(names, accuracies, color="skyblue")
        plt.ylim(0, 1)
        plt.ylabel("Accuracy")
        plt.title("Model Comparison")
        plt.show()


if __name__ == "__main__":
    # Load data
    loader = EmnistLoader("../../../data/raw/emnist-letters-train.csv", "../../../data/raw/emnist-letters-test.csv")
    train_split = loader.load_train()
    test_split = loader.load_test()

    # Preprocess
    X_train = Transformer.normalize(Transformer.flatten(train_split.images))
    y_train = train_split.labels
    X_test = Transformer.normalize(Transformer.flatten(test_split.images))
    y_test = test_split.labels

    # Define models
    models_to_compare = [
            ("DecisionTree", DecisionTreeClassifier(max_depth=20)),
            ("RandomForest", RandomForestClassifier(n_estimators=100)),
    ]

    # Run comparison
    comparison = ModelComparison(models_to_compare)
    comparison.run(X_train, y_train, X_test, y_test)
    comparison.plot_results()
