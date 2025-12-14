# src/ocr_project/training/cross_validator.py
from typing import Dict, Any, Optional
import numpy as np
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold
from sklearn.base import ClassifierMixin


class CrossValidator:
    """
    Cross-validation evaluator for scikit-learn compatible models.
    
    Uses stratified k-fold cross-validation to ensure balanced class distribution
    across folds, which is important for multi-class classification tasks.
    """

    def __init__(
        self,
        model: ClassifierMixin,
        cv: int = 5,
        scoring: str = "accuracy",
        return_train_score: bool = False,
        random_state: Optional[int] = None,
    ):
        """
        Initialize the cross-validator.

        Args:
            model: A scikit-learn compatible classifier
            cv: Number of cross-validation folds (default: 5)
            scoring: Scoring metric to use (default: "accuracy")
            return_train_score: Whether to return training scores (default: False)
            random_state: Random state for reproducibility (default: None)
        """
        self.model = model
        self.cv = cv
        self.scoring = scoring
        self.return_train_score = return_train_score
        self.random_state = random_state
        self.cv_results_: Optional[Dict[str, Any]] = None

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Perform cross-validation on the model.

        Args:
            X: Feature matrix
            y: Target labels

        Returns:
            Dictionary containing:
                - mean_test_score: Mean test score across folds
                - std_test_score: Standard deviation of test scores
                - test_scores: List of test scores for each fold
                - mean_train_score: Mean train score (if return_train_score=True)
                - std_train_score: Std train score (if return_train_score=True)
                - train_scores: List of train scores (if return_train_score=True)
        """
        # Use StratifiedKFold to ensure balanced class distribution
        cv_splitter = StratifiedKFold(
            n_splits=self.cv,
            shuffle=True,
            random_state=self.random_state,
        )

        # Perform cross-validation
        cv_results = cross_validate(
            self.model,
            X,
            y,
            cv=cv_splitter,
            scoring=self.scoring,
            return_train_score=self.return_train_score,
            n_jobs=-1,  # Use all available CPU cores
        )

        # Extract results
        test_scores = cv_results["test_score"]
        result = {
            "mean_test_score": float(np.mean(test_scores)),
            "std_test_score": float(np.std(test_scores)),
            "test_scores": test_scores.tolist(),
        }

        if self.return_train_score:
            train_scores = cv_results["train_score"]
            result.update({
                "mean_train_score": float(np.mean(train_scores)),
                "std_train_score": float(np.std(train_scores)),
                "train_scores": train_scores.tolist(),
            })

        self.cv_results_ = result
        return result

    def get_accuracy(self) -> float:
        """
        Get the mean cross-validation accuracy.

        Returns:
            Mean test accuracy across all folds

        Raises:
            RuntimeError: If cross_validate() hasn't been called yet
        """
        if self.cv_results_ is None:
            raise RuntimeError(
                "Cross-validation not performed yet. Call cross_validate() first."
            )
        return self.cv_results_["mean_test_score"]

    def get_accuracy_std(self) -> float:
        """
        Get the standard deviation of cross-validation accuracy.

        Returns:
            Standard deviation of test accuracy across all folds

        Raises:
            RuntimeError: If cross_validate() hasn't been called yet
        """
        if self.cv_results_ is None:
            raise RuntimeError(
                "Cross-validation not performed yet. Call cross_validate() first."
            )
        return self.cv_results_["std_test_score"]

    def print_summary(self) -> None:
        """
        Print a summary of cross-validation results.
        """
        if self.cv_results_ is None:
            print("No cross-validation results available.")
            return

        print(f"Cross-Validation Results ({self.cv} folds):")
        print(f"  Mean Accuracy: {self.cv_results_['mean_test_score']:.4f}")
        print(f"  Std Accuracy:  {self.cv_results_['std_test_score']:.4f}")
        print(f"  Fold Accuracies: {[f'{s:.4f}' for s in self.cv_results_['test_scores']]}")

        if self.return_train_score and "mean_train_score" in self.cv_results_:
            print(f"  Mean Train Accuracy: {self.cv_results_['mean_train_score']:.4f}")
            print(f"  Std Train Accuracy:  {self.cv_results_['std_train_score']:.4f}")

