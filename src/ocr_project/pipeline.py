from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np

from . import config
from .io.dataset_loader import EmnistLoader
from .io.persistence import save_model_artifact
from .models.decision_tree import DecisionTree
from .models.random_forest import RandomForest
from .preprocess.transformer import Transformer
from .training.trainer import Trainer
from .training.cross_validator import CrossValidator
from .evaluation.evaluator import Evaluator
from .features.base import BaseFeatureExtractor
from .features.hog_pca import HOGPCAFeatures


@dataclass
class ModelResult:
    name: str
    model: Any
    accuracy: float  # CV mean accuracy (if CV used) or test accuracy (if train/test split)
    accuracy_std: float = 0.0  # Standard deviation for cross-validation
    test_accuracy: float = 0.0  # Accuracy on holdout test set


@dataclass
class OCRPipeline:
    """
    End-to-end pipeline for training and evaluating OCR models on EMNIST letters.

    Responsibilities:
    - Load train/test CSVs (or combined data for cross-validation)
    - Preprocess images (flatten + normalize)
    - Train Decision Tree and Random Forest
    - Evaluate each model using cross-validation
    - Save trained models as artifacts for the GUI
    """

    train_csv: Path = config.EMNIST_LETTERS_TRAIN
    test_csv: Path = config.EMNIST_LETTERS_TEST
    # Save artifacts in a top-level "artifacts" directory so the Streamlit app
    # can load them directly (see app.py::load_models).
    # NOTE: this intentionally differs from older versions that wrote into
    # data/processed – if you have old models there, retrain with this pipeline.
    artifacts_dir: Path = Path(__file__).resolve().parents[2] / "artifacts"

    # Optional controls
    # Use the full dataset by default for better accuracy; set these to small
    # integers in notebooks/scripts if you want a faster debug run.
    train_limit: Optional[int] = None
    test_limit: Optional[int] = None
    use_augmentation: bool = False  # kept for API compatibility; not used now
    feature_extractor: Optional[BaseFeatureExtractor] = None

    # Cross-validation settings
    use_cross_validation: bool = True  # Use cross-validation on training data
    cv_folds: int = 5  # Number of cross-validation folds
    random_state: Optional[int] = config.RANDOM_SEED  # Random state for reproducibility

    # Training and test data (always separate - test is holdout)
    _X_train: np.ndarray | None = field(init=False, default=None)
    _y_train: np.ndarray | None = field(init=False, default=None)
    _X_test: np.ndarray | None = field(init=False, default=None)
    _y_test: np.ndarray | None = field(init=False, default=None)
    results: Dict[str, ModelResult] = field(init=False, default_factory=dict)

    def load_data(self) -> None:
        """
        Load EMNIST data and preprocess it.
        
        Always loads separate train and test splits. Test data is kept as a holdout set.
        Cross-validation is performed only on training data.
        """
        loader = EmnistLoader(self.train_csv, self.test_csv)
        train_split = loader.load_train(limit=self.train_limit)
        test_split = loader.load_test(limit=self.test_limit)

        # Preprocess: flatten and normalize
        X_train = Transformer.normalize(Transformer.flatten(train_split.images))
        X_test = Transformer.normalize(Transformer.flatten(test_split.images))

        # Optional feature extraction stage (e.g. HOG or PCA)
        # Fit on training data only, then transform both train and test
        if self.feature_extractor is not None:
            X_train = self.feature_extractor.fit_transform(X_train, train_split.labels)
            X_test = self.feature_extractor.transform(X_test)

        self._X_train = X_train
        self._y_train = train_split.labels
        self._X_test = X_test
        self._y_test = test_split.labels

    def _ensure_data_loaded(self) -> None:
        if self._X_train is None or self._X_test is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")

    def _train_single_model(self, name: str, estimator) -> ModelResult:
        """
        Train and evaluate a single sklearn-style estimator.
        
        If use_cross_validation is True:
        - Performs cross-validation on training data only
        - Trains final model on all training data
        - Evaluates final model on holdout test set
        
        Otherwise (legacy mode):
        - Trains on training data
        - Evaluates on test data
        """
        self._ensure_data_loaded()
        X_train, y_train = self._X_train, self._y_train
        X_test, y_test = self._X_test, self._y_test
        assert X_train is not None and y_train is not None
        assert X_test is not None and y_test is not None

        if self.use_cross_validation:
            # Perform cross-validation on training data only
            cv_validator = CrossValidator(
                model=estimator,
                cv=self.cv_folds,
                random_state=self.random_state,
            )
            cv_results = cv_validator.cross_validate(X_train, y_train)
            
            # Train final model on all training data
            trainer = Trainer(estimator)
            trainer.train(X_train, y_train)
            
            # Evaluate final model on holdout test set
            evaluator = Evaluator(trainer.model)
            test_acc, _ = evaluator.evaluate(X_test, y_test)
            
            return ModelResult(
                name=name,
                model=trainer.model,
                accuracy=cv_results["mean_test_score"],  # CV mean accuracy
                accuracy_std=cv_results["std_test_score"],  # CV std
                test_accuracy=float(test_acc),  # Holdout test accuracy
            )
        else:
            # Legacy mode: simple train/test split
            trainer = Trainer(estimator)
            trainer.train(X_train, y_train)
            evaluator = Evaluator(trainer.model)
            acc, _ = evaluator.evaluate(X_test, y_test)

            return ModelResult(
                name=name,
                model=trainer.model,
                accuracy=float(acc),
                test_accuracy=float(acc),  # Same as accuracy in legacy mode
            )

    def train_and_evaluate(self) -> None:
        """
        Train all configured models (Decision Tree, Random Forest) and store results.
        """
        eval_method = "Cross-Validation (on training data)" if self.use_cross_validation else "Train/Test Split"
        print(f"Evaluating models using: {eval_method}")
        if self.use_cross_validation:
            print(f"Number of CV folds: {self.cv_folds}")
            print("Test data is kept as a separate holdout set.")
        
        # Decision Tree
        print("\nTraining Decision Tree...")
        dt_result = self._train_single_model(
            "decision_tree",
            # Match the compare_models experiment: DecisionTreeClassifier(max_depth=20)
            DecisionTree(max_depth=20).model,
        )
        self.results[dt_result.name] = dt_result
        if self.use_cross_validation:
            print(f"  CV Mean Accuracy: {dt_result.accuracy:.4f} (+/- {dt_result.accuracy_std:.4f})")
            print(f"  Test Set Accuracy: {dt_result.test_accuracy:.4f}")
        else:
            print(f"  Test Accuracy: {dt_result.accuracy:.4f}")

        # Random Forest
        print("\nTraining Random Forest...")
        rf_result = self._train_single_model(
            "random_forest",
            # Match the compare_models experiment: RandomForestClassifier(n_estimators=100)
            RandomForest(n_estimators=100, max_depth=None).model,
        )
        self.results[rf_result.name] = rf_result
        if self.use_cross_validation:
            print(f"  CV Mean Accuracy: {rf_result.accuracy:.4f} (+/- {rf_result.accuracy_std:.4f})")
            print(f"  Test Set Accuracy: {rf_result.test_accuracy:.4f}")
        else:
            print(f"  Test Accuracy: {rf_result.accuracy:.4f}")

    def save_artifacts(self) -> None:
        """
        Persist trained models (and their accuracies) to disk for use by the GUI.
        """
        if not self.results:
            raise RuntimeError("No trained models to save. Call train_and_evaluate() first.")

        self.artifacts_dir.mkdir(exist_ok=True, parents=True)

        for key, result in self.results.items():
            artifact_path = self.artifacts_dir / f"{key}.pkl"
            metadata = {}
            if self.feature_extractor is not None:
                # Persist the feature extractor so the app can apply the same
                # transformation at inference time.
                metadata["feature_extractor"] = self.feature_extractor
            
            # Save CV accuracy as the main accuracy, and test accuracy separately
            metadata["cv_accuracy"] = result.accuracy
            metadata["cv_accuracy_std"] = result.accuracy_std
            metadata["test_accuracy"] = result.test_accuracy
            
            save_model_artifact(
                artifact_path,
                model=result.model,
                accuracy=result.accuracy,  # CV mean accuracy (for backward compatibility)
                metadata=metadata or None,
            )

    def run(self) -> Dict[str, ModelResult]:
        """
        Convenience method: fully run the pipeline and return results.
        """
        self.load_data()
        self.train_and_evaluate()
        self.save_artifacts()
        return self.results


def run_default_pipeline() -> Dict[str, ModelResult]:
    """
    Helper for scripts and notebooks: run the **default HOG+PCA pipeline**.

    This uses the combined HOG + PCA feature extractor and saves artifacts
    into the top-level ``artifacts/`` directory, which the Streamlit app
    treats as the "HOG + PCA" feature option.
    """
    pipeline = OCRPipeline(feature_extractor=HOGPCAFeatures())
    return pipeline.run()


