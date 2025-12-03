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
from .preprocess.augmentations import Augmentations
from .training.trainer import Trainer
from .evaluation.evaluator import Evaluator
from .features.base import BaseFeatureExtractor


@dataclass
class ModelResult:
    name: str
    model: Any
    accuracy: float


@dataclass
class OCRPipeline:
    """
    End-to-end pipeline for training and evaluating OCR models on EMNIST letters.

    Responsibilities:
    - Load train/test CSVs
    - Preprocess images (flatten + normalize)
    - Train Decision Tree and Random Forest
    - Evaluate each model
    - Save trained models as artifacts for the GUI
    """

    train_csv: Path = config.EMNIST_LETTERS_TRAIN
    test_csv: Path = config.EMNIST_LETTERS_TEST
    artifacts_dir: Path = Path("artifacts")

    # Optional controls
    train_limit: Optional[int] = 20000
    test_limit: Optional[int] = 5000
    use_augmentation: bool = False
    feature_extractor: Optional[BaseFeatureExtractor] = None

    _X_train: np.ndarray | None = field(init=False, default=None)
    _y_train: np.ndarray | None = field(init=False, default=None)
    _X_test: np.ndarray | None = field(init=False, default=None)
    _y_test: np.ndarray | None = field(init=False, default=None)
    results: Dict[str, ModelResult] = field(init=False, default_factory=dict)

    def load_data(self) -> None:
        """Load EMNIST train and test splits from CSV files."""
        loader = EmnistLoader(self.train_csv, self.test_csv)
        train_split = loader.load_train(limit=self.train_limit)
        test_split = loader.load_test(limit=self.test_limit)

        # Start from float images [0, 1] for potential augmentation
        train_images = train_split.images.astype(np.float32) / 255.0
        test_images = test_split.images.astype(np.float32) / 255.0

        if self.use_augmentation:
            train_images = Augmentations.augment(train_images)

        # Back to [0, 255] uint8 so we can reuse Transformer.normalize
        train_images_uint8 = np.clip(train_images * 255.0, 0, 255).astype(np.uint8)
        test_images_uint8 = np.clip(test_images * 255.0, 0, 255).astype(np.uint8)

        X_train = Transformer.normalize(Transformer.flatten(train_images_uint8))
        X_test = Transformer.normalize(Transformer.flatten(test_images_uint8))

        # Optional feature extraction stage (e.g. HOG or PCA)
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
        Train and evaluate a single sklearn-style estimator using Trainer/Evaluator.
        """
        self._ensure_data_loaded()
        X_train, y_train = self._X_train, self._y_train
        X_test, y_test = self._X_test, self._y_test
        assert X_train is not None and y_train is not None
        assert X_test is not None and y_test is not None

        trainer = Trainer(estimator)
        trainer.train(X_train, y_train)
        evaluator = Evaluator(trainer.model)
        acc, _ = evaluator.evaluate(X_test, y_test)

        return ModelResult(name=name, model=trainer.model, accuracy=float(acc))

    def train_and_evaluate(self) -> None:
        """
        Train all configured models (Decision Tree, Random Forest) and store results.
        """
        # Decision Tree
        dt_result = self._train_single_model(
            "decision_tree",
            DecisionTree(max_depth=20).model,
        )
        self.results[dt_result.name] = dt_result

        # Random Forest
        rf_result = self._train_single_model(
            "random_forest",
            RandomForest(n_estimators=200, max_depth=None).model,
        )
        self.results[rf_result.name] = rf_result

    def save_artifacts(self) -> None:
        """
        Persist trained models (and their accuracies) to disk for use by the GUI.
        """
        if not self.results:
            raise RuntimeError("No trained models to save. Call train_and_evaluate() first.")

        self.artifacts_dir.mkdir(exist_ok=True, parents=True)

        for key, result in self.results.items():
            artifact_path = self.artifacts_dir / f"{key}.pkl"
            save_model_artifact(
                artifact_path,
                model=result.model,
                accuracy=result.accuracy,
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
    Helper for scripts and notebooks: run pipeline with default config.
    """
    pipeline = OCRPipeline()
    return pipeline.run()


