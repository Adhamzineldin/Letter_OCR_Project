from pathlib import Path
import numpy as np
import pandas as pd

from .. import config


class DatasetSplit:
    def __init__(self, images: np.ndarray, labels: np.ndarray):
        self.images = images
        self.labels = labels


class EmnistLoader:
    """
    Minimal EMNIST loader:
    - Reads CSV directly
    - Fixes orientation
    - Returns images + labels
    """

    def __init__(self, train_csv: Path, test_csv: Path):
        self.train_csv = train_csv
        self.test_csv = test_csv

    def _load_csv(self, path: Path, limit: int | None = None) -> DatasetSplit:
        df = pd.read_csv(path, header=None)
        if limit is not None:
            df = df.iloc[:limit]

        labels = df.iloc[:, 0].to_numpy(np.int8)
        pixels = df.iloc[:, 1:].to_numpy(np.uint8)

        images = pixels.reshape((-1, config.IMAGE_WIDTH, config.IMAGE_HEIGHT))

    
        images = np.transpose(images, (0, 2, 1))
        images = np.flip(images, axis=2)

        return DatasetSplit(images, labels)

    def load_train(self, limit: int | None = None) -> DatasetSplit:
        return self._load_csv(self.train_csv, limit)

    def load_test(self, limit: int | None = None) -> DatasetSplit:
        return self._load_csv(self.test_csv, limit)

    def load_combined(self, train_limit: int | None = None, test_limit: int | None = None) -> DatasetSplit:
        """
        Load and combine both train and test datasets for cross-validation.
        
        Args:
            train_limit: Optional limit on number of training samples
            test_limit: Optional limit on number of test samples
            
        Returns:
            DatasetSplit containing combined images and labels
        """
        train_split = self.load_train(limit=train_limit)
        test_split = self.load_test(limit=test_limit)
        
        # Combine images and labels
        combined_images = np.concatenate([train_split.images, test_split.images], axis=0)
        combined_labels = np.concatenate([train_split.labels, test_split.labels], axis=0)
        
        return DatasetSplit(combined_images, combined_labels)