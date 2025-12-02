# src/ocr_project/preprocess/transformer.py
import numpy as np

class Transformer:
    """Transforms raw EMNIST images for model consumption."""

    @staticmethod
    def normalize(images: np.ndarray) -> np.ndarray:
        """
        Normalize pixel values from [0, 255] to [0.0, 1.0]
        """
        return images.astype(np.float32) / 255.0

    @staticmethod
    def flatten(images: np.ndarray) -> np.ndarray:
        """
        Flatten images from (n_samples, H, W) -> (n_samples, H*W)
        """
        n_samples = images.shape[0]
        return images.reshape(n_samples, -1)


    @staticmethod
    def one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
        """
        Convert integer labels to one-hot vectors
        """
        return np.eye(num_classes)[labels]
