# src/ocr_project/preprocess/augmentations.py
import numpy as np
from scipy.ndimage import rotate, shift

class Augmentations:
    """Apply image augmentations to increase dataset diversity."""

    @staticmethod
    def random_rotation(images: np.ndarray, max_angle: float = 15.0) -> np.ndarray:
        """Randomly rotate each image within [-max_angle, max_angle] degrees."""
        rotated_images = np.empty_like(images)
        for i in range(images.shape[0]):
            angle = np.random.uniform(-max_angle, max_angle)
            rotated_images[i] = rotate(images[i], angle, reshape=False, mode='nearest')
        return rotated_images

    @staticmethod
    def random_shift(images: np.ndarray, max_shift: int = 2) -> np.ndarray:
        """Randomly shift images horizontally and vertically."""
        shifted_images = np.empty_like(images)
        for i in range(images.shape[0]):
            dx = np.random.randint(-max_shift, max_shift + 1)
            dy = np.random.randint(-max_shift, max_shift + 1)
            shifted_images[i] = shift(images[i], shift=(dx, dy), mode='nearest')
        return shifted_images

    @staticmethod
    def add_gaussian_noise(images: np.ndarray, noise_level: float = 0.05) -> np.ndarray:
        """Add Gaussian noise to images and clip between 0 and 1."""
        noisy = images + np.random.normal(0.0, noise_level, images.shape)
        return np.clip(noisy, 0.0, 1.0)

    @staticmethod
    def augment(images: np.ndarray, rotate_max: float = 15.0, shift_max: int = 2, noise_level: float = 0.03) -> np.ndarray:
        """Apply a combination of augmentations."""
        images_aug = Augmentations.random_rotation(images, rotate_max)
        images_aug = Augmentations.random_shift(images_aug, shift_max)
        images_aug = Augmentations.add_gaussian_noise(images_aug, noise_level)
        return images_aug
