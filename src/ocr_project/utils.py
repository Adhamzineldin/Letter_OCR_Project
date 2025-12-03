import numpy as np

from . import config


def emnist_label_to_letter(label: int) -> str:
    """
    Map EMNIST 'letters' labels to lowercase ASCII letters.

    According to the EMNIST Letters specification, labels are typically 1..26
    mapped to 'a'..'z'. If your mapping file indicates a different convention,
    adjust this function accordingly.
    """
    return chr(ord("a") + int(label) - 1)


def prepare_single_image_for_model(img_28x28: np.ndarray) -> np.ndarray:
    """
    Take a single 28x28 grayscale image and convert it to the same representation
    used for model training: shape (1, 28*28) and normalized to [0, 1].
    """
    img = img_28x28.astype(np.float32) / 255.0

    # Ensure shape (1, H, W)
    if img.ndim == 2:
        img = img[None, :, :]

    # Sanity check on image size
    if img.shape[1] != config.IMAGE_HEIGHT or img.shape[2] != config.IMAGE_WIDTH:
        raise ValueError(
            f"Expected image of shape ({config.IMAGE_HEIGHT}, {config.IMAGE_WIDTH}), "
            f"got {img.shape[1:]}"
        )

    n = img.shape[0]
    return img.reshape(n, -1)


