from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
ARCHIVE_DIR = DATA_DIR / "archive"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

EMNIST_LETTERS_TRAIN = RAW_DATA_DIR / "emnist-letters-train.csv"
EMNIST_LETTERS_TEST = RAW_DATA_DIR / "emnist-letters-test.csv"
EMNIST_LETTERS_MAPPING = RAW_DATA_DIR / "emnist-letters-mapping.txt"

IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28

RANDOM_SEED = 42


def ensure_directories():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
