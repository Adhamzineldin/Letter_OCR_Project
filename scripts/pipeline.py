"""
Entry point script to run the end-to-end OCR pipeline.

Usage (from project root):

    python -m scripts.pipeline

This will:
  - load EMNIST letters data
  - train Decision Tree and Random Forest models
  - evaluate them
  - save artifacts under the 'artifacts' directory
"""

from src.ocr_project.pipeline import run_default_pipeline


def main() -> None:
    results = run_default_pipeline()

    for name, result in results.items():
        print(f"{name}: accuracy = {result.accuracy:.4f}")


if __name__ == "__main__":
    main()


