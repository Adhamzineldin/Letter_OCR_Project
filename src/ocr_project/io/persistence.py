from pathlib import Path
from typing import Any, Mapping

import joblib


def save_model_artifact(
    path: str | Path,
    model: Any,
    accuracy: float | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """
    Save a single model (and optional accuracy / metadata) to disk.
    """
    artifact: dict[str, Any] = {"model": model}
    if accuracy is not None:
        artifact["accuracy"] = float(accuracy)
    if metadata is not None:
        artifact["metadata"] = dict(metadata)

    joblib.dump(artifact, Path(path))


def load_model_artifact(path: str | Path) -> dict[str, Any]:
    """
    Load a model artifact saved with save_model_artifact().
    """
    return joblib.load(Path(path))


