
"""
==============================================================================
Model Factory
==============================================================================

Professional model manager for the Segmentation Project.

Responsibilities
----------------
✓ Validate model name
✓ Download pretrained weights if missing
✓ Store weights permanently inside the project
✓ Load YOLO model from local weights
✓ Log every action

Author : Huzaifa
Version: 2.0
"""

from pathlib import Path
import shutil

from ultralytics import YOLO

import config
import logger


SUPPORTED_MODELS = {
    "yolo11n-seg": "yolo11n-seg.pt",
    "yolo11s-seg": "yolo11s-seg.pt",
    "yolo11m-seg": "yolo11m-seg.pt",
    "yolo11l-seg": "yolo11l-seg.pt",
    "yolo11x-seg": "yolo11x-seg.pt",
}


def available_models() -> list[str]:
    """Return supported model names."""
    return list(SUPPORTED_MODELS.keys())


def get_model(model_name: str = "yolo11m-seg") -> YOLO:
    """
    Load a pretrained YOLO segmentation model.

    Parameters
    ----------
    model_name : str
        Example:
            yolo11m-seg

    Returns
    -------
    ultralytics.YOLO
    """

    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model '{model_name}'.\n"
            f"Supported models:\n{available_models()}"
        )

    config.WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    weights_name = SUPPORTED_MODELS[model_name]
    project_weights = config.WEIGHTS_DIR / weights_name

    logger.log("=" * 70)
    logger.log(f"Loading model: {model_name}")

    # ------------------------------------------------------------------
    # Download only once
    # ------------------------------------------------------------------
    if not project_weights.exists():

        logger.log("Weights not found.")
        logger.log("Downloading official pretrained weights...")

        temp_model = YOLO(weights_name)

        source = Path(temp_model.ckpt_path)

        if not source.exists():
            raise FileNotFoundError(
                f"Downloaded checkpoint not found:\n{source}"
            )

        shutil.copy2(source, project_weights)

        logger.log(f"Weights copied to:\n{project_weights}")

    else:

        logger.log("Existing weights found.")
        logger.log(str(project_weights))

    # ------------------------------------------------------------------
    # Load model from project directory
    # ------------------------------------------------------------------

    model = YOLO(str(project_weights))

    logger.log("Model loaded successfully.")
    logger.log("=" * 70)

    return model
