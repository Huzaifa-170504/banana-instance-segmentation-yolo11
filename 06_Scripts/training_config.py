"""
================================================================================
Training Configuration
Project : Segmentation Project
Version : 3.0.0
================================================================================
"""

from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json

import config


@dataclass
class TrainingConfig:

    # -------------------------------------------------------------------------
    # Experiment
    # -------------------------------------------------------------------------

    experiment_name: str = "banana_segmentation"
    dataset: str = "research"
    model: str = "yolo11m-seg"

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------

    epochs: int = 100
    batch: int = 16
    imgsz: int = 640
    workers: int = 2
    device: int = 0

    seed: int = 42
    deterministic: bool = True

    amp: bool = True
    cache: bool = False

    # -------------------------------------------------------------------------
    # Optimizer
    # -------------------------------------------------------------------------

    optimizer: str = "auto"

    lr0: float = 0.01
    lrf: float = 0.01

    momentum: float = 0.937
    weight_decay: float = 0.0005

    warmup_epochs: float = 3.0

    patience: int = 50

    # -------------------------------------------------------------------------
    # Saving
    # -------------------------------------------------------------------------

    save: bool = True
    save_period: int = 10

    plots: bool = True
    verbose: bool = True
    pretrained: bool = True

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------

    def __post_init__(self):

        if self.dataset.lower() == "research":
            self.yaml = config.RESEARCH_YAML
        elif self.dataset.lower() == "benchmark":
            self.yaml = config.BENCHMARK_YAML
        else:
            raise ValueError(
                f"Unknown dataset: {self.dataset}"
            )

        self.run_directory = (
            config.TRAINING_RUNS /
            self.experiment_name
        )

        self.validate()

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    def validate(self):

        if self.epochs <= 0:
            raise ValueError("epochs must be > 0")

        if self.batch <= 0:
            raise ValueError("batch must be > 0")

        if self.imgsz % 32 != 0:
            raise ValueError("imgsz must be divisible by 32")

        if not Path(self.yaml).exists():
            raise FileNotFoundError(self.yaml)

        valid_models = {
            "yolo11n-seg",
            "yolo11s-seg",
            "yolo11m-seg",
            "yolo11l-seg",
            "yolo11x-seg",
        }

        if self.model not in valid_models:
            raise ValueError(
                f"Unsupported model: {self.model}"
            )

        return True

    # -------------------------------------------------------------------------
    # Dictionary
    # -------------------------------------------------------------------------

    def to_dict(self):

        data = asdict(self)

        data["yaml"] = str(self.yaml)
        data["run_directory"] = str(self.run_directory)

        return data

    # -------------------------------------------------------------------------
    # Fingerprint
    # -------------------------------------------------------------------------

    def fingerprint(self):

        text = json.dumps(
            self.to_dict(),
            sort_keys=True
        )

        return hashlib.sha256(
            text.encode()
        ).hexdigest()

    # -------------------------------------------------------------------------
    # JSON
    # -------------------------------------------------------------------------

    def save_json(self):

        self.run_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        output = (
            self.run_directory /
            "training_config.json"
        )

        with open(output, "w") as f:

            json.dump(
                self.to_dict(),
                f,
                indent=4
            )

        return output
