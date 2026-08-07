"""
==============================================================================
Production Trainer
Project : Segmentation Project
Version : 4.0.0
==============================================================================

Production YOLO instance-segmentation trainer with:

- Environment and dataset verification
- Deterministic execution
- Fresh and resumed experiments
- Automatic interrupted-checkpoint discovery
- Controlled one-epoch sanity mode
- Early stopping
- AMP
- Periodic checkpoints
- Best and last checkpoint persistence
- Metrics and artifact registration
- Failure and interruption recovery
"""

from pathlib import Path
import json
import random
import shutil
import time
import traceback

import numpy as np
import torch
from ultralytics import YOLO

import config
import logger
import model_factory
import utils
import verify_dataset
import verify_environment

from training_config import TrainingConfig
from experiment_manager import ExperimentManager


class Trainer:
    """Coordinate fresh or resumed segmentation training."""

    VERSION = "4.0.0"

    TERMINAL_STATUSES = {
        "completed",
        "sanity_completed",
    }

    def __init__(self, cfg=None):

        self.cfg = cfg or TrainingConfig()
        self.cfg.validate()

        self.experiment = None
        self.run_log = None

        self.model = None
        self.results = None

        self.environment_report = None
        self.dataset_report = None

        self.resume_checkpoint = None
        self.training_save_dir = None

        self._set_determinism()

    # =========================================================================
    # DETERMINISM
    # =========================================================================

    def _set_determinism(self):

        seed = int(self.cfg.seed)

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = bool(
            self.cfg.deterministic
        )

        torch.backends.cudnn.benchmark = not bool(
            self.cfg.deterministic
        )

        utils.set_seed(seed)

    # =========================================================================
    # LOGGING
    # =========================================================================

    def _log(self, message=""):

        message = str(message)
        logger.log(message)

        if self.run_log is None:
            return

        self.run_log.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            self.run_log,
            "a",
            encoding="utf-8",
        ) as file:
            file.write(
                f"[{logger.timestamp()}] {message}\n"
            )

    # =========================================================================
    # JSON UTILITIES
    # =========================================================================

    @staticmethod
    def _read_json(path):

        path = Path(path)

        if not path.exists():
            return {}

        try:
            with open(
                path,
                "r",
                encoding="utf-8",
            ) as file:
                data = json.load(file)

            return data if isinstance(data, dict) else {}

        except Exception:
            return {}

    @staticmethod
    def _json_safe(value):

        if value is None or isinstance(
            value,
            (str, int, float, bool),
        ):
            return value

        if isinstance(value, Path):
            return str(value)

        if isinstance(value, dict):
            return {
                str(key): Trainer._json_safe(item)
                for key, item in value.items()
            }

        if isinstance(value, (list, tuple, set)):
            return [
                Trainer._json_safe(item)
                for item in value
            ]

        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass

        if hasattr(value, "tolist"):
            try:
                return value.tolist()
            except Exception:
                pass

        return str(value)

    # =========================================================================
    # RESUME DISCOVERY
    # =========================================================================

    def _experiment_root_from_checkpoint(
        self,
        checkpoint,
    ):

        checkpoint = (
            Path(checkpoint)
            .expanduser()
            .resolve()
        )

        for parent in checkpoint.parents:

            experiment_file = (
                parent
                / "metadata"
                / "experiment.json"
            )

            if experiment_file.exists():
                return parent

        return None

    def _matching_incomplete_run(
        self,
        run_dir,
    ):

        training_config_file = (
            run_dir
            / "metadata"
            / "training_config.json"
        )

        experiment_file = (
            run_dir
            / "metadata"
            / "experiment.json"
        )

        saved_config = self._read_json(
            training_config_file
        )

        experiment_data = self._read_json(
            experiment_file
        )

        if not saved_config:
            return False

        if (
            saved_config.get("experiment_name")
            != self.cfg.experiment_name
        ):
            return False

        if (
            saved_config.get("dataset")
            != self.cfg.dataset
        ):
            return False

        if (
            saved_config.get("model")
            != self.cfg.model
        ):
            return False

        status = str(
            experiment_data.get("status", "")
        ).lower()

        if status in self.TERMINAL_STATUSES:
            return False

        return True

    def _find_resume_checkpoint(self):

        runs_root = Path(
            config.TRAINING_RUNS
        )

        if not runs_root.exists():
            return None

        candidates = []

        for run_dir in runs_root.iterdir():

            if not run_dir.is_dir():
                continue

            if not self._matching_incomplete_run(
                run_dir
            ):
                continue

            checkpoint_locations = [
                (
                    run_dir
                    / "training"
                    / "weights"
                    / "last.pt"
                ),
                (
                    run_dir
                    / "train"
                    / "weights"
                    / "last.pt"
                ),
                (
                    run_dir
                    / "weights"
                    / "last.pt"
                ),
            ]

            for checkpoint in checkpoint_locations:

                if (
                    checkpoint.exists()
                    and checkpoint.is_file()
                    and checkpoint.stat().st_size > 0
                ):
                    candidates.append(checkpoint)
                    break

        if not candidates:
            return None

        return max(
            candidates,
            key=lambda path: path.stat().st_mtime,
        ).resolve()

    def _validate_resume_checkpoint(
        self,
        checkpoint,
    ):

        checkpoint = (
            Path(checkpoint)
            .expanduser()
            .resolve()
        )

        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Resume checkpoint not found: "
                f"{checkpoint}"
            )

        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Resume checkpoint is not a file: "
                f"{checkpoint}"
            )

        if checkpoint.suffix.lower() != ".pt":
            raise ValueError(
                "Resume checkpoint must be a .pt file."
            )

        if checkpoint.stat().st_size == 0:
            raise ValueError(
                f"Resume checkpoint is empty: "
                f"{checkpoint}"
            )

        experiment_root = (
            self._experiment_root_from_checkpoint(
                checkpoint
            )
        )

        if experiment_root is None:
            raise RuntimeError(
                "The checkpoint is not inside a managed "
                "experiment containing metadata/experiment.json."
            )

        return checkpoint

    def _resolve_resume(self, resume):

        if resume is False or resume is None:
            return None

        if isinstance(resume, Path):
            return self._validate_resume_checkpoint(
                resume
            )

        if isinstance(resume, str):

            normalized = resume.strip().lower()

            if normalized == "auto":

                checkpoint = (
                    self._find_resume_checkpoint()
                )

                if checkpoint is None:
                    return None

                return self._validate_resume_checkpoint(
                    checkpoint
                )

            return self._validate_resume_checkpoint(
                resume
            )

        if resume is True:

            checkpoint = (
                self._find_resume_checkpoint()
            )

            if checkpoint is None:
                raise FileNotFoundError(
                    "resume=True was requested, but no "
                    "matching incomplete last.pt checkpoint "
                    "was found."
                )

            return self._validate_resume_checkpoint(
                checkpoint
            )

        raise TypeError(
            "resume must be False, True, 'auto', "
            "or a managed checkpoint path."
        )

    # =========================================================================
    # EXPERIMENT SETUP
    # =========================================================================

    def _configure_experiment(
        self,
        resume_checkpoint=None,
    ):

        existing_run = None

        if resume_checkpoint is not None:

            existing_run = (
                self._experiment_root_from_checkpoint(
                    resume_checkpoint
                )
            )

            if existing_run is None:
                raise RuntimeError(
                    "Could not locate the experiment root "
                    "for the resume checkpoint."
                )

        self.experiment = ExperimentManager(
            self.cfg,
            run_dir=existing_run,
        )

        self.run_log = (
            self.experiment.logs_dir
            / "run.log"
        )

        return self.experiment

    # =========================================================================
    # VERIFICATION
    # =========================================================================

    def verify(self):

        self._log("=" * 70)
        self._log("VERIFYING ENVIRONMENT")

        self.environment_report = (
            verify_environment.verify()
        )

        if not isinstance(
            self.environment_report,
            dict,
        ):
            raise TypeError(
                "Environment verification did not "
                "return a dictionary."
            )

        if not self.environment_report.get(
            "passed",
            False,
        ):
            raise RuntimeError(
                "Environment verification failed."
            )

        self._log(
            "Environment verification passed."
        )

        self._log("=" * 70)
        self._log(
            f"VERIFYING "
            f"{self.cfg.dataset.upper()} DATASET"
        )

        self.dataset_report = (
            verify_dataset.verify(
                self.cfg.dataset
            )
        )

        if not isinstance(
            self.dataset_report,
            dict,
        ):
            raise TypeError(
                "Dataset verification did not "
                "return a dictionary."
            )

        if not self.dataset_report.get(
            "passed",
            False,
        ):
            raise RuntimeError(
                "Dataset verification failed."
            )

        self._log(
            "Dataset verification passed."
        )

        return (
            self.environment_report,
            self.dataset_report,
        )

    # =========================================================================
    # MODEL
    # =========================================================================

    def load_model(
        self,
        resume_checkpoint=None,
    ):

        self._log("=" * 70)

        if resume_checkpoint is None:

            self._log(
                f"LOADING PRETRAINED MODEL: "
                f"{self.cfg.model}"
            )

            self.model = model_factory.get_model(
                self.cfg.model
            )

        else:

            self._log(
                f"LOADING RESUME CHECKPOINT: "
                f"{resume_checkpoint}"
            )

            self.model = YOLO(
                str(resume_checkpoint)
            )

        model_task = getattr(
            self.model,
            "task",
            None,
        )

        if model_task != "segment":
            raise RuntimeError(
                "Expected a segmentation model, "
                f"but loaded task was: {model_task}"
            )

        self._log(
            "Segmentation model loaded successfully."
        )

        return self.model

    # =========================================================================
    # PREPARATION
    # =========================================================================

    def prepare(
        self,
        resume_checkpoint=None,
    ):

        self._configure_experiment(
            resume_checkpoint
        )

        (
            environment_report,
            dataset_report,
        ) = self.verify()

        self.experiment.initialize(
            environment_report=environment_report,
            dataset_report=dataset_report,
            resumed_from=resume_checkpoint,
        )

        self.load_model(
            resume_checkpoint
        )

        if resume_checkpoint is None:

            pretrained_weights = (
                config.WEIGHTS_DIR
                / f"{self.cfg.model}.pt"
            )

            if not pretrained_weights.exists():
                raise FileNotFoundError(
                    "Expected pretrained weights were "
                    f"not found: {pretrained_weights}"
                )

            self.experiment.register_artifact(
                "pretrained_weights",
                pretrained_weights,
            )

        else:

            self.experiment.register_artifact(
                "resume_checkpoint",
                resume_checkpoint,
            )

        self.experiment.update_status(
            "ready_for_training"
        )

        self._log("=" * 70)
        self._log(
            "TRAINER PREPARATION COMPLETED"
        )

        return self.experiment.run_dir

    # =========================================================================
    # TRAINING ARGUMENTS
    # =========================================================================

    def _fresh_training_arguments(
        self,
        sanity=False,
    ):

        arguments = {
            "data": str(self.cfg.yaml),
            "epochs": int(self.cfg.epochs),
            "batch": int(self.cfg.batch),
            "imgsz": int(self.cfg.imgsz),
            "workers": int(self.cfg.workers),
            "device": self.cfg.device,
            "seed": int(self.cfg.seed),

            "deterministic": bool(
                self.cfg.deterministic
            ),

            "amp": bool(self.cfg.amp),
            "cache": self.cfg.cache,
            "optimizer": self.cfg.optimizer,
            "lr0": float(self.cfg.lr0),
            "lrf": float(self.cfg.lrf),

            "momentum": float(
                self.cfg.momentum
            ),

            "weight_decay": float(
                self.cfg.weight_decay
            ),

            "warmup_epochs": float(
                self.cfg.warmup_epochs
            ),

            "patience": int(
                self.cfg.patience
            ),

            "save": bool(self.cfg.save),

            "save_period": int(
                self.cfg.save_period
            ),

            "plots": bool(self.cfg.plots),
            "verbose": bool(self.cfg.verbose),

            "pretrained": bool(
                self.cfg.pretrained
            ),

            "project": str(
                self.experiment.run_dir
            ),

            "name": "training",
            "exist_ok": True,
            "val": True,
        }

        if sanity:

            arguments.update({
                "epochs": 1,

                "batch": min(
                    int(self.cfg.batch),
                    8,
                ),

                "workers": min(
                    int(self.cfg.workers),
                    2,
                ),

                "patience": 1,
                "save_period": -1,
                "cache": False,
                "fraction": 0.05,
            })

        return arguments

    def _resume_training_arguments(self):

        return {
            "resume": True,
            "device": self.cfg.device,
            "workers": int(self.cfg.workers),
            "plots": bool(self.cfg.plots),
            "verbose": bool(self.cfg.verbose),
        }

    def build_training_arguments(
        self,
        sanity=False,
        resume_checkpoint=None,
    ):

        if resume_checkpoint is not None:
            return self._resume_training_arguments()

        return self._fresh_training_arguments(
            sanity=sanity
        )

    # =========================================================================
    # METRICS
    # =========================================================================

    def _extract_metrics(self):

        possible_sources = [
            self.results,
            getattr(
                self.model,
                "metrics",
                None,
            ),
            getattr(
                getattr(
                    self.model,
                    "trainer",
                    None,
                ),
                "validator",
                None,
            ),
        ]

        for source in possible_sources:

            if source is None:
                continue

            results_dict = getattr(
                source,
                "results_dict",
                None,
            )

            if isinstance(results_dict, dict):
                return self._json_safe(
                    results_dict
                )

            metrics_object = getattr(
                source,
                "metrics",
                None,
            )

            nested_results = getattr(
                metrics_object,
                "results_dict",
                None,
            )

            if isinstance(nested_results, dict):
                return self._json_safe(
                    nested_results
                )

            if isinstance(source, dict):
                return self._json_safe(source)

        return {}

    # =========================================================================
    # ARTIFACT COLLECTION
    # =========================================================================

    @staticmethod
    def _copy_checkpoint(
        source,
        destination,
    ):

        if source is None:
            return None

        source = Path(source)
        destination = Path(destination)

        if (
            not source.exists()
            or not source.is_file()
            or source.stat().st_size == 0
        ):
            return None

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if source.resolve() != destination.resolve():
            shutil.copy2(
                source,
                destination,
            )

        return destination

    def collect_artifacts(self):

        trainer_object = getattr(
            self.model,
            "trainer",
            None,
        )

        if trainer_object is None:
            raise RuntimeError(
                "Ultralytics trainer object is "
                "unavailable after training."
            )

        save_dir = Path(
            trainer_object.save_dir
        ).resolve()

        if not save_dir.exists():
            raise FileNotFoundError(
                "Ultralytics training output "
                f"directory is missing: {save_dir}"
            )

        self.training_save_dir = save_dir

        best_source = getattr(
            trainer_object,
            "best",
            save_dir / "weights" / "best.pt",
        )

        last_source = getattr(
            trainer_object,
            "last",
            save_dir / "weights" / "last.pt",
        )

        best_copy = self._copy_checkpoint(
            best_source,
            self.experiment.weights_dir
            / "best.pt",
        )

        last_copy = self._copy_checkpoint(
            last_source,
            self.experiment.weights_dir
            / "last.pt",
        )

        if best_copy is None:
            raise FileNotFoundError(
                "Training completed, but best.pt "
                "was not found."
            )

        if last_copy is None:
            raise FileNotFoundError(
                "Training completed, but last.pt "
                "was not found."
            )

        self.experiment.register_artifact(
            "training_output_directory",
            save_dir,
        )

        self.experiment.register_artifact(
            "best_weights",
            best_copy,
        )

        self.experiment.register_artifact(
            "last_weights",
            last_copy,
        )

        optional_artifacts = {
            "results_csv":
                save_dir / "results.csv",

            "results_plot":
                save_dir / "results.png",

            "training_arguments":
                save_dir / "args.yaml",

            "confusion_matrix":
                save_dir / "confusion_matrix.png",

            "normalized_confusion_matrix":
                (
                    save_dir
                    / "confusion_matrix_normalized.png"
                ),

            "labels_plot":
                save_dir / "labels.jpg",

            "labels_correlogram":
                save_dir / "labels_correlogram.jpg",
        }

        for name, path in optional_artifacts.items():

            if path.exists():
                self.experiment.register_artifact(
                    name,
                    path,
                )

        return (
            best_copy,
            last_copy,
            save_dir,
        )

    # =========================================================================
    # TRAINING
    # =========================================================================

    def train(
        self,
        sanity=False,
        resume="auto",
    ):

        start_time = time.time()

        if sanity:

            if resume not in {
                False,
                None,
                "auto",
            }:
                raise ValueError(
                    "Sanity training cannot resume an "
                    "existing full-training checkpoint."
                )

            resume = False

        try:

            self.resume_checkpoint = (
                self._resolve_resume(resume)
            )

            self.prepare(
                self.resume_checkpoint
            )

            arguments = (
                self.build_training_arguments(
                    sanity=sanity,
                    resume_checkpoint=(
                        self.resume_checkpoint
                    ),
                )
            )

            self.experiment.save_metadata(
                "runtime_training_arguments.json",
                arguments,
            )

            if (
                self.resume_checkpoint is None
                and str(
                    self.cfg.optimizer
                ).lower() == "auto"
            ):
                self._log(
                    "NOTICE: optimizer='auto' allows "
                    "Ultralytics to select the optimizer "
                    "and may override lr0 and momentum."
                )

            if self.resume_checkpoint is not None:

                training_status = (
                    "resuming_training"
                )

                self._log("=" * 70)
                self._log(
                    "RESUMING INTERRUPTED TRAINING"
                )
                self._log(
                    f"Checkpoint: "
                    f"{self.resume_checkpoint}"
                )

            elif sanity:

                training_status = (
                    "sanity_training"
                )

                self._log("=" * 70)
                self._log(
                    "STARTING ONE-EPOCH "
                    "SANITY TRAINING"
                )

            else:

                training_status = "training"

                self._log("=" * 70)
                self._log(
                    "STARTING FULL TRAINING"
                )

            self.experiment.update_status(
                training_status
            )

            self._log("=" * 70)

            self.results = self.model.train(
                **arguments
            )

            metrics = self._extract_metrics()

            self.experiment.save_metrics(
                metrics
            )

            (
                best_weights,
                last_weights,
                save_dir,
            ) = self.collect_artifacts()

            elapsed_seconds = (
                time.time() - start_time
            )

            final_status = (
                "sanity_completed"
                if sanity
                else "completed"
            )

            self.experiment.finish(
                status=final_status,
                elapsed_seconds=elapsed_seconds,
                best_weights=best_weights,
                last_weights=last_weights,
                metrics=metrics,
            )

            self._log("=" * 70)
            self._log(
                f"TRAINING FINISHED: "
                f"{final_status}"
            )
            self._log(
                f"Elapsed minutes: "
                f"{elapsed_seconds / 60:.2f}"
            )
            self._log("=" * 70)

            return {
                "status": final_status,

                "run_name":
                    self.experiment.run_name,

                "run_directory":
                    str(self.experiment.run_dir),

                "training_directory":
                    str(save_dir),

                "best_weights":
                    str(best_weights),

                "last_weights":
                    str(last_weights),

                "elapsed_seconds":
                    elapsed_seconds,

                "metrics":
                    metrics,

                "resumed_from": (
                    str(self.resume_checkpoint)
                    if self.resume_checkpoint
                    is not None
                    else None
                ),
            }

        except KeyboardInterrupt as error:

            elapsed_seconds = (
                time.time() - start_time
            )

            traceback_text = (
                traceback.format_exc()
            )

            if self.experiment is not None:

                self.experiment.save_error(
                    error,
                    traceback_text=traceback_text,
                    elapsed_seconds=elapsed_seconds,
                )

                self.experiment.update_status(
                    "interrupted"
                )

            self._log(
                "Training interrupted. Run the same "
                "configuration again with resume='auto'."
            )

            raise

        except Exception as error:

            elapsed_seconds = (
                time.time() - start_time
            )

            traceback_text = (
                traceback.format_exc()
            )

            if self.experiment is not None:

                self.experiment.save_error(
                    error,
                    traceback_text=traceback_text,
                    elapsed_seconds=elapsed_seconds,
                )

            self._log(
                f"{type(error).__name__}: {error}"
            )

            raise


def train():
    """Start or automatically resume full training."""
    return Trainer().train(
        sanity=False,
        resume="auto",
    )


def sanity_train():
    """Run a fresh controlled one-epoch sanity experiment."""
    return Trainer().train(
        sanity=True,
        resume=False,
    )
