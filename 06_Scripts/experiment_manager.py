"""
==============================================================================
Experiment Manager
Project : Segmentation Project
Version : 4.0.0
==============================================================================

Manages new and resumed experiments, including:

- Timestamped run directories
- Reopening an existing interrupted run
- Atomic JSON metadata writes
- Environment and dataset metadata
- Configuration fingerprints
- Artifact hashing and registration
- Training and validation metrics
- Error records
- Resume provenance
- Best and last checkpoint summaries
"""

from datetime import datetime
from pathlib import Path
import hashlib
import json
import platform

import torch
import ultralytics

import config


class ExperimentManager:
    """Manage one new or resumed training experiment."""

    VERSION = "4.0.0"

    def __init__(self, cfg, run_dir=None):

        self.cfg = cfg
        self.is_existing = run_dir is not None

        if run_dir is None:

            timestamp = datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )

            base_name = (
                f"{cfg.experiment_name}_{timestamp}"
            )

            candidate = (
                config.TRAINING_RUNS /
                base_name
            )

            counter = 1

            while candidate.exists():

                candidate = (
                    config.TRAINING_RUNS /
                    f"{base_name}_{counter:02d}"
                )

                counter += 1

            self.run_dir = candidate.resolve()

        else:

            self.run_dir = (
                Path(run_dir)
                .expanduser()
                .resolve()
            )

        self.run_name = self.run_dir.name

        self.metadata_dir = (
            self.run_dir / "metadata"
        )

        self.logs_dir = (
            self.run_dir / "logs"
        )

        self.weights_dir = (
            self.run_dir / "weights"
        )

        self.plots_dir = (
            self.run_dir / "plots"
        )

        self.predictions_dir = (
            self.run_dir / "predictions"
        )

        self.exports_dir = (
            self.run_dir / "exports"
        )

        for directory in [
            self.run_dir,
            self.metadata_dir,
            self.logs_dir,
            self.weights_dir,
            self.plots_dir,
            self.predictions_dir,
            self.exports_dir,
        ]:
            directory.mkdir(
                parents=True,
                exist_ok=True,
            )

        self.artifacts = self._load_artifacts()

    # =========================================================================
    # JSON UTILITIES
    # =========================================================================

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
                str(key):
                    ExperimentManager._json_safe(item)
                for key, item in value.items()
            }

        if isinstance(
            value,
            (list, tuple, set),
        ):
            return [
                ExperimentManager._json_safe(item)
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

    def _load_json(
        self,
        filename,
        default=None,
    ):

        path = self.metadata_dir / filename

        if not path.exists():
            return {} if default is None else default

        try:

            with open(
                path,
                "r",
                encoding="utf-8",
            ) as file:
                return json.load(file)

        except Exception:
            return {} if default is None else default

    def _save_json(
        self,
        data,
        filename,
    ):

        path = self.metadata_dir / filename

        temporary = path.with_suffix(
            path.suffix + ".tmp"
        )

        safe_data = self._json_safe(data)

        with open(
            temporary,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                safe_data,
                file,
                indent=4,
                ensure_ascii=False,
            )

        temporary.replace(path)

        return path

    def save_metadata(
        self,
        filename,
        data,
    ):
        """Save an additional metadata JSON document."""

        return self._save_json(
            data,
            filename,
        )

    # =========================================================================
    # INITIAL METADATA
    # =========================================================================

    def save_experiment(
        self,
        status="initialized",
        resumed_from=None,
    ):

        data = self._load_json(
            "experiment.json"
        )

        data.setdefault(
            "created",
            datetime.now().isoformat(),
        )

        data.update({
            "experiment_name":
                self.run_name,

            "configured_experiment_name":
                self.cfg.experiment_name,

            "manager_version":
                self.VERSION,

            "updated":
                datetime.now().isoformat(),

            "status":
                status,

            "model":
                self.cfg.model,

            "dataset":
                self.cfg.dataset,

            "configuration_fingerprint":
                self.cfg.fingerprint(),
        })

        if resumed_from is not None:

            data["resumed_from"] = str(
                Path(resumed_from)
                .expanduser()
                .resolve()
            )

            data["resume_time"] = (
                datetime.now().isoformat()
            )

        return self._save_json(
            data,
            "experiment.json",
        )

    def save_environment(
        self,
        report=None,
    ):

        data = {
            "python":
                platform.python_version(),

            "platform":
                platform.platform(),

            "torch":
                torch.__version__,

            "cuda_available":
                torch.cuda.is_available(),

            "cuda_version":
                torch.version.cuda,

            "gpu": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),

            "gpu_memory_bytes": (
                torch.cuda
                .get_device_properties(0)
                .total_memory
                if torch.cuda.is_available()
                else None
            ),

            "ultralytics_version":
                ultralytics.__version__,

            "ultralytics_location":
                ultralytics.__file__,
        }

        if report is not None:
            data["verification_report"] = report

        return self._save_json(
            data,
            "environment.json",
        )

    def save_dataset(
        self,
        verification_report=None,
    ):

        data = {
            "dataset":
                self.cfg.dataset,

            "yaml":
                str(self.cfg.yaml),

            "class_names":
                config.CLASS_NAMES,

            "number_of_classes":
                config.NUM_CLASSES,
        }

        if verification_report is not None:

            data["verification_report"] = (
                verification_report
            )

        return self._save_json(
            data,
            "dataset_info.json",
        )

    def save_training_config(self):

        data = self.cfg.to_dict()

        data["fingerprint"] = (
            self.cfg.fingerprint()
        )

        return self._save_json(
            data,
            "training_config.json",
        )

    def initialize(
        self,
        environment_report=None,
        dataset_report=None,
        resumed_from=None,
    ):

        status = (
            "resuming"
            if resumed_from is not None
            else "initialized"
        )

        self.save_experiment(
            status=status,
            resumed_from=resumed_from,
        )

        self.save_environment(
            environment_report
        )

        self.save_dataset(
            dataset_report
        )

        self.save_training_config()

        artifacts_file = (
            self.metadata_dir /
            "artifacts.json"
        )

        if not artifacts_file.exists():

            self._save_json(
                self.artifacts,
                "artifacts.json",
            )

        return self.run_dir

    # =========================================================================
    # STATUS
    # =========================================================================

    def update_status(self, status):

        data = self._load_json(
            "experiment.json"
        )

        data.setdefault(
            "experiment_name",
            self.run_name,
        )

        data.setdefault(
            "configured_experiment_name",
            self.cfg.experiment_name,
        )

        data.setdefault(
            "created",
            datetime.now().isoformat(),
        )

        data["manager_version"] = self.VERSION
        data["status"] = str(status)

        data["updated"] = (
            datetime.now().isoformat()
        )

        return self._save_json(
            data,
            "experiment.json",
        )

    # =========================================================================
    # ARTIFACTS
    # =========================================================================

    def _load_artifacts(self):

        path = (
            self.metadata_dir /
            "artifacts.json"
        )

        if not path.exists():
            return {}

        try:

            with open(
                path,
                "r",
                encoding="utf-8",
            ) as file:
                data = json.load(file)

            return (
                data
                if isinstance(data, dict)
                else {}
            )

        except Exception:
            return {}

    @staticmethod
    def _sha256(path):

        digest = hashlib.sha256()

        with open(path, "rb") as file:

            while True:

                chunk = file.read(
                    1024 * 1024
                )

                if not chunk:
                    break

                digest.update(chunk)

        return digest.hexdigest()

    def register_artifact(
        self,
        name,
        path,
    ):

        path = (
            Path(path)
            .expanduser()
            .resolve()
        )

        entry = {
            "path":
                str(path),

            "exists":
                path.exists(),

            "registered":
                datetime.now().isoformat(),
        }

        if path.is_file():

            entry.update({
                "type":
                    "file",

                "size_bytes":
                    path.stat().st_size,

                "sha256":
                    self._sha256(path),
            })

        elif path.is_dir():

            entry.update({
                "type":
                    "directory",

                "file_count":
                    sum(
                        1
                        for item in path.rglob("*")
                        if item.is_file()
                    ),
            })

        else:

            entry["type"] = "missing"

        self.artifacts[str(name)] = entry

        self._save_json(
            self.artifacts,
            "artifacts.json",
        )

        return entry

    def get_artifact(self, name):
        return self.artifacts.get(str(name))

    # =========================================================================
    # METRICS, ERRORS, COMPLETION
    # =========================================================================

    def save_metrics(
        self,
        metrics,
        filename="metrics.json",
    ):

        return self._save_json(
            metrics,
            filename,
        )

    def save_error(
        self,
        error,
        traceback_text=None,
        elapsed_seconds=None,
    ):

        data = {
            "status":
                "failed",

            "failed_at":
                datetime.now().isoformat(),

            "error_type":
                type(error).__name__,

            "error":
                str(error),

            "traceback":
                traceback_text,

            "elapsed_seconds":
                elapsed_seconds,
        }

        path = self._save_json(
            data,
            "error.json",
        )

        self.update_status("failed")

        return path

    def finish(
        self,
        status="completed",
        elapsed_seconds=None,
        best_weights=None,
        last_weights=None,
        metrics=None,
        validation_metrics=None,
    ):

        summary = {
            "experiment":
                self.run_name,

            "status":
                str(status),

            "completed":
                datetime.now().isoformat(),

            "elapsed_seconds":
                elapsed_seconds,

            "elapsed_minutes": (
                round(
                    elapsed_seconds / 60,
                    3,
                )
                if elapsed_seconds is not None
                else None
            ),

            "best_weights": (
                str(best_weights)
                if best_weights is not None
                else None
            ),

            "last_weights": (
                str(last_weights)
                if last_weights is not None
                else None
            ),

            "training_metrics":
                metrics,

            "validation_metrics":
                validation_metrics,
        }

        path = self._save_json(
            summary,
            "experiment_summary.json",
        )

        self.update_status(status)

        return path
