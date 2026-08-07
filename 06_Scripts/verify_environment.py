
from pathlib import Path
import sys
import torch
import ultralytics


def verify():

    project_root = Path("/content/drive/MyDrive/Segmentation_Project")

    report = {}

    report["python"] = sys.version.split()[0]
    report["torch"] = torch.__version__
    report["cuda"] = torch.version.cuda
    report["cuda_available"] = torch.cuda.is_available()

    report["gpu"] = (
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else None
    )

    report["project_exists"] = project_root.exists()

    report["repository_exists"] = (
        project_root /
        "02_Repositories" /
        "ultralytics"
    ).exists()

    report["dataset_exists"] = (
        project_root /
        "01_Datasets"
    ).exists()

    report["ultralytics_version"] = ultralytics.__version__
    report["ultralytics_location"] = ultralytics.__file__

    errors = []

    if not report["project_exists"]:
        errors.append("Project folder missing.")

    if not report["repository_exists"]:
        errors.append("Ultralytics repository missing.")

    if not report["dataset_exists"]:
        errors.append("Dataset folder missing.")

    if not report["cuda_available"]:
        errors.append("CUDA is not available.")

    expected_repo = str(
        project_root /
        "02_Repositories" /
        "ultralytics"
    )

    if expected_repo not in report["ultralytics_location"]:
        errors.append(
            "Ultralytics is NOT imported from the local cloned repository."
        )

    report["passed"] = len(errors) == 0
    report["errors"] = errors

    if errors:
        raise RuntimeError(
            "\\n".join(errors)
        )

    return report
