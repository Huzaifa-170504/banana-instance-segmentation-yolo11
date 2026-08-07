
from pathlib import Path
import sys
import importlib
import torch
from datetime import datetime


def initialize():

    project_root = Path("/content/drive/MyDrive/Segmentation_Project")

    if not project_root.exists():
        raise FileNotFoundError(
            "Project folder not found."
        )

    scripts = project_root / "06_Scripts"

    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))

    repo = (
        project_root /
        "02_Repositories" /
        "ultralytics"
    )

    if not repo.exists():
        raise FileNotFoundError(
            "Ultralytics repository not found."
        )

    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    importlib.invalidate_caches()

    import ultralytics

    info = {

        "timestamp": str(datetime.now()),

        "project_root": str(project_root),

        "repository": str(repo),

        "ultralytics_version": ultralytics.__version__,

        "ultralytics_location": ultralytics.__file__,

        "python": sys.version.split()[0],

        "torch": torch.__version__,

        "cuda": torch.version.cuda,

        "cuda_available": torch.cuda.is_available(),

        "gpu":
        (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        )
    }

    print("=" * 90)
    print("PROJECT INITIALIZATION SUCCESSFUL")
    print("=" * 90)

    for key, value in info.items():
        print(f"{key:25}: {value}")

    return info
