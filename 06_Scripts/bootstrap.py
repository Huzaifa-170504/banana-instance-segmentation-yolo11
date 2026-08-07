
"""
============================================================

Segmentation Project

bootstrap.py

Purpose
-------
Loads and verifies the LOCAL Ultralytics repository.

This script is the ONLY place responsible for:

- adding repository to sys.path
- installing editable finder if required
- importing ultralytics
- verifying repository location

============================================================
"""

from pathlib import Path
import sys
import importlib.util


PROJECT_ROOT = Path("/content/drive/MyDrive/Segmentation_Project")

REPOSITORY = PROJECT_ROOT / "02_Repositories" / "ultralytics"


def load_ultralytics():

    if not REPOSITORY.exists():
        raise FileNotFoundError(
            f"Repository not found:\n{REPOSITORY}"
        )

    repo = str(REPOSITORY)

    if repo not in sys.path:
        sys.path.insert(0, repo)

    try:

        import __editable___ultralytics_8_4_115_finder as finder

        finder.install()

    except Exception:
        pass

    import ultralytics

    return ultralytics


def verify():

    ul = load_ultralytics()

    print("=" * 80)
    print("LOCAL ULTRALYTICS VERIFIED")
    print("=" * 80)

    print("Version :", ul.__version__)
    print("Location:", ul.__file__)

    return True


if __name__ == "__main__":

    verify()
