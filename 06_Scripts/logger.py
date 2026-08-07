
from pathlib import Path
from datetime import datetime
import json

PROJECT_ROOT = Path("/content/drive/MyDrive/Segmentation_Project")
LOG_DIR = PROJECT_ROOT / "05_Logs"


def timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(message):
    """
    Print a timestamped message to the notebook.
    """
    print(f"[{timestamp()}] {message}")


def save_json(filename, data):
    """
    Save a dictionary as formatted JSON in 05_Logs.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    path = LOG_DIR / filename

    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Saved: {path}")

    return path


def start_run():
    """
    Create a timestamped log file for the current run.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S.log")
    path = LOG_DIR / run_name

    with open(path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("Segmentation Project Log\n")
        f.write("=" * 80 + "\n")
        f.write(f"Started : {timestamp()}\n")

    print(f"Run log created: {path}")

    return path
