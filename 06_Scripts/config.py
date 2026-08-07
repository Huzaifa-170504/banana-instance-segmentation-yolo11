
"""
==============================================================================
Segmentation Project Configuration
==============================================================================

Centralized project paths and constants.

Every notebook and script should import from this module instead of hardcoding
paths.
"""

from pathlib import Path

# =============================================================================
# PROJECT ROOT
# =============================================================================

PROJECT_ROOT = Path("/content/drive/MyDrive/Segmentation_Project")

# =============================================================================
# DIRECTORIES
# =============================================================================

NOTEBOOK_DIR = PROJECT_ROOT / "01_Notebooks"

DATASET_ROOT = PROJECT_ROOT / "01_Datasets"
RAW_DATASET = DATASET_ROOT / "dataset"
PREPARED_DATASET = DATASET_ROOT / "Prepared_Dataset"

REPOSITORY_ROOT = PROJECT_ROOT / "02_Repositories"
ULTRALYTICS_REPO = REPOSITORY_ROOT / "ultralytics"

WEIGHTS_DIR = PROJECT_ROOT / "03_Pretrained_Weights"
RUNS_DIR = PROJECT_ROOT / "04_Training_Runs"

# Compatibility alias used by training modules.
TRAINING_RUNS = RUNS_DIR
CHECKPOINT_DIR = PROJECT_ROOT / "06_Checkpoints"
EXPORT_DIR = PROJECT_ROOT / "07_Exports"
LOG_DIR = PROJECT_ROOT / "05_Logs"
SCRIPT_DIR = PROJECT_ROOT / "06_Scripts"
BACKUP_DIR = PROJECT_ROOT / "10_Backups"
CONFIG_DIR = PROJECT_ROOT / "11_Configs"
RESULT_DIR = PROJECT_ROOT / "07_Results"
VISUALIZATION_DIR = PROJECT_ROOT / "08_Visualizations"
TEMP_DIR = PROJECT_ROOT / "14_Temp"

# =============================================================================
# DATASET FILES
# =============================================================================

RESEARCH_YAML = (
    PREPARED_DATASET /
    "research" /
    "dataset.yaml"
)

BENCHMARK_YAML = (
    PREPARED_DATASET /
    "benchmark" /
    "dataset.yaml"
)

# =============================================================================
# CLASS INFORMATION
# =============================================================================

CLASS_NAMES = {
    0: "healthy_banana",
    1: "unhealthy_banana"
}

NUM_CLASSES = len(CLASS_NAMES)

# =============================================================================
# PROJECT VERIFICATION
# =============================================================================

def verify_project():

    required = [
        PROJECT_ROOT,
        DATASET_ROOT,
        REPOSITORY_ROOT,
        ULTRALYTICS_REPO,
        LOG_DIR,
        SCRIPT_DIR,
        CONFIG_DIR
    ]

    return all(path.exists() for path in required)
