
"""
==============================================================================
Utilities
==============================================================================

Shared helper functions used throughout the project.

Responsibilities
----------------
✓ JSON save/load
✓ Timestamp generation
✓ Random seed initialization
✓ Pretty printing
✓ Directory creation

Author : Huzaifa
Version: 2.0
"""

from pathlib import Path
import json
import random
from datetime import datetime

import numpy as np
import torch


def timestamp() -> str:
    """Return current timestamp."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(path: Path) -> Path:
    """Create directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: dict, path: Path) -> Path:
    """Save dictionary as JSON."""
    ensure_dir(path.parent)

    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    return path


def load_json(path: Path) -> dict:
    """Load JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def set_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_header(title: str) -> None:
    """Print a formatted section header."""

    print("=" * 90)
    print(title)
    print("=" * 90)
