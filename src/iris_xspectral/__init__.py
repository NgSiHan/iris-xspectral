"""Cross-spectral iris verification pipeline glue code."""

import os
from pathlib import Path

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_paths():
    """Load the active path config based on IRIS_ENV (default: windows)."""
    env = os.environ.get("IRIS_ENV", "windows")
    config_path = _PROJECT_ROOT / "configs" / f"paths.{env}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Path config not found: {config_path}  (IRIS_ENV={env})"
        )
    with open(config_path) as f:
        paths = yaml.safe_load(f)
    # Expand ~ and make relative paths absolute from project root
    for key, val in paths.items():
        if isinstance(val, str):
            val = os.path.expanduser(val)
            if not os.path.isabs(val):
                val = str(_PROJECT_ROOT / val)
            paths[key] = val
    return paths
