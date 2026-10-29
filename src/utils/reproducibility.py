"""Helpers that make every result directory self-describing.

Each experiment writes, next to its numbers and figures:

    config.yaml    the exact settings it used
    metadata.json  creation time, command line, run time, Python and package
                   versions, platform, and the git commit (if the code is in a
                   git repository)

(Planned as ``utils/logging.py``; renamed so it cannot be confused with
Python's standard ``logging`` module.)
"""

from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import yaml

PACKAGES = ("numpy", "scipy", "matplotlib", "PyYAML", "pytest", "torch")


def load_config(path: str | Path) -> dict:
    """Read a YAML configuration file."""
    with open(path) as f:
        return yaml.safe_load(f)


def save_yaml(obj: dict, path: str | Path) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def save_json(obj, path: str | Path) -> None:
    """Write JSON, converting numpy scalars and arrays to plain Python types."""
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_to_builtin)
        f.write("\n")


def save_csv(path: str | Path, columns: dict[str, np.ndarray]) -> None:
    """Write equal-length 1-D arrays as CSV columns, using the dict keys as the header."""
    names = list(columns)
    arrays = [np.asarray(columns[name]) for name in names]
    n_rows = len(arrays[0])
    if any(len(a) != n_rows for a in arrays):
        raise ValueError("all columns must have the same length")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(names)
        for i in range(n_rows):
            writer.writerow([a[i].item() for a in arrays])


def git_info(repo_root: str | Path) -> dict:
    """Current commit and whether tracked files have uncommitted changes.

    Untracked files (such as freshly generated results) do not count as changes.
    Both values are None if repo_root is not the top level of a git repository,
    if the repository has no commits yet, or if git is not installed.
    """
    repo_root = Path(repo_root).resolve()

    def git(*args: str) -> str:
        result = subprocess.run(["git", *args], cwd=repo_root, capture_output=True, text=True, check=True)
        return result.stdout.strip()

    try:
        if Path(git("rev-parse", "--show-toplevel")).resolve() != repo_root:
            return {"commit": None, "dirty": None}
        return {
            "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def environment_info() -> dict:
    """Python version, platform and installed versions of the packages this project uses."""
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": versions}


def write_run_record(out_dir: str | Path, config: dict, repo_root: str | Path, runtime_seconds: float) -> None:
    """Write config.yaml and metadata.json into out_dir."""
    out_dir = Path(out_dir)
    save_yaml(config, out_dir / "config.yaml")
    save_json(
        {
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "command": sys.argv,
            "runtime_seconds": round(runtime_seconds, 3),
            "git": git_info(repo_root),
            "environment": environment_info(),
        },
        out_dir / "metadata.json",
    )


def _to_builtin(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"cannot convert {type(obj).__name__} to JSON")
