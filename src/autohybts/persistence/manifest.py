"""Environment and provenance recorded with every artifact (§17.3)."""

from __future__ import annotations

import importlib.metadata
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import autohybts

__all__ = ["environment", "git_commit", "utc_now"]

_PACKAGES = ("numpy", "pandas", "scikit-learn", "statsmodels", "scipy")


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def environment() -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    for package in _PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "autohybts": autohybts.__version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "dependencies": versions,
        "git_commit": git_commit(),
    }


def git_commit(cwd: Path | None = None) -> str | None:
    """Current commit if run inside a git work tree with commits; never raises."""

    try:
        output = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = output.stdout.strip()
    return commit or None
