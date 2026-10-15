"""Environment and provenance recorded with every artifact (§17.3)."""

from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import re
import sys
from collections.abc import Mapping
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np

import autohybts
from autohybts.domain.identity import stable_hash
from autohybts.exceptions import ConfigurationError

__all__ = ["environment", "git_commit", "json_safe", "source_fingerprint", "utc_now"]

_PACKAGES = (
    "numpy",
    "pandas",
    "scikit-learn",
    "statsmodels",
    "scipy",
    "statsforecast",
    "optuna",
    "torch",
    "tensorflow",
    "keras",
    "lightgbm",
    "xgboost",
    "catboost",
)
_SHA = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
_REF = re.compile(r"refs/[A-Za-z0-9._/+-]+")


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
    """Commit checked out in the git work tree around ``cwd``; never raises.

    Reads ``.git`` directly instead of running ``git``: on Windows a bare ``git`` would be
    looked up in the current directory first, so a planted ``git.exe`` next to the user's
    data would run on every ``save_model``.
    """

    try:
        git_dir = _git_dir((cwd or Path.cwd()).resolve())
        if git_dir is None:
            return None
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        if not head.startswith("ref:"):
            return head if _SHA.fullmatch(head) else None
        ref = head[4:].strip()
        if not _REF.fullmatch(ref) or ".." in ref:
            return None
        common = git_dir
        if (git_dir / "commondir").is_file():
            common = git_dir / (git_dir / "commondir").read_text(encoding="utf-8").strip()
        for folder in dict.fromkeys((git_dir, common)):
            loose = folder / ref
            if loose.is_file():
                value = loose.read_text(encoding="utf-8").strip()
                return value if _SHA.fullmatch(value) else None
        packed = common / "packed-refs"
        if packed.is_file():
            for line in packed.read_text(encoding="utf-8").splitlines():
                value, _, name = line.partition(" ")
                if name.strip() == ref and _SHA.fullmatch(value):
                    return value
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    return None


def _git_dir(start: Path) -> Path | None:
    for folder in (start, *start.parents):
        dot_git = folder / ".git"
        if dot_git.is_dir():
            return dot_git
        if dot_git.is_file():  # worktree or submodule: "gitdir: <path>"
            text = dot_git.read_text(encoding="utf-8").strip()
            if text.startswith("gitdir:"):
                return (folder / text[7:].strip()).resolve()
            return None
    return None


def json_safe(value: Any) -> Any:
    """JSON representation for reporting, with undefined numbers explicitly null."""

    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    raise ConfigurationError(f"Tipo não serializável no protocolo: {type(value).__name__}.")


def source_fingerprint() -> str:
    """Hash the installed package's Python sources, including uncommitted edits."""

    root = Path(autohybts.__file__).resolve().parent
    files = sorted(root.rglob("*.py"))
    return stable_hash(
        [
            (file.relative_to(root).as_posix(), hashlib.sha256(file.read_bytes()).hexdigest())
            for file in files
        ]
    )
