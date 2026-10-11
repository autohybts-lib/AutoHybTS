"""Version requirements of external libraries (PyTorch, TensorFlow, LightGBM ...).

A model registered by the user declares what it needs, e.g. ``{"torch": ">=2.2,<3"}``.
The check happens when the model is built for a trial, never on ``import autohybts``:
a missing or too old library makes the trial ``SKIPPED`` with the reason and the ``pip``
command, instead of breaking the experiment.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from importlib import metadata
from typing import Final

from autohybts.exceptions import ConfigurationError, UnsupportedCapabilityError

__all__ = ["check_specifier", "framework_versions", "installed_version", "require"]

# Import name -> distributions that provide it (the first one installed wins).
_DISTRIBUTIONS: Final = {
    "torch": ("torch",),
    "tensorflow": ("tensorflow", "tensorflow-cpu", "tensorflow-intel", "tensorflow-macos"),
    "keras": ("keras",),
    "lightgbm": ("lightgbm",),
    "xgboost": ("xgboost",),
    "catboost": ("catboost",),
    "sklearn": ("scikit-learn",),
}
_CLAUSE = re.compile(r"^\s*(==|!=|>=|<=|>|<|~=)\s*([0-9][0-9A-Za-z.+!-]*)\s*$")


def installed_version(package: str) -> str | None:
    """Installed version of an import name (``"torch"``, ``"tensorflow"`` ...), or ``None``."""

    for distribution in _DISTRIBUTIONS.get(package, (package,)):
        try:
            return metadata.version(distribution)
        except metadata.PackageNotFoundError:
            continue
    return None


def _release(version: str) -> tuple[int, ...]:
    """Numeric release part: ``2.5.1+cpu`` -> (2, 5, 1); pre-release tags are ignored."""

    numbers = []
    for part in re.split(r"[.+!-]", version):
        digits = re.match(r"\d+", part)
        if digits is None:
            break
        numbers.append(int(digits.group()))
        if digits.group() != part:
            break
    return tuple(numbers)


def _compare(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    width = max(len(left), len(right))
    a = left + (0,) * (width - len(left))
    b = right + (0,) * (width - len(right))
    return (a > b) - (a < b)


def check_specifier(version: str, specifier: str) -> bool:
    """``True`` if ``version`` satisfies a comma-separated specifier like ``">=2.2,<3"``."""

    installed = _release(version)
    for clause in filter(None, (c.strip() for c in specifier.split(","))):
        match = _CLAUSE.match(clause)
        if match is None:
            raise ConfigurationError(
                f"Especificação de versão inválida {clause!r}; use por exemplo '>=2.2,<3'."
            )
        operator, bound_text = match.groups()
        bound = _release(bound_text)
        order = _compare(installed, bound)
        if operator == "~=":
            prefix = bound[:-1] if len(bound) > 1 else bound
            ok = order >= 0 and installed[: len(prefix)] == prefix
        else:
            ok = {
                "==": order == 0,
                "!=": order != 0,
                ">=": order >= 0,
                "<=": order <= 0,
                ">": order > 0,
                "<": order < 0,
            }[operator]
        if not ok:
            return False
    return True


def require(requirements: Mapping[str, str] | None) -> None:
    """Raise ``UnsupportedCapabilityError`` unless every requirement is installed."""

    for package, specifier in (requirements or {}).items():
        version = installed_version(package)
        install = f'pip install "{_DISTRIBUTIONS.get(package, (package,))[0]}{specifier}"'
        if version is None:
            raise UnsupportedCapabilityError(
                f"O modelo exige {package}{specifier}, que não está instalado: {install}."
            )
        if specifier and not check_specifier(version, specifier):
            raise UnsupportedCapabilityError(
                f"O modelo exige {package}{specifier}; instalado {version}: {install}."
            )


def framework_versions(packages: Iterable[str]) -> dict[str, str | None]:
    """Installed versions, for manifests and reports."""

    return {package: installed_version(package) for package in sorted(set(packages))}
