"""Version requirements of external libraries (PyTorch, TensorFlow, LightGBM ...).

A model registered by the user declares what it needs, e.g. ``{"torch": ">=2.2,<3"}``.
The check happens when the model is built for a trial, never on ``import autohybts``:
a missing or too old library makes the trial ``SKIPPED`` with the reason and the ``pip``
command, instead of breaking the experiment.

Specifiers and versions follow PEP 440 through the ``packaging`` library (ADR-0006), as
``pip check`` does for installed distributions: versions are ordered by PEP 440, so a
pre-release such as ``2.2.0rc1`` comes *before* ``2.2`` and does not satisfy ``>=2.2`` or
``==2.2``; an installed pre-release inside the range (``2.3.0rc1`` for ``>=2.2``) does.
``packaging`` is imported only here, when a requirement is checked; it ships with the
``torch`` and ``tensorflow`` extras.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from importlib import metadata
from typing import Any, Final

from autohybts.exceptions import ConfigurationError, UnsupportedCapabilityError

__all__ = [
    "check_specifier",
    "framework_versions",
    "installed_version",
    "require",
    "validate_requirements",
]

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
_NEEDS_PACKAGING: Final = (
    "Verificar requisitos de versão exige o pacote 'packaging': pip install packaging "
    '(já incluído em pip install "autohybts[torch]" e "autohybts[tensorflow]").'
)


def installed_version(package: str) -> str | None:
    """Installed version of an import name (``"torch"``, ``"tensorflow"`` ...), or ``None``."""

    for distribution in _DISTRIBUTIONS.get(package, (package,)):
        try:
            return metadata.version(distribution)
        except metadata.PackageNotFoundError:
            continue
    return None


def _specifier_set(specifier: str) -> Any:
    try:
        from packaging.specifiers import InvalidSpecifier, SpecifierSet
    except ModuleNotFoundError as error:  # pragma: no cover - packaging is in the dev env
        raise UnsupportedCapabilityError(_NEEDS_PACKAGING) from error
    try:
        return SpecifierSet(specifier)
    except InvalidSpecifier as error:
        raise ConfigurationError(
            f"Especificação de versão inválida {specifier!r} (PEP 440); use por exemplo "
            "'>=2.2,<3', '~=2.2' ou '!=2.3.*'."
        ) from error


def check_specifier(version: str, specifier: str) -> bool:
    """``True`` if the installed ``version`` satisfies a PEP 440 specifier like ``">=2.2,<3"``.

    An invalid specifier raises ``ConfigurationError``; a version that does not follow
    PEP 440 raises ``UnsupportedCapabilityError`` (it cannot be checked, so the model is
    skipped with the reason).
    """

    specifiers = _specifier_set(specifier)
    from packaging.version import InvalidVersion, Version

    try:
        parsed = Version(version)
    except InvalidVersion as error:
        raise UnsupportedCapabilityError(
            f"A versão instalada {version!r} não segue a PEP 440; não dá para verificar "
            f"{specifier!r}."
        ) from error
    return bool(specifiers.contains(parsed, prereleases=True))


def validate_requirements(requirements: Mapping[str, str] | None) -> None:
    """Refuse malformed requirements at registration (``ConfigurationError``).

    Without ``packaging`` installed the syntax cannot be checked here; ``require`` will
    then skip the model with the installation hint.
    """

    for package, specifier in (requirements or {}).items():
        if not isinstance(package, str) or not package or not isinstance(specifier, str):
            raise ConfigurationError(
                f"requires espera {{'pacote': 'especificação'}}; recebido {package!r}: "
                f"{specifier!r}."
            )
        if not specifier:
            continue
        try:
            _specifier_set(specifier)
        except UnsupportedCapabilityError:  # pragma: no cover - packaging is in the dev env
            return


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
