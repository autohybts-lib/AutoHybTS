"""Registry of model factories (architecture §12.3).

There is no process-wide registry: ``default_registry()`` builds a new one with the
built-in models on every call, and custom models are registered on an instance that is
then handed to the experiment. Factories receive a parameter mapping and must return a
fresh, unfitted object every time.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Final, Literal, TypeAlias

import pandas as pd

from autohybts.adapters.base import BaseForecaster
from autohybts.exceptions import ConfigurationError
from autohybts.models.requirements import require
from autohybts.registry.capabilities import Capabilities

__all__ = ["FAMILIES", "ModelEntry", "ModelKind", "ModelRegistry", "default_registry"]

ModelKind: TypeAlias = Literal["base", "corrector"]
Factory: TypeAlias = Callable[[Mapping[str, Any]], Any]

# Default family order of the incremental selection (§24 `family_order`).
FAMILIES: Final = ("statistical", "linear", "kernel", "tree", "neural")
_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True, slots=True)
class ModelEntry:
    name: str
    kind: ModelKind
    family: str
    factory: Factory
    capabilities: Capabilities
    description: str = ""
    default_params: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    # Models registered by the user (PyTorch, TensorFlow ...): library versions they need,
    # the version of the registration (part of every trial key) and a tune_model space.
    requires: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))
    version: str | None = None
    tuning_space: Mapping[str, Any] | None = None

    @property
    def external(self) -> bool:
        return self.version is not None

    def describe(self) -> dict[str, Any]:
        """Identity of a user-registered model, as it enters the evaluator protocol."""

        return {
            "kind": self.kind,
            "family": self.family,
            "version": self.version,
            "requires": dict(self.requires),
            "default_params": dict(self.default_params),
        }


class ModelRegistry:
    def __init__(self) -> None:
        self._entries: dict[tuple[ModelKind, str], ModelEntry] = {}

    def register_base(
        self,
        name: str,
        factory: Factory,
        *,
        family: str,
        capabilities: Capabilities | None = None,
        description: str = "",
        default_params: Mapping[str, Any] | None = None,
        replace: bool = False,
        requires: Mapping[str, str] | None = None,
        version: str | None = None,
        tuning_space: Mapping[str, Any] | None = None,
    ) -> None:
        """Register a base forecaster; the factory output is checked against the contract.

        ``requires`` (e.g. ``{"torch": ">=2.2"}``) is checked whenever the model is built:
        if it is not met the trial is ``SKIPPED`` with the reason. ``version`` names the
        registration (change it when the model code changes: it enters every trial key).
        ``tuning_space`` is the default space of ``tune_model`` for this model.
        """

        capabilities = capabilities or Capabilities()
        params = dict(default_params or {})
        probe = self._probe(name, factory, params)
        missing = [
            method
            for method in ("fit", "predict", "clone", "get_params")
            if not callable(getattr(probe, method, None))
        ]
        if missing or not isinstance(probe, BaseForecaster):
            raise ConfigurationError(
                f"O modelo {name!r} não cumpre o contrato BaseForecaster; faltam: {missing}."
            )
        if capabilities.in_sample_fitted and not callable(getattr(probe, "fitted_values", None)):
            raise ConfigurationError(
                f"{name!r} declara in_sample_fitted mas não implementa fitted_values()."
            )
        self._add(
            ModelEntry(
                name,
                "base",
                family,
                factory,
                capabilities,
                description,
                MappingProxyType(params),
                MappingProxyType(dict(requires or {})),
                version,
                tuning_space,
            ),
            replace,
        )

    def register_corrector(
        self,
        name: str,
        factory: Factory,
        *,
        capabilities: Capabilities | None = None,
        description: str = "",
        default_params: Mapping[str, Any] | None = None,
        replace: bool = False,
        requires: Mapping[str, str] | None = None,
        version: str | None = None,
        tuning_space: Mapping[str, Any] | None = None,
    ) -> None:
        """Register a residual corrector; the factory output is checked against the contract.

        ``requires``, ``version`` and ``tuning_space`` as in ``register_base``.
        """

        from autohybts.residuals.correctors import ResidualCorrector

        params = dict(default_params or {})
        probe = self._probe(name, factory, params)
        if not isinstance(probe, ResidualCorrector):
            raise ConfigurationError(
                f"O corretor {name!r} não cumpre o contrato ResidualCorrector "
                "(fit, predict_next, clone, get_params)."
            )
        self._add(
            ModelEntry(
                name,
                "corrector",
                "corrector",
                factory,
                capabilities or Capabilities(multi_step=False, online_state_update=False),
                description,
                MappingProxyType(params),
                MappingProxyType(dict(requires or {})),
                version,
                tuning_space,
            ),
            replace,
        )

    def make_corrector(self, name: str, params: Mapping[str, Any] | None = None) -> Any:
        """A new unfitted corrector; ``params`` override the entry's defaults."""

        from autohybts.residuals.correctors import ResidualCorrector

        entry = self.entry(name, "corrector")
        require(entry.requires)
        corrector = entry.factory({**entry.default_params, **dict(params or {})})
        if not isinstance(corrector, ResidualCorrector):
            raise ConfigurationError(f"A fábrica de {name!r} não devolveu um ResidualCorrector.")
        return corrector

    def make_base(self, name: str, params: Mapping[str, Any] | None = None) -> BaseForecaster:
        """A new unfitted instance; ``params`` override the entry's defaults."""

        entry = self.entry(name, "base")
        require(entry.requires)
        model = entry.factory({**entry.default_params, **dict(params or {})})
        if not isinstance(model, BaseForecaster):
            raise ConfigurationError(f"A fábrica de {name!r} não devolveu um BaseForecaster.")
        return model

    def entry(self, name: str, kind: ModelKind = "base") -> ModelEntry:
        try:
            return self._entries[(kind, name)]
        except KeyError:
            available = ", ".join(self.names(kind)) or "nenhum"
            raise ConfigurationError(
                f"Modelo {name!r} ({kind}) não registrado. Disponíveis: {available}."
            ) from None

    def names(self, kind: ModelKind = "base", family: str | None = None) -> tuple[str, ...]:
        return tuple(
            entry.name
            for (entry_kind, _), entry in sorted(self._entries.items())
            if entry_kind == kind and (family is None or entry.family == family)
        )

    def external(self) -> dict[str, dict[str, Any]]:
        """User-registered models (those with a ``version``), by ``kind:name``."""

        return {
            f"{kind}:{name}": entry.describe()
            for (kind, name), entry in sorted(self._entries.items())
            if entry.external
        }

    def families(self, kind: ModelKind = "base") -> tuple[str, ...]:
        present = {entry.family for (k, _), entry in self._entries.items() if k == kind}
        known = [family for family in FAMILIES if family in present]
        return (*known, *sorted(present - set(FAMILIES)))

    def copy(self) -> ModelRegistry:
        """Independent snapshot; later registrations on either side do not leak."""

        clone = ModelRegistry()
        clone._entries = dict(self._entries)
        return clone

    def __deepcopy__(self, memo: dict[int, Any]) -> ModelRegistry:
        # Entries are immutable; a deep copy is an independent registry of the same entries.
        return self.copy()

    def to_frame(self) -> pd.DataFrame:
        """Catalog shown by ``exp.models()``."""

        rows = [
            {
                "kind": entry.kind,
                "name": entry.name,
                "family": entry.family,
                **entry.capabilities.as_dict(),
                "requires": ", ".join(f"{k}{v}" for k, v in entry.requires.items()) or None,
                "version": entry.version,
                "description": entry.description,
            }
            for _, entry in sorted(self._entries.items())
        ]
        return pd.DataFrame(rows).set_index(["kind", "name"])

    def __contains__(self, name: object) -> bool:
        return any(key[1] == name for key in self._entries)

    def _add(self, entry: ModelEntry, replace: bool) -> None:
        key = (entry.kind, entry.name)
        if key in self._entries and not replace:
            raise ConfigurationError(
                f"{entry.name!r} já está registrado; use replace=True para substituir."
            )
        self._entries[key] = entry

    @staticmethod
    def _probe(name: str, factory: Factory, params: Mapping[str, Any]) -> Any:
        if not isinstance(name, str) or not _NAME.match(name):
            raise ConfigurationError(
                f"Nome de modelo inválido {name!r}: use minúsculas, dígitos e '_'."
            )
        if not callable(factory):
            raise ConfigurationError(f"A fábrica de {name!r} não é chamável.")
        return factory(params)


def default_registry() -> ModelRegistry:
    """A new registry with the built-in base models and correctors (§8.1)."""

    from autohybts.adapters.naive import NaiveForecaster, SeasonalNaiveForecaster
    from autohybts.adapters.statsmodels_arima import ArimaForecaster

    registry = ModelRegistry()
    registry.register_base(
        "naive",
        lambda p: NaiveForecaster(**p),
        family="statistical",
        description="Repete o último valor observado.",
    )
    registry.register_base(
        "seasonal_naive",
        lambda p: SeasonalNaiveForecaster(**p),
        family="statistical",
        description="Repete a última temporada (season_length).",
    )
    registry.register_base(
        "arima",
        lambda p: ArimaForecaster(**p),
        family="statistical",
        capabilities=Capabilities(in_sample_fitted=True, requires_extra="stats"),
        description="ARIMA (statsmodels); ordem automática por KPSS e AIC.",
    )
    lag_models = {
        "linear": ("linear", True, "Regressão linear sobre lags."),
        "knn": ("kernel", True, "k vizinhos mais próximos sobre lags."),
        "svm": ("kernel", True, "SVR sobre lags."),
        "tree": ("tree", True, "Árvore de decisão sobre lags."),
        "random_forest": ("tree", False, "Floresta aleatória sobre lags."),
        "mlp": ("neural", False, "Perceptron multicamadas sobre lags."),
    }
    for name, (family, deterministic, description) in lag_models.items():
        registry.register_base(
            name,
            _lag_factory(name),
            family=family,
            capabilities=Capabilities(deterministic=deterministic),
            description=description,
        )
    for name, (deterministic, description) in {
        "linear": (True, "Regressão linear sobre janelas de resíduos."),
        "knn": (True, "k vizinhos mais próximos sobre janelas de resíduos."),
        "mlp": (False, "Perceptron multicamadas sobre janelas de resíduos."),
    }.items():
        registry.register_corrector(
            name,
            _corrector_factory(name),
            capabilities=Capabilities(
                multi_step=False, online_state_update=False, deterministic=deterministic
            ),
            description=description,
        )
    return registry


def _corrector_factory(estimator: str) -> Factory:
    from autohybts.residuals.correctors import SklearnCorrector

    def factory(params: Mapping[str, Any]) -> SklearnCorrector:
        if params.get("estimator", estimator) != estimator:
            raise ConfigurationError(
                f"O corretor {estimator!r} não aceita estimator={params['estimator']!r}."
            )
        return SklearnCorrector(**{**params, "estimator": estimator})

    return factory


def _lag_factory(estimator: str) -> Factory:
    from autohybts.adapters.sklearn_regression import LagRegressionForecaster

    def factory(params: Mapping[str, Any]) -> LagRegressionForecaster:
        # get_params() includes `estimator`, so it may come back in params.
        if params.get("estimator", estimator) != estimator:
            raise ConfigurationError(
                f"O modelo {estimator!r} não aceita estimator={params['estimator']!r}."
            )
        return LagRegressionForecaster(**{**params, "estimator": estimator})

    return factory
