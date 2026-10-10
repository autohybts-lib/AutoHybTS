"""Immutable candidate specifications and their stable identities (§5.3, §8.4)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Final

from autohybts.domain.identity import canonical_json, stable_hash
from autohybts.exceptions import ConfigurationError
from autohybts.types import PROTOCOL_VARIANTS, RESIDUAL_SOURCES, ProtocolVariant

__all__ = [
    "COMBINATION_WEIGHTS",
    "DECOMPOSITION_METHODS",
    "MAX_STAGES",
    "STRATEGIES",
    "CandidateSpec",
    "ModelSpec",
    "trial_key",
]

# dynamic_selection is reserved: allowed only in variant B (R07, T16), built in I13.
STRATEGIES: Final = (
    "standalone",
    "residual_additive",
    "residual_multistage",
    "decomposition",
    "fixed_combination",
    "residual_combination",
    "dynamic_selection",
)
_RESIDUAL_FIELDS: Final = ("corrector", "window_size", "alpha", "residual_source")
# Corrections of corrections are capped (§16): each stage adds a model fitted on fewer rows.
MAX_STAGES: Final = 3
DECOMPOSITION_METHODS: Final = ("ma", "stl")
COMBINATION_WEIGHTS: Final = ("equal", "inverse_mse")


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """A registered model name plus its parameters, stored in canonical form.

    Parameters are kept as canonical JSON, so the spec is deeply immutable and two specs
    with equal parameters (in any key order, tuples or lists) are equal.
    """

    name: str
    params_json: str = "{}"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ConfigurationError("ModelSpec precisa de um nome.")
        params = json.loads(self.params_json)
        if not isinstance(params, dict):
            raise ConfigurationError("Parâmetros de ModelSpec devem ser um objeto JSON.")
        object.__setattr__(self, "params_json", canonical_json(params))

    @classmethod
    def of(cls, name: str, params: Mapping[str, Any] | None = None) -> ModelSpec:
        return cls(name, canonical_json(dict(params or {})))

    @property
    def params(self) -> dict[str, Any]:
        """A fresh copy of the parameters."""

        loaded: dict[str, Any] = json.loads(self.params_json)
        return loaded

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "params": self.params}

    @property
    def label(self) -> str:
        params = self.params
        if not params:
            return self.name
        inner = ", ".join(f"{key}={value}" for key, value in sorted(params.items()))
        return f"{self.name}({inner})"


@dataclass(frozen=True, slots=True)
class CandidateSpec:
    """One configuration to evaluate: strategy, base, optional corrector, variant.

    Strategies and the fields they use:

    * ``standalone``: ``base`` only.
    * ``residual_additive``: ``base + alpha * corrector`` on residual windows.
    * ``residual_multistage``: as above, then each corrector in ``members`` corrects the
      residuals of the previous stage; at most ``MAX_STAGES`` stages. Extra stages share
      window, alpha and source unless ``options["stages"]`` gives, per extra stage, a
      dict with any of ``window_size``, ``alpha`` and ``residual_source``.
    * ``decomposition``: ``base`` forecasts the smooth part and ``corrector`` the volatile
      part on windows of ``window_size``; ``options`` holds ``method`` (``"ma"`` or
      ``"stl"``, the latter only in variant A), ``ma_window`` (``None``: chosen by
      kurtosis on the training vector) and ``period`` (seasonal period, required by STL).
    * ``fixed_combination``: ``base`` and ``members`` combined with fixed weights;
      ``options["weights"]`` is ``"equal"`` or ``"inverse_mse"`` (estimated on the
      training vector only).
    * ``residual_combination``: ``base + alpha * Σ w_j r̂_j``, the corrector and the
      correctors in ``members`` combined with fixed weights (``options["weights"]``).

    ``members`` and ``options`` enter ``candidate_id`` only when non-empty, so identities
    of the other strategies do not change.
    """

    family: str
    strategy: str
    base: ModelSpec
    corrector: ModelSpec | None = None
    window_size: int | None = None
    alpha: float | None = None
    residual_source: str | None = None
    variant: ProtocolVariant = "A"
    members: tuple[ModelSpec, ...] = ()
    options_json: str = "{}"
    candidate_id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.strategy not in STRATEGIES:
            raise ConfigurationError(
                f"Estratégia desconhecida {self.strategy!r}; opções: {STRATEGIES}."
            )
        if self.variant not in PROTOCOL_VARIANTS:
            raise ConfigurationError(f"Variante deve ser 'A' ou 'B'; recebido {self.variant!r}.")
        if self.strategy == "dynamic_selection" and self.variant != "B":
            raise ConfigurationError(
                "A seleção dinâmica só existe na variante B: em A não há observação nova entre "
                "os passos para escolher o especialista (R07)."
            )
        if not isinstance(self.family, str) or not self.family:
            raise ConfigurationError("CandidateSpec precisa de uma família.")
        object.__setattr__(self, "members", tuple(self.members))
        if not all(isinstance(member, ModelSpec) for member in self.members):
            raise ConfigurationError("members deve conter apenas ModelSpec.")
        options = json.loads(self.options_json)
        if not isinstance(options, dict):
            raise ConfigurationError("options de CandidateSpec deve ser um objeto JSON.")
        object.__setattr__(self, "options_json", canonical_json(options))
        self._validate_fields(options)
        object.__setattr__(
            self, "candidate_id", stable_hash({"schema": "candidate/1", **self.describe()})
        )

    def _validate_fields(self, options: dict[str, Any]) -> None:
        residual = {name: getattr(self, name) for name in _RESIDUAL_FIELDS}
        given = [name for name, value in residual.items() if value is not None]
        missing = [name for name, value in residual.items() if value is None]
        strategy = self.strategy
        if strategy in ("standalone", "fixed_combination"):
            if given:
                raise ConfigurationError(f"Candidata {strategy} não aceita {given}.")
        elif strategy == "decomposition":
            for name in ("corrector", "window_size"):
                if residual[name] is None:
                    raise ConfigurationError(
                        f"Decomposição exige {name} (modelo da parte volátil)."
                    )
            extra = [name for name in ("alpha", "residual_source") if residual[name] is not None]
            if extra:
                raise ConfigurationError(f"Decomposição não usa {extra}.")
        else:
            if missing:
                raise ConfigurationError(f"Candidata residual exige {missing}.")
            if self.residual_source not in RESIDUAL_SOURCES:
                raise ConfigurationError(
                    f"Fonte residual desconhecida {self.residual_source!r}; opções: "
                    f"{RESIDUAL_SOURCES}."
                )
        if strategy == "residual_combination":
            if not self.members:
                raise ConfigurationError(
                    "residual_combination exige ao menos um corretor em members."
                )
            names = [self.corrector, *self.members]
            if len(set(names)) != len(names):
                raise ConfigurationError("Corretores repetidos na combinação.")
        elif strategy == "residual_multistage":
            if not 1 <= len(self.members) <= MAX_STAGES - 1:
                raise ConfigurationError(
                    f"residual_multistage tem de 2 a {MAX_STAGES} estágios: informe de 1 a "
                    f"{MAX_STAGES - 1} corretores extras em members."
                )
        elif strategy == "fixed_combination":
            if not self.members:
                raise ConfigurationError("fixed_combination exige ao menos um membro além da base.")
            names = [self.base, *self.members]
            if len(set(names)) != len(names):
                raise ConfigurationError("Membros repetidos na combinação.")
        elif self.members and strategy not in ("dynamic_selection", "residual_combination"):
            raise ConfigurationError(f"Candidata {strategy} não aceita members.")
        allowed = {
            "decomposition": {"method", "ma_window", "period"},
            "fixed_combination": {"weights"},
            "residual_combination": {"weights"},
            "residual_multistage": {"stages"},
            "dynamic_selection": {"method", "neighbors", "lag_size", "dsel_size", "local_alpha"},
        }.get(strategy, set())
        unknown = set(options) - allowed
        if unknown:
            raise ConfigurationError(f"Opções desconhecidas para {strategy}: {sorted(unknown)}.")
        if strategy == "decomposition":
            method = options.get("method")
            if method not in DECOMPOSITION_METHODS:
                raise ConfigurationError(
                    f"Decomposição exige method em {DECOMPOSITION_METHODS}; recebido {method!r}."
                )
            if method == "stl" and self.variant != "A":
                raise ConfigurationError(
                    "A STL só existe na variante A: refazê-la a cada observação custaria uma "
                    "decomposição por passo, e mantê-la fixa defasaria o componente."
                )
            period = options.get("period")
            if method == "stl" and (
                isinstance(period, bool) or not isinstance(period, int) or period < 2
            ):
                raise ConfigurationError("A STL exige period inteiro >= 2 nas opções.")
            ma_window = options.get("ma_window")
            if ma_window is not None and (
                isinstance(ma_window, bool) or not isinstance(ma_window, int) or ma_window < 1
            ):
                raise ConfigurationError("ma_window deve ser inteiro >= 1 ou None.")
        if strategy == "residual_multistage" and "stages" in options:
            _check_stages(options["stages"], len(self.members))
        if strategy == "dynamic_selection" and options.get("method", "dref") not in (
            "dref",
            "local",
        ):
            raise ConfigurationError("Seleção dinâmica exige method 'dref' ou 'local'.")
        if (
            strategy in ("fixed_combination", "residual_combination")
            and options.get("weights") not in COMBINATION_WEIGHTS
        ):
            raise ConfigurationError(
                f"fixed_combination exige weights em {COMBINATION_WEIGHTS}; recebido "
                f"{options.get('weights')!r}."
            )

    @property
    def options(self) -> dict[str, Any]:
        """A fresh copy of the strategy options."""

        loaded: dict[str, Any] = json.loads(self.options_json)
        return loaded

    @property
    def stages(self) -> int:
        """Number of correction stages (0 for strategies without a residual corrector)."""

        if self.strategy == "residual_additive":
            return 1
        if self.strategy == "residual_multistage":
            return 1 + len(self.members)
        if self.strategy == "residual_combination":
            return 1
        return 0

    @classmethod
    def standalone(
        cls,
        name: str,
        params: Mapping[str, Any] | None = None,
        *,
        family: str,
        variant: ProtocolVariant = "A",
    ) -> CandidateSpec:
        return cls(family, "standalone", ModelSpec.of(name, params), variant=variant)

    def describe(self) -> dict[str, Any]:
        """Canonical content of the spec; the source of ``candidate_id``."""

        described: dict[str, Any] = {
            "family": self.family,
            "strategy": self.strategy,
            "base": self.base.describe(),
            "corrector": None if self.corrector is None else self.corrector.describe(),
            "window_size": self.window_size,
            "alpha": self.alpha,
            "residual_source": self.residual_source,
            "variant": self.variant,
        }
        if self.members:
            described["members"] = [member.describe() for member in self.members]
        if self.options:
            described["options"] = self.options
        return described

    @property
    def name(self) -> str:
        """Short readable name of the structure, e.g. ``arima-mlp`` (parameters omitted).

        ``label`` keeps the full configuration; several configurations share a name.
        """

        if self.strategy == "fixed_combination":
            return "comb(" + ",".join(s.name for s in (self.base, *self.members)) + ")"
        if self.corrector is None:
            return self.base.name
        if self.strategy == "decomposition":
            return f"{self.options['method']}({self.base.name}-{self.corrector.name})"
        if self.strategy == "dynamic_selection":
            pool = ",".join(s.name for s in (self.corrector, *self.members))
            return f"{self.options.get('method', 'dref')}({self.base.name}-{pool})"
        if self.strategy == "residual_combination":
            pool = ",".join(s.name for s in (self.corrector, *self.members))
            return f"{self.base.name}-comb({pool})"
        return "-".join(s.name for s in (self.base, self.corrector, *self.members))

    @property
    def label(self) -> str:
        if self.strategy == "fixed_combination":
            names = " | ".join(spec.label for spec in (self.base, *self.members))
            return f"combinação[{names}; {self.options['weights']}]"
        if self.strategy == "decomposition" and self.corrector is not None:
            options = self.options
            method = options["method"]
            if method == "ma":
                window = options.get("ma_window")
                method = f"ma({'curtose' if window is None else window})"
            return f"{method}: {self.base.label} | {self.corrector.label} [w={self.window_size}]"
        if self.strategy == "residual_combination" and self.corrector is not None:
            pool = " | ".join(s.label for s in (self.corrector, *self.members))
            return (
                f"{self.base.label} + comb[{pool}; {self.options['weights']}] "
                f"[w={self.window_size}, alpha={self.alpha}, {self.residual_source}]"
            )
        if self.strategy == "dynamic_selection" and self.corrector is not None:
            options = self.options
            pool = " | ".join(s.label for s in (self.corrector, *self.members))
            return (
                f"{options.get('method', 'dref')}: {self.base.label} + {{{pool}}} "
                f"[w={self.window_size}, k={options.get('neighbors', 10)}, "
                f"{self.residual_source}]"
            )
        if self.strategy == "standalone" or self.corrector is None:
            return self.base.label
        correctors = " + ".join(spec.label for spec in (self.corrector, *self.members))
        stages = self.options.get("stages")
        if stages:
            extra = "; ".join(
                ", ".join(f"{k}={v}" for k, v in sorted(stage.items())) or "=" for stage in stages
            )
            correctors += f" {{estágios: {extra}}}"
        return (
            f"{self.base.label} + {correctors} "
            f"[w={self.window_size}, alpha={self.alpha}, {self.residual_source}]"
        )


_STAGE_KEYS: Final = frozenset({"window_size", "alpha", "residual_source"})


def _check_stages(stages: object, n_extra: int) -> None:
    """Per-stage settings: one dict per extra stage, with known keys and valid values."""

    if not isinstance(stages, list) or len(stages) != n_extra:
        raise ConfigurationError(
            f"options['stages'] deve ter um dicionário por estágio extra ({n_extra})."
        )
    for stage in stages:
        if not isinstance(stage, dict) or not set(stage) <= _STAGE_KEYS:
            raise ConfigurationError(
                f"Cada estágio aceita só {sorted(_STAGE_KEYS)}; recebido {stage!r}."
            )
        window = stage.get("window_size")
        if window is not None and (
            isinstance(window, bool) or not isinstance(window, int) or window < 1
        ):
            raise ConfigurationError("window_size do estágio deve ser inteiro >= 1.")
        alpha = stage.get("alpha")
        if alpha is not None and (isinstance(alpha, bool) or not isinstance(alpha, int | float)):
            raise ConfigurationError("alpha do estágio deve ser numérico.")
        source = stage.get("residual_source")
        if source is not None and source not in RESIDUAL_SOURCES:
            raise ConfigurationError(f"Fonte residual desconhecida no estágio: {source!r}.")


def trial_key(
    candidate: CandidateSpec,
    *,
    unit_id: str,
    seed: int | None,
    data_fingerprint: str,
    code_version: str,
    protocol: Mapping[str, Any] | None = None,
) -> str:
    """Cache key of one trial; any change in inputs, data, protocol or code changes it (§8.4).

    ``protocol`` holds evaluation settings shared by all candidates (metrics, seasonal
    period, residual generation), which also change the outcome of a trial.
    """

    return stable_hash(
        {
            "schema": "trial/1",
            "candidate_id": candidate.candidate_id,
            "unit_id": unit_id,
            "seed": seed,
            "data": data_fingerprint,
            "code": code_version,
            "protocol": dict(protocol or {}),
        }
    )
