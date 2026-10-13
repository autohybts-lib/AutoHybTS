"""Build the forecaster a candidate spec describes, with per-unit seeds (§18)."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, TypeVar

from autohybts.adapters.base import BaseForecaster
from autohybts.domain.identity import derive_seed
from autohybts.domain.specs import CandidateSpec, ModelSpec
from autohybts.exceptions import ConfigurationError, UnsupportedCapabilityError
from autohybts.registry.registry import ModelRegistry
from autohybts.residuals.correctors import ResidualCorrector
from autohybts.residuals.generation import ResidualSettings
from autohybts.strategies.additive_residual import ResidualHybridForecaster
from autohybts.strategies.combination import CorrectorCombination, FixedCombinationForecaster
from autohybts.strategies.decomposition import DecompositionForecaster
from autohybts.strategies.dynamic import DynamicSelectionForecaster
from autohybts.strategies.ensemble import EnsembleForecaster
from autohybts.strategies.multistage import stack_stages

__all__ = ["build_model"]

T = TypeVar("T")


def build_model(
    candidate: CandidateSpec,
    registry: ModelRegistry,
    *,
    unit_seed: int,
    residual_settings: ResidualSettings | None = None,
) -> tuple[BaseForecaster, int | None]:
    """Fresh, unfitted model for ``candidate`` and the seed it received (``None`` if none).

    Components that accept ``random_state`` and do not fix it in their spec get a seed
    derived from ``unit_seed`` (base and corrector get different ones).
    """

    if candidate.strategy == "standalone":
        model, seeded = _seeded(registry.make_base, candidate.base, unit_seed)
        return model, seeded
    if candidate.strategy == "residual_additive":
        if candidate.corrector is None:  # guaranteed by CandidateSpec; keeps mypy honest
            raise ConfigurationError("Candidata residual sem corretor.")
        base, base_seed = _seeded(
            registry.make_base, candidate.base, derive_seed(unit_seed, "base")
        )
        corrector, corrector_seed = _seeded(
            registry.make_corrector, candidate.corrector, derive_seed(unit_seed, "corrector")
        )
        hybrid = ResidualHybridForecaster(
            base,
            corrector,
            window_size=int(candidate.window_size or 0),
            alpha=float(candidate.alpha if candidate.alpha is not None else 1.0),
            residual_source=str(candidate.residual_source),
            settings=residual_settings,
        )
        used = base_seed is not None or corrector_seed is not None
        return hybrid, unit_seed if used else None
    if candidate.strategy == "residual_multistage":
        if candidate.corrector is None:
            raise ConfigurationError("Candidata em estágios sem corretor.")
        base, base_seed = _seeded(
            registry.make_base, candidate.base, derive_seed(unit_seed, "base")
        )
        correctors = []
        seeded = base_seed is not None
        for stage, spec in enumerate((candidate.corrector, *candidate.members), start=1):
            label = "corrector" if stage == 1 else f"corrector-{stage}"
            corrector, corrector_seed = _seeded(
                registry.make_corrector, spec, derive_seed(unit_seed, label)
            )
            correctors.append(corrector)
            seeded = seeded or corrector_seed is not None
        model = stack_stages(
            base,
            correctors,
            window_size=int(candidate.window_size or 0),
            alpha=float(candidate.alpha if candidate.alpha is not None else 1.0),
            residual_source=str(candidate.residual_source),
            settings=residual_settings,
            stage_params=candidate.options.get("stages"),
        )
        return model, unit_seed if seeded else None
    if candidate.strategy == "decomposition":
        if candidate.corrector is None:
            raise ConfigurationError("Decomposição sem modelo da parte volátil.")
        options = candidate.options
        smooth, smooth_seed = _seeded(
            registry.make_base, candidate.base, derive_seed(unit_seed, "base")
        )
        volatile, volatile_seed = _seeded(
            registry.make_corrector, candidate.corrector, derive_seed(unit_seed, "corrector")
        )
        decomposition = DecompositionForecaster(
            smooth,
            volatile,
            window_size=int(candidate.window_size or 0),
            method=str(options["method"]),
            ma_window=options.get("ma_window"),
            period=int(options.get("period") or 7),
        )
        used = smooth_seed is not None or volatile_seed is not None
        return decomposition, unit_seed if used else None
    if candidate.strategy == "fixed_combination":
        members = []
        seeded = False
        for index, spec in enumerate((candidate.base, *candidate.members)):
            member, member_seed = _seeded(
                registry.make_base, spec, derive_seed(unit_seed, f"member-{index}")
            )
            members.append(member)
            seeded = seeded or member_seed is not None
        combination = FixedCombinationForecaster(
            members, weights=str(candidate.options["weights"]), settings=residual_settings
        )
        return combination, unit_seed if seeded else None
    if candidate.strategy == "residual_combination":
        if candidate.corrector is None:
            raise ConfigurationError("Combinação de corretores sem corretor.")
        base, base_seed = _seeded(
            registry.make_base, candidate.base, derive_seed(unit_seed, "base")
        )
        pool: list[ResidualCorrector] = []
        seeded = base_seed is not None
        for index, spec in enumerate((candidate.corrector, *candidate.members)):
            corrector, corrector_seed = _seeded(
                registry.make_corrector, spec, derive_seed(unit_seed, f"corrector-{index}")
            )
            pool.append(corrector)
            seeded = seeded or corrector_seed is not None
        combined = ResidualHybridForecaster(
            base,
            CorrectorCombination(pool, weights=str(candidate.options["weights"])),
            window_size=int(candidate.window_size or 0),
            alpha=float(candidate.alpha if candidate.alpha is not None else 1.0),
            residual_source=str(candidate.residual_source),
            settings=residual_settings,
        )
        return combined, unit_seed if seeded else None
    if candidate.strategy == "dynamic_selection":
        if candidate.corrector is None:
            raise ConfigurationError("Seleção dinâmica sem corretores.")
        options = candidate.options
        base, base_seed = _seeded(
            registry.make_base, candidate.base, derive_seed(unit_seed, "base")
        )
        pool = []
        seeded = base_seed is not None
        for index, spec in enumerate((candidate.corrector, *candidate.members)):
            corrector, corrector_seed = _seeded(
                registry.make_corrector, spec, derive_seed(unit_seed, f"pool-{index}")
            )
            pool.append(corrector)
            seeded = seeded or corrector_seed is not None
        dynamic = DynamicSelectionForecaster(
            base,
            pool,
            window_size=int(candidate.window_size or 0),
            method=str(options.get("method", "dref")),
            neighbors=int(options.get("neighbors", 10)),
            lag_size=int(options.get("lag_size", 7)),
            dsel_size=options.get("dsel_size"),
            local_alpha=float(options.get("local_alpha", 1.0)),
            residual_source=str(candidate.residual_source),
            settings=residual_settings,
        )
        return dynamic, unit_seed if seeded else None
    if candidate.strategy == "ensemble":
        components = []
        seeded = False
        for index, component in enumerate(candidate.components):
            model, component_seed = build_model(
                component,
                registry,
                unit_seed=derive_seed(unit_seed, f"component-{index}"),
                residual_settings=residual_settings,
            )
            components.append(model)
            seeded = seeded or component_seed is not None
        options = candidate.options
        ensemble = EnsembleForecaster(
            components,
            method=str(options["method"]),
            trim=float(options.get("trim", 0.2)),
            shrinkage=float(options.get("shrinkage", 0.5)),
            settings=residual_settings,
        )
        return ensemble, unit_seed if seeded else None
    if candidate.strategy == "residual_composite":
        if candidate.corrector is None:
            raise ConfigurationError("Composição residual sem corretor.")
        inner, inner_seed = build_model(
            candidate.components[0],
            registry,
            unit_seed=derive_seed(unit_seed, "component-0"),
            residual_settings=residual_settings,
        )
        corrector, corrector_seed = _seeded(
            registry.make_corrector, candidate.corrector, derive_seed(unit_seed, "corrector")
        )
        hybrid = ResidualHybridForecaster(
            inner,
            corrector,
            window_size=int(candidate.window_size or 0),
            alpha=float(candidate.alpha if candidate.alpha is not None else 1.0),
            residual_source=str(candidate.residual_source),
            settings=residual_settings,
        )
        used = inner_seed is not None or corrector_seed is not None
        return hybrid, unit_seed if used else None
    raise UnsupportedCapabilityError(f"Estratégia {candidate.strategy!r} não implementada.")


def _seeded(
    make: Callable[[str, Mapping[str, Any] | None], T], spec: ModelSpec, seed: int
) -> tuple[T, int | None]:
    params = spec.params
    model = make(spec.name, params)
    if "random_state" in params:
        fixed = params["random_state"]
        return model, fixed if isinstance(fixed, int) else None
    if "random_state" not in model.get_params():  # type: ignore[attr-defined]
        return model, None
    return make(spec.name, {**params, "random_state": seed}), seed
