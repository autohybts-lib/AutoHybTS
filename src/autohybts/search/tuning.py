"""Hyper-parameter tuning of one candidate on ``V_search`` only (§34.3 ``tune_model``).

The tuner changes the parameters of the candidate's base model and scores every trial on
the search layer; the selection layer and the sealed test never reach it, so the tuned
candidate can still be compared honestly on ``V_select`` afterwards. The original
configuration is always trial 0, so tuning cannot pick something worse on ``V_search``.

Samplers: ``"optuna"`` (TPE, optional extra ``optuna``), ``"random"`` (seeded uniform
draws, no extra dependency) and ``"auto"`` (Optuna when installed, random otherwise).
"""

from __future__ import annotations

import importlib.util
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any, Final, Literal, TypeAlias

import numpy as np
import pandas as pd

from autohybts.data.panel import PanelPlan
from autohybts.data.splitter import TemporalPlan, ensure_not_sealed
from autohybts.domain.identity import derive_seed
from autohybts.domain.specs import CandidateSpec, ModelSpec
from autohybts.evaluation.temporal import TemporalEvaluator
from autohybts.exceptions import ConfigurationError, UnsupportedCapabilityError

__all__ = [
    "LAG_SPACES",
    "PLAIN_SPACES",
    "TUNING_SPACES",
    "Distribution",
    "Tuner",
    "TuningResult",
    "default_space",
]

Sampler: TypeAlias = Literal["auto", "optuna", "random"]
Kind: TypeAlias = Literal["int", "float", "log_float", "categorical"]


@dataclass(frozen=True, slots=True)
class Distribution:
    """One tunable parameter. Dotted names (``estimator_params.C``) set nested keys."""

    kind: Kind
    low: float = 0.0
    high: float = 0.0
    choices: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        if self.kind == "categorical":
            if not self.choices:
                raise ConfigurationError("Distribuição categórica sem opções.")
        elif not self.low <= self.high:
            raise ConfigurationError(f"Intervalo inválido [{self.low}, {self.high}].")
        if self.kind == "log_float" and self.low <= 0:
            raise ConfigurationError("Escala log exige limites positivos.")

    def sample(self, rng: np.random.Generator) -> Any:
        if self.kind == "int":
            return int(rng.integers(int(self.low), int(self.high) + 1))
        if self.kind == "float":
            return float(rng.uniform(self.low, self.high))
        if self.kind == "log_float":
            return float(math.exp(rng.uniform(math.log(self.low), math.log(self.high))))
        return self.choices[int(rng.integers(len(self.choices)))]

    def suggest(self, trial: Any, name: str) -> Any:
        if self.kind == "int":
            return trial.suggest_int(name, int(self.low), int(self.high))
        if self.kind == "float":
            return trial.suggest_float(name, self.low, self.high)
        if self.kind == "log_float":
            return trial.suggest_float(name, self.low, self.high, log=True)
        return trial.suggest_categorical(name, list(self.choices))


def _lags(seasonal_period: int) -> Distribution:
    return Distribution("int", 2, min(60, max(14, 4 * seasonal_period)))


def _log(low: float, high: float) -> Distribution:
    return Distribution("log_float", low, high)


def _int(low: int, high: int) -> Distribution:
    return Distribution("int", low, high)


def _choice(*choices: Any) -> Distribution:
    return Distribution("categorical", choices=choices)


# Built-in spaces of models on lags; ``lags`` is added from the seasonal period.
LAG_SPACES: Final[dict[str, dict[str, Distribution]]] = {
    "linear": {"differencing": Distribution("categorical", choices=(0, 1))},
    "knn": {"estimator_params.n_neighbors": Distribution("int", 1, 20)},
    "svm": {
        "estimator_params.C": Distribution("log_float", 1e-2, 1e2),
        "estimator_params.epsilon": Distribution("log_float", 1e-3, 1.0),
    },
    "tree": {
        "estimator_params.max_depth": Distribution("int", 2, 12),
        "estimator_params.min_samples_leaf": Distribution("int", 1, 20),
    },
    "random_forest": {
        "estimator_params.n_estimators": Distribution("int", 20, 200),
        "estimator_params.max_depth": Distribution("int", 2, 12),
    },
    "mlp": {
        "estimator_params.alpha": Distribution("log_float", 1e-5, 1e-1),
        "estimator_params.hidden_layer_sizes": Distribution(
            "categorical", choices=("16", "32", "64", "32,16")
        ),
    },
    # Opt-in additions of the ensemble plan (F1).
    "ridge_regression": {"estimator_params.alpha": _log(1e-3, 1e2)},
    "lasso_regression": {"estimator_params.alpha": _log(1e-4, 1.0)},
    "elastic_net_regression": {
        "estimator_params.alpha": _log(1e-4, 1.0),
        "estimator_params.l1_ratio": Distribution("float", 0.1, 0.9),
    },
    "extra_trees_regressor": {
        "estimator_params.n_estimators": _int(20, 200),
        "estimator_params.max_depth": _int(2, 12),
        "estimator_params.min_samples_leaf": _int(1, 20),
    },
    "hist_gradient_boosting": {
        "estimator_params.learning_rate": _log(0.01, 0.3),
        "estimator_params.max_iter": _int(50, 300),
        "estimator_params.max_leaf_nodes": _int(7, 63),
        "estimator_params.min_samples_leaf": _int(5, 50),
    },
    "adaboost_r2": {
        "estimator_params.n_estimators": _int(20, 200),
        "estimator_params.learning_rate": _log(0.01, 1.0),
        "estimator_params.loss": _choice("linear", "square", "exponential"),
    },
    "lightgbm_regressor": {
        "estimator_params.n_estimators": _int(50, 400),
        "estimator_params.learning_rate": _log(0.01, 0.3),
        "estimator_params.num_leaves": _int(7, 63),
        "estimator_params.min_child_samples": _int(5, 40),
    },
    "xgboost_regressor": {
        "estimator_params.n_estimators": _int(50, 400),
        "estimator_params.learning_rate": _log(0.01, 0.3),
        "estimator_params.max_depth": _int(2, 8),
        "estimator_params.subsample": Distribution("float", 0.6, 1.0),
    },
    "catboost_regressor": {
        "estimator_params.iterations": _int(50, 400),
        "estimator_params.learning_rate": _log(0.01, 0.3),
        "estimator_params.depth": _int(2, 8),
    },
}
# Built-in spaces of models without lags (statistical, intermittent demand).
PLAIN_SPACES: Final[dict[str, dict[str, Distribution]]] = {
    "window_average": {"window": _int(3, 60)},
    "seasonal_window_average": {"window": _int(1, 8)},
    "exponential_smoothing": {"trend": _choice(True, False)},
    "harmonic_regression": {
        "ar_lags": _int(0, 7),
        "trend": _choice(True, False),
        "ridge": _log(1e-8, 1e-1),
    },
    "croston_sba": {"alpha": Distribution("float", 0.05, 0.5)},
    "tsb": {
        "alpha_d": Distribution("float", 0.05, 0.5),
        "alpha_p": Distribution("float", 0.05, 0.5),
    },
    "adida": {"alpha": Distribution("float", 0.05, 0.5)},
    "imapa": {"alpha": Distribution("float", 0.05, 0.5)},
}
# Every built-in space. Models missing here choose their own settings (``arima``, the
# ``sf_*`` models) or have nothing to tune (``naive``, ``historic_average`` ...).
TUNING_SPACES: Final[dict[str, dict[str, Distribution]]] = {**LAG_SPACES, **PLAIN_SPACES}


def default_space(name: str, seasonal_period: int) -> dict[str, Distribution]:
    """Built-in space of a base model; raises if the model has nothing to tune."""

    if name not in TUNING_SPACES:
        raise UnsupportedCapabilityError(
            f"{name} não tem espaço de ajuste embutido; passe search_space={{...}} ou escolha "
            f"um de {sorted(TUNING_SPACES)}."
        )
    if name in LAG_SPACES:
        return {"lags": _lags(seasonal_period), **LAG_SPACES[name]}
    return dict(PLAIN_SPACES[name])


@dataclass(frozen=True)
class TuningResult:
    """Every trial of a tuning run, the best candidate and how it was searched."""

    original: CandidateSpec
    best: CandidateSpec
    trials: pd.DataFrame
    sampler: str
    metric: str

    @property
    def improvement(self) -> float:
        """Relative gain of the best over the original on ``V_search`` (0 if unchanged)."""

        scores = self.trials.set_index("candidate_id")["score"]
        before, after = (
            float(scores[self.original.candidate_id]),
            float(scores[self.best.candidate_id]),
        )
        return (before - after) / abs(before) if before not in (0.0, math.inf) else 0.0


class Tuner:
    def __init__(
        self,
        evaluator: TemporalEvaluator,
        *,
        n_trials: int = 20,
        sampler: Sampler = "auto",
        session_seed: int = 0,
    ) -> None:
        if isinstance(n_trials, bool) or not isinstance(n_trials, int) or n_trials < 1:
            raise ConfigurationError("n_trials deve ser inteiro >= 1.")
        if sampler not in ("auto", "optuna", "random"):
            raise ConfigurationError(
                f"sampler deve ser 'auto', 'optuna' ou 'random'; recebido {sampler!r}."
            )
        self.evaluator = evaluator
        self.n_trials = n_trials
        self.sampler = sampler
        self.session_seed = session_seed

    @property
    def metric(self) -> str:
        return self.evaluator.metrics[0]

    def tune(
        self,
        candidate: CandidateSpec,
        plan: TemporalPlan | PanelPlan,
        space: Mapping[str, Distribution],
    ) -> TuningResult:
        """Best of ``n_trials`` configurations of ``candidate.base`` on ``V_search``."""

        ensure_not_sealed(candidate, plan)
        if not space:
            raise ConfigurationError("Espaço de ajuste vazio.")
        seed = derive_seed(self.session_seed, "tune", candidate.candidate_id)
        rows: list[dict[str, Any]] = []
        scored: dict[str, tuple[float, CandidateSpec]] = {}

        def objective(params: dict[str, Any]) -> float:
            spec = _with_base_params(candidate, params)
            if spec.candidate_id not in scored:
                result = self.evaluator.evaluate(spec, plan, "search")
                full = result.coverage(self.metric) >= 1.0
                score = result.mean(self.metric) if full else math.inf
                scored[spec.candidate_id] = (score, spec)
                rows.append(
                    {
                        "trial": len(rows),
                        "candidate_id": spec.candidate_id,
                        "label": spec.label,
                        "score": score,
                        "coverage": result.coverage(self.metric),
                        **params,
                    }
                )
            return scored[spec.candidate_id][0]

        # Trial 0: the configuration as given, with its values of the tuned parameters.
        original = {name: _lookup(candidate.base.params, name) for name in space}
        objective({name: value for name, value in original.items() if value is not None})
        sampler = self._resolve()
        if sampler == "optuna":
            _run_optuna(objective, space, self.n_trials - 1, seed)
        else:
            rng = np.random.default_rng(seed)
            for _ in range(self.n_trials - 1):
                objective({name: dist.sample(rng) for name, dist in space.items()})
        best = min(scored.values(), key=lambda item: (item[0], item[1].candidate_id))[1]
        if not math.isfinite(scored[best.candidate_id][0]):
            best = candidate
        trials = pd.DataFrame(rows).set_index("trial")
        return TuningResult(candidate, best, trials, sampler, self.metric)

    def _resolve(self) -> str:
        installed = importlib.util.find_spec("optuna") is not None
        if self.sampler == "optuna" and not installed:
            raise UnsupportedCapabilityError(
                'O sampler Optuna requer o extra `optuna`: pip install "autohybts[optuna]".'
            )
        if self.sampler == "auto":
            return "optuna" if installed else "random"
        return self.sampler


def _with_base_params(candidate: CandidateSpec, params: Mapping[str, Any]) -> CandidateSpec:
    merged = candidate.base.params
    for dotted, value in params.items():
        target = merged
        *parents, leaf = dotted.split(".")
        for parent in parents:
            child = target.get(parent)
            if child is None:
                child = target[parent] = {}
            if not isinstance(child, dict):
                raise ConfigurationError(f"{parent} não é um dicionário de parâmetros.")
            target = child
        target[leaf] = _decode(leaf, value)
    return replace(candidate, base=ModelSpec.of(candidate.base.name, merged))


def _lookup(params: Mapping[str, Any], dotted: str) -> Any:
    value: Any = params
    for part in dotted.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    if isinstance(value, list) and dotted.endswith("hidden_layer_sizes"):
        return ",".join(str(units) for units in value)
    return value


def _decode(name: str, value: Any) -> Any:
    if name == "hidden_layer_sizes" and isinstance(value, str):
        return [int(units) for units in value.split(",")]
    return value


def _run_optuna(
    objective: Callable[[dict[str, Any]], float],
    space: Mapping[str, Distribution],
    n_trials: int,
    seed: int,
) -> None:
    import warnings

    with warnings.catch_warnings():
        # tqdm warns (with the absolute path of the user's environment) when a notebook has
        # no ipywidgets; it says nothing about the tuning.
        warnings.filterwarnings("ignore", message="IProgress not found")
        import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))

    def wrapped(trial: Any) -> float:
        score = objective({name: dist.suggest(trial, name) for name, dist in space.items()})
        return score if math.isfinite(score) else 1e300

    study.optimize(wrapped, n_trials=n_trials)
