"""scikit-learn regressors on causal lag windows, forecasting multi-step recursively."""

from __future__ import annotations

import copy
import importlib
import warnings
from collections.abc import Mapping
from contextlib import AbstractContextManager, nullcontext
from typing import Any, Final

import numpy as np

from autohybts.adapters.base import FloatArray, ForecasterBase
from autohybts.data.preprocessing import Differencer
from autohybts.data.stationarity import check_differencing, differencer_for, max_lost
from autohybts.data.windows import lag_matrix, latest_lags, normalize_lags
from autohybts.exceptions import ConfigurationError, UnsupportedCapabilityError

__all__ = [
    "ESTIMATORS",
    "LagRegressionForecaster",
    "build_estimator",
    "check_scale",
    "column_scale",
    "fit_estimator",
    "scalar_scale",
    "scale_values",
    "single_threaded",
    "unscale_values",
]

# Scalings fitted per lag column and on the target separately.
PER_COLUMN: Final = ("columns", "minmax")
# name -> (module, class); imported only when a model is fitted.
ESTIMATORS: Final[dict[str, tuple[str, str]]] = {
    "linear": ("sklearn.linear_model", "LinearRegression"),
    "knn": ("sklearn.neighbors", "KNeighborsRegressor"),
    "svm": ("sklearn.svm", "SVR"),
    "tree": ("sklearn.tree", "DecisionTreeRegressor"),
    "random_forest": ("sklearn.ensemble", "RandomForestRegressor"),
    "mlp": ("sklearn.neural_network", "MLPRegressor"),
    # Regularized linear models and extra trees (plano de ensembles L07).
    "ridge_regression": ("sklearn.linear_model", "Ridge"),
    "lasso_regression": ("sklearn.linear_model", "Lasso"),
    "elastic_net_regression": ("sklearn.linear_model", "ElasticNet"),
    "extra_trees_regressor": ("sklearn.ensemble", "ExtraTreesRegressor"),
    # Boosting on lags (plano de ensembles F1c): B1 gradient boosting and B2 AdaBoost.R2
    # (scikit-learn's AdaBoostRegressor is Drucker's AdaBoost.R2).
    "hist_gradient_boosting": ("sklearn.ensemble", "HistGradientBoostingRegressor"),
    "adaboost_r2": ("sklearn.ensemble", "AdaBoostRegressor"),
    "lightgbm_regressor": ("lightgbm", "LGBMRegressor"),
    "xgboost_regressor": ("xgboost", "XGBRegressor"),
    "catboost_regressor": ("catboost", "CatBoostRegressor"),
}
DEFAULT_ESTIMATOR_PARAMS: Final[dict[str, dict[str, Any]]] = {
    "mlp": {"hidden_layer_sizes": (32,), "max_iter": 1000},
    # Small data per origin: fewer leaves and quiet logs; regularize through these.
    # n_jobs=1: each model sees a few hundred rows, where OpenMP threads cost far more than
    # they save (seconds per fit on a 20-core machine); the experiment parallelizes across
    # candidates instead (n_jobs of the experiment).
    "lightgbm_regressor": {
        "n_estimators": 200,
        "num_leaves": 15,
        "min_child_samples": 10,
        "verbose": -1,
        "n_jobs": 1,
    },
    "xgboost_regressor": {
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.1,
        "n_jobs": 1,
        "verbosity": 0,
    },
    # allow_writing_files=False: otherwise every fit writes a catboost_info/ directory.
    "catboost_regressor": {
        "iterations": 200,
        "depth": 4,
        "thread_count": 1,
        "verbose": 0,
        "allow_writing_files": False,
    },
}
# Estimators whose OpenMP threads cannot be set by a parameter: limited while they run.
OPENMP_ESTIMATORS: Final = frozenset({"hist_gradient_boosting"})
_CONTROLLER: Any = None
SEEDED: Final = frozenset(
    {
        "tree",
        "random_forest",
        "extra_trees_regressor",
        "mlp",
        "hist_gradient_boosting",
        "adaboost_r2",
        "lightgbm_regressor",
        "xgboost_regressor",
        "catboost_regressor",
    }
)
# Estimators from an optional extra: name -> (extra, pip distribution).
EXTRA_ESTIMATORS: Final[dict[str, tuple[str, str]]] = {
    "lightgbm_regressor": ("boosting", "lightgbm"),
    "xgboost_regressor": ("boosting", "xgboost"),
    "catboost_regressor": ("boosting", "catboost"),
}


class LagRegressionForecaster(ForecasterBase):
    """Regressor trained on ``[y[t - lag] for lag in lags] -> y[t]``.

    Optional differencing and standardization are fitted on the training vector only;
    ``differencing="auto"`` chooses ``d`` (by ``differencing_test``) and, with
    ``season_length >= 3``, one seasonal difference on that vector
    (:mod:`autohybts.data.stationarity`), then keeps the choice frozen.
    At prediction time the differencing anchors come from the context (observed values),
    while the learned scale and estimator stay frozen. Multi-step forecasts are recursive:
    each prediction becomes a lag for the next step.

    ``scale=True`` standardizes the series with one mean and spread; ``scale="columns"``
    standardizes each lag column and the target separately (scikit-learn's
    ``StandardScaler`` on X and y, as in the replication project) and ``"minmax"`` maps
    each of them to [0, 1] (``MinMaxScaler``). Per-column scaling changes distance-,
    kernel- and gradient-based estimators (KNN, SVR, MLP) but not linear models or trees.
    """

    def __init__(
        self,
        estimator: str = "linear",
        lags: int | tuple[int, ...] = 7,
        differencing: int | str = 0,
        scale: bool | str = True,
        estimator_params: Mapping[str, Any] | None = None,
        random_state: int | None = 0,
        seasonal_differencing: int = 0,
        season_length: int = 1,
        differencing_test: str = "kpss",
    ) -> None:
        super().__init__()
        if estimator not in ESTIMATORS:
            raise ConfigurationError(
                f"Estimador desconhecido {estimator!r}; opções: {sorted(ESTIMATORS)}."
            )
        check_differencing(differencing, seasonal_differencing, season_length, differencing_test)
        self._lags = normalize_lags(lags)
        self.estimator = estimator
        self.lags = lags
        self.differencing = differencing
        self.scale = check_scale(scale)
        self.estimator_params = estimator_params
        self.random_state = random_state
        self.seasonal_differencing = seasonal_differencing
        self.season_length = season_length
        self.differencing_test = differencing_test

    @property
    def min_history(self) -> int:
        differencer: Differencer | None = getattr(self, "differencer_", None)
        if differencer is not None:
            return max(self._lags) + differencer.lost
        return max(self._lags) + max_lost(
            self.differencing, self.seasonal_differencing, self.season_length
        )

    @property
    def min_train_size(self) -> int:
        return self.min_history + 1

    def _fit(self, y: FloatArray) -> None:
        self.differencer_ = differencer_for(
            y,
            self.differencing,
            self.seasonal_differencing,
            self.season_length,
            self.differencing_test,
        )
        level = self._differenced(y)
        self.model_ = build_estimator(self.estimator, self.estimator_params, self.random_state)
        if self.scale in PER_COLUMN:
            features, targets = lag_matrix(level, self._lags)
            self.x_center_, self.x_spread_ = column_scale(features, self.scale)
            self.center_, self.spread_ = scalar_scale(targets, self.scale)
            with single_threaded(self.estimator):
                self.fit_warnings_ = fit_estimator(
                    self.model_,
                    scale_values(features, self.x_center_, self.x_spread_, self.scale),
                    scale_values(targets, self.center_, self.spread_, self.scale),
                )
            return
        self.center_, self.spread_ = scalar_scale(level, self.scale)
        features, targets = lag_matrix(self._standardized(level), self._lags)
        with single_threaded(self.estimator):
            self.fit_warnings_ = fit_estimator(self.model_, features, targets)

    def _predict(self, horizon: int, history: FloatArray) -> FloatArray:
        anchors = None
        level = history
        if self.differencer_.lost:
            anchors, level = self.differencer_.fit_transform(history)
        columns = self.scale in PER_COLUMN
        # Per-column scaling works on raw lags; the global one on the standardized series.
        window = list(level if columns else self._standardized(level))
        steps = []
        with single_threaded(self.estimator):
            for _ in range(horizon):
                row = latest_lags(np.asarray(window), self._lags)
                if columns:
                    row = scale_values(row, self.x_center_, self.x_spread_, self.scale)
                step = float(self.model_.predict(row.reshape(1, -1))[0])
                if columns:
                    step = float(unscale_values(step, self.center_, self.spread_, self.scale))
                window.append(step)
                steps.append(step)
        forecast = np.asarray(steps)
        if not columns:
            forecast = forecast * self.spread_ + self.center_
        return forecast if anchors is None else anchors.inverse_forecast(forecast)

    def _differenced(self, y: FloatArray) -> FloatArray:
        if not self.differencer_.lost:
            return y
        return self.differencer_.fit_transform(y)[1]

    def _standardized(self, values: FloatArray) -> FloatArray:
        return (values - self.center_) / self.spread_


def check_scale(scale: object) -> bool | str:
    """``True``, ``False``, ``"columns"`` or ``"minmax"`` (both per column)."""

    if isinstance(scale, bool):
        return scale
    if scale in PER_COLUMN:
        return str(scale)
    raise ConfigurationError(
        f"scale deve ser True, False, 'columns' ou 'minmax'; recebido {scale!r}."
    )


def column_scale(
    features: FloatArray, scale: bool | str = "columns"
) -> tuple[FloatArray, FloatArray]:
    """Center and spread of each column: mean and population std, or min and range
    (``"minmax"``); a constant column keeps spread 1."""

    if scale == "minmax":
        center = features.min(axis=0)
        spread = features.max(axis=0) - center
    else:
        center = features.mean(axis=0)
        spread = features.std(axis=0)
    return center, np.where(spread > 0, spread, 1.0)


def scale_values(values: Any, center: Any, spread: Any, scale: bool | str) -> Any:
    """Apply a fitted scaling. ``"minmax"`` uses scikit-learn's arithmetic
    (``x * (1 / range) - min / range``) so results match ``MinMaxScaler`` to the last bit;
    the others compute ``(x - center) / spread``."""

    if scale == "minmax":
        factor = 1.0 / spread
        return values * factor + (0.0 - center * factor)
    return (values - center) / spread


def unscale_values(values: Any, center: Any, spread: Any, scale: bool | str) -> Any:
    """Inverse of :func:`scale_values` (``MinMaxScaler.inverse_transform`` for ``"minmax"``)."""

    if scale == "minmax":
        factor = 1.0 / spread
        return (values - (0.0 - center * factor)) / factor
    return values * spread + center


def scalar_scale(values: FloatArray, scale: bool | str) -> tuple[float, float]:
    """One center and spread for all values: ``0, 1`` when not scaling, min and range for
    ``"minmax"``, mean and std otherwise."""

    if not scale:
        return 0.0, 1.0
    if scale == "minmax":
        low = float(np.min(values))
        spread = float(np.max(values)) - low
        return low, spread if spread > 0 else 1.0
    spread = float(np.std(values))
    return float(np.mean(values)), spread if spread > 0 else 1.0


def build_estimator(
    name: str, estimator_params: Mapping[str, Any] | None, random_state: int | None
) -> Any:
    """Unfitted scikit-learn estimator; seeded estimators get ``random_state`` unless overridden."""

    if name not in ESTIMATORS:
        raise ConfigurationError(f"Estimador desconhecido {name!r}; opções: {sorted(ESTIMATORS)}.")
    module_name, class_name = ESTIMATORS[name]
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        extra, distribution = EXTRA_ESTIMATORS.get(name, ("", module_name))
        hint = f'pip install "autohybts[{extra}]"' if extra else f"pip install {distribution}"
        raise UnsupportedCapabilityError(
            f"O estimador {name!r} requer {distribution}, que não está instalado: {hint}."
        ) from error
    cls = getattr(module, class_name)
    params = {
        **copy.deepcopy(DEFAULT_ESTIMATOR_PARAMS.get(name, {})),
        **dict(estimator_params or {}),
    }
    if name in SEEDED:
        params.setdefault("random_state", random_state)
    try:
        return cls(**params)
    except TypeError as error:
        raise ConfigurationError(f"Parâmetro inválido para {name}: {error}") from error


def single_threaded(name: str) -> AbstractContextManager[Any]:
    """One OpenMP thread while an estimator of ``OPENMP_ESTIMATORS`` fits or predicts."""

    if name not in OPENMP_ESTIMATORS:
        return nullcontext()
    global _CONTROLLER
    if _CONTROLLER is None:
        from threadpoolctl import ThreadpoolController

        _CONTROLLER = ThreadpoolController()
    return _CONTROLLER.limit(limits=1, user_api="openmp")  # type: ignore[no-any-return]


def fit_estimator(model: Any, features: FloatArray, targets: FloatArray) -> list[str]:
    """Fit ``model`` and return its warnings (e.g. MLP non-convergence) instead of emitting them.

    The warnings are kept on the fitted object for auditing; printing one per trial and
    origin would flood the notebook without changing the outcome.
    """

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(features, targets)
    return [f"{w.category.__name__}: {w.message}" for w in caught]
