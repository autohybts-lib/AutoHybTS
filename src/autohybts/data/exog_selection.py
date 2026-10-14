"""Automatic choice of exogenous features (ADR-0011).

Every method looks only at the training data it is given; the experiment passes the data
before its first ``V_select`` origin (like tuning on ``V_search``), so the choice never uses
``V_select`` or the sealed test.

Candidate features are each column at the lags the protocol allows: ``future`` columns at
lag 0, ``past`` columns at lags ``step .. max_lag``. Series and covariates are differenced
once when the series is not stationary (KPSS, as ``auto.arima``), and the series' own lags
``1..p`` are controls, so a feature is chosen for what it adds to the series' past.

Methods (each returns at most one lag per column, except ``lasso`` and ``stepwise_bic``):

* ``ccf``: cross-correlation of the prewhitened series and covariate (Box-Jenkins): both
  filtered by an AR model of the covariate; Holm-adjusted p-values of the correlations.
* ``granger``: F test of the covariate's lags over the controls, Holm across columns; a
  selected column keeps its lag with the largest |t|.
* ``lasso``: Lasso over standardized features and controls, with the penalty of the
  one-standard-error rule on a temporal cross-validation (sparser than the CV optimum).
* ``stepwise_bic``: forward selection by BIC of an OLS on the controls, adding a feature
  while BIC falls (BIC, not AIC: with many candidate lags AIC admits noise).
* ``mutual_info``: mutual information of each feature with the series' residual after its
  own lags, kept when above the 95th percentile of the largest score over all features in
  30 permutations of that residual (a family-wise null).
* ``vote``: columns chosen by most of the methods above, at their most chosen lag.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd

from autohybts.domain.specs import ExogFeature
from autohybts.exceptions import ConfigurationError, InsufficientHistoryError

__all__ = ["BASE_METHODS", "SELECTION_METHODS", "ExogSelection", "select_exog", "vote_selection"]

BASE_METHODS: Final = ("ccf", "granger", "lasso", "stepwise_bic", "mutual_info")
SELECTION_METHODS: Final = (*BASE_METHODS, "vote")
_CONTROLS: Final = 7
_ALPHA: Final = 0.05
_PERMUTATIONS: Final = 30

FloatArray = npt.NDArray[np.float64]
_Scores = tuple[FloatArray, FloatArray, npt.NDArray[np.bool_]]


@dataclass(frozen=True, eq=False)
class ExogSelection:
    """Features chosen by one method and the scores of every candidate feature."""

    method: str
    features: tuple[ExogFeature, ...]
    table: pd.DataFrame


def select_exog(
    y: npt.ArrayLike,
    covariates: pd.DataFrame,
    kinds: Mapping[str, str],
    step: int,
    method: str,
    *,
    max_lag: int | None = None,
    seed: int = 0,
) -> ExogSelection:
    """Features of ``covariates`` chosen by ``method`` on the training values ``y``.

    ``covariates`` has one row per value of ``y`` (same dates). ``past`` lags go from
    ``step`` to ``max_lag`` (default ``step + 6``).
    """

    if method not in SELECTION_METHODS:
        raise ConfigurationError(
            f"Método de seleção desconhecido {method!r}; opções: {SELECTION_METHODS}."
        )
    if method == "vote":
        parts = [
            select_exog(y, covariates, kinds, step, m, max_lag=max_lag, seed=seed)
            for m in BASE_METHODS
        ]
        return vote_selection(parts)
    data = _Prepared.build(np.asarray(y, dtype=np.float64), covariates, kinds, step, max_lag)
    scores = _METHODS[method](data, seed)
    table = pd.DataFrame(
        {
            "method": method,
            "column": [f.column for f in data.features],
            "lag": [f.lag for f in data.features],
            "feature": [f.name for f in data.features],
            "score": scores[0],
            "p_value": scores[1],
            "selected": scores[2],
        }
    )
    chosen = tuple(f for f, keep in zip(data.features, scores[2], strict=True) if keep)
    return ExogSelection(method, tuple(sorted(chosen)), table)


@dataclass(frozen=True, eq=False)
class _Prepared:
    target: FloatArray  # differenced series on the aligned rows
    controls: FloatArray  # its lags 1..p on the same rows
    features: tuple[ExogFeature, ...]
    matrix: FloatArray  # one column per feature, aligned with ``target``
    series: FloatArray  # whole differenced series (prewhitening)
    columns: Mapping[str, FloatArray]  # whole differenced covariates

    @classmethod
    def build(
        cls,
        y: FloatArray,
        covariates: pd.DataFrame,
        kinds: Mapping[str, str],
        step: int,
        max_lag: int | None,
    ) -> _Prepared:
        if covariates.shape[0] != y.size:
            raise ConfigurationError("As exógenas devem ter uma linha por valor do treino.")
        top = step + 6 if max_lag is None else max_lag
        if top < step:
            raise ConfigurationError(f"max_lag deve ser >= {step}; recebido {top}.")
        d = _differences(y)
        series = np.diff(y, n=d) if d else y.copy()
        columns = {
            str(c): (
                np.diff(covariates[c].to_numpy(np.float64), n=d)
                if d
                else covariates[c].to_numpy(np.float64)
            )
            for c in covariates.columns
        }
        features = tuple(
            ExogFeature(c, lag)
            for c in columns
            for lag in ((0,) if kinds.get(c) == "future" else range(step, top + 1))
        )
        first = max(_CONTROLS, max(f.lag for f in features))
        rows = series.size - first
        if rows < 20 + len(features):
            raise InsufficientHistoryError(
                f"Treino curto demais para escolher exógenas: {rows} linhas para "
                f"{len(features)} features."
            )
        target = series[first:]
        controls = np.column_stack(
            [series[first - k : series.size - k] for k in range(1, _CONTROLS + 1)]
        )
        matrix = np.column_stack(
            [columns[f.column][first - f.lag : series.size - f.lag] for f in features]
        )
        return cls(target, controls, features, matrix, series, columns)


def _differences(y: FloatArray) -> int:
    """One difference when KPSS rejects level stationarity (statsmodels), else none."""

    from autohybts.data.stationarity import choose_d
    from autohybts.exceptions import UnsupportedCapabilityError

    try:
        return choose_d(y, max_d=1)
    except UnsupportedCapabilityError:  # without the stats extra: a lag-1 autocorrelation proxy
        centered = y - y.mean()
        rho = float(centered[1:] @ centered[:-1]) / max(float(centered @ centered), 1e-12)
        return int(rho > 0.9)


def _ols(design: FloatArray, target: FloatArray) -> tuple[FloatArray, FloatArray, float]:
    """Coefficients, standard errors and residual sum of squares of ``target ~ [1, design]``."""

    full = np.column_stack([np.ones(target.size), design])
    coefficients, *_ = np.linalg.lstsq(full, target, rcond=None)
    residual = target - full @ coefficients
    rss = float(residual @ residual)
    dof = max(int(target.size) - int(full.shape[1]), 1)
    covariance = np.linalg.pinv(full.T @ full) * rss / dof
    return coefficients[1:], np.sqrt(np.clip(np.diag(covariance)[1:], 0, None)), rss


def _holm(p_values: FloatArray) -> FloatArray:
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 0.0
    m = p_values.size
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def _normal_p(z: FloatArray) -> FloatArray:
    return np.array([math.erfc(abs(v) / math.sqrt(2)) for v in z])


def _one_per_column(
    data: _Prepared, keep: npt.NDArray[np.bool_], score: FloatArray
) -> npt.NDArray[np.bool_]:
    """Of the selected features of a column, keep the one with the largest score."""

    out = np.zeros(len(data.features), dtype=bool)
    best: dict[str, int] = {}
    for index, feature in enumerate(data.features):
        if keep[index] and (
            feature.column not in best or score[index] > score[best[feature.column]]
        ):
            best[feature.column] = index
    out[list(best.values())] = True
    return out


def _ccf(data: _Prepared, seed: int) -> _Scores:
    del seed
    scores = np.zeros(len(data.features))
    for column, values in data.columns.items():
        coefficients = _ar_filter(values)
        a = _apply_filter(values, coefficients)
        b = _apply_filter(data.series, coefficients)
        for index, feature in enumerate(data.features):
            if feature.column != column:
                continue
            k = feature.lag
            x, z = a[: a.size - k] if k else a, b[k:]
            scores[index] = abs(float(np.corrcoef(x, z)[0, 1])) if x.size > 2 else 0.0
    n = data.target.size
    p_values = _holm(_normal_p(scores * math.sqrt(n)))
    keep = p_values < _ALPHA
    return scores, p_values, _one_per_column(data, keep, scores)


def _ar_filter(values: FloatArray, max_order: int = 5) -> FloatArray:
    """AR coefficients of ``values`` by OLS, order chosen by AIC (0 means no filter)."""

    best: tuple[float, FloatArray] = (math.inf, np.zeros(0))
    for p in range(max_order + 1):
        if p == 0:
            residual = values - values.mean()
            coefficients = np.zeros(0)
        else:
            design = np.column_stack([values[p - k : values.size - k] for k in range(1, p + 1)])
            coefficients, _, _ = _ols(design, values[p:])
            residual = values[p:] - design @ coefficients
            residual = residual - residual.mean()
        n = residual.size
        aic = n * math.log(max(float(residual @ residual) / n, 1e-12)) + 2 * (p + 1)
        if aic < best[0]:
            best = (aic, coefficients)
    return best[1]


def _apply_filter(values: FloatArray, coefficients: FloatArray) -> FloatArray:
    p = coefficients.size
    if p == 0:
        return values - values.mean()
    filtered = values[p:].copy()
    for k in range(1, p + 1):
        filtered -= coefficients[k - 1] * values[p - k : values.size - k]
    return filtered - filtered.mean()


def _granger(data: _Prepared, seed: int) -> _Scores:
    from scipy import stats

    del seed
    _, _, restricted = _ols(data.controls, data.target)
    column_p: dict[str, float] = {}
    t_scores = np.zeros(len(data.features))
    for column in data.columns:
        indexes = [i for i, f in enumerate(data.features) if f.column == column]
        design = np.column_stack([data.controls, data.matrix[:, indexes]])
        coefficients, errors, rss = _ols(design, data.target)
        q = len(indexes)
        dof = data.target.size - design.shape[1] - 1
        statistic = ((restricted - rss) / q) / max(rss / max(dof, 1), 1e-12)
        column_p[column] = float(stats.f.sf(statistic, q, max(dof, 1)))
        tail = slice(data.controls.shape[1], None)
        t_scores[indexes] = np.abs(
            coefficients[tail] / np.where(errors[tail] > 0, errors[tail], np.inf)
        )
    names = list(column_p)
    adjusted = dict(zip(names, _holm(np.array([column_p[c] for c in names])), strict=True))
    p_values = np.array([adjusted[f.column] for f in data.features])
    return t_scores, p_values, _one_per_column(data, p_values < _ALPHA, t_scores)


def _lasso(data: _Prepared, seed: int) -> _Scores:
    from sklearn.linear_model import Lasso, LassoCV
    from sklearn.model_selection import TimeSeriesSplit

    design = np.column_stack([data.controls, data.matrix])
    center, spread = design.mean(axis=0), design.std(axis=0)
    spread = np.where(spread > 0, spread, 1.0)
    scaled = (design - center) / spread
    search = LassoCV(cv=TimeSeriesSplit(n_splits=5), random_state=seed, max_iter=20000)
    search.fit(scaled, data.target)
    # One-standard-error rule: the sparsest penalty whose CV error is within one standard
    # error of the best; the CV optimum itself keeps too many features.
    errors = np.asarray(search.mse_path_)
    mean = errors.mean(axis=1)
    se = errors.std(axis=1) / math.sqrt(errors.shape[1])
    best = int(np.argmin(mean))
    alpha = float(np.max(np.asarray(search.alphas_)[mean <= mean[best] + se[best]]))
    model = Lasso(alpha=alpha, max_iter=20000).fit(scaled, data.target)
    coefficients = np.abs(model.coef_[data.controls.shape[1] :])
    return coefficients, np.full(coefficients.size, np.nan), coefficients > 1e-8


def _stepwise_bic(data: _Prepared, seed: int) -> _Scores:
    del seed
    n = data.target.size

    def bic(indexes: list[int]) -> float:
        design = (
            np.column_stack([data.controls, data.matrix[:, indexes]]) if indexes else data.controls
        )
        _, _, rss = _ols(design, data.target)
        return float(n * math.log(max(rss / n, 1e-12)) + math.log(n) * (design.shape[1] + 1))

    chosen: list[int] = []
    current = bic(chosen)
    gains = np.zeros(len(data.features))
    while True:
        options = [i for i in range(len(data.features)) if i not in chosen]
        if not options:
            break
        scored = [(bic([*chosen, i]), i) for i in options]
        value, best = min(scored)
        if current - value <= 0.0:
            break
        gains[best] = current - value
        chosen.append(best)
        current = value
    keep = np.zeros(len(data.features), dtype=bool)
    keep[chosen] = True
    return gains, np.full(len(data.features), np.nan), keep


def _mutual_info(data: _Prepared, seed: int) -> _Scores:
    from sklearn.feature_selection import mutual_info_regression

    # Information beyond the series' own past: the target is the residual of the controls.
    design = np.column_stack([np.ones(data.target.size), data.controls])
    coefficients, *_ = np.linalg.lstsq(design, data.target, rcond=None)
    residual = data.target - design @ coefficients
    scores = np.asarray(mutual_info_regression(data.matrix, residual, random_state=seed))
    # Family-wise null: permuting the residual breaks every feature at once; the 95th
    # percentile of the largest score over all features controls false picks across lags.
    rng = np.random.default_rng(seed)
    null = np.array(
        [
            float(
                np.max(
                    mutual_info_regression(
                        data.matrix, rng.permutation(residual), random_state=seed
                    )
                )
            )
            for _ in range(_PERMUTATIONS)
        ]
    )
    p_values = np.array(
        [(1 + int(np.sum(null >= value))) / (1 + _PERMUTATIONS) for value in scores]
    )
    keep = p_values < _ALPHA
    return scores, p_values, _one_per_column(data, keep, scores)


_METHODS: Final = {
    "ccf": _ccf,
    "granger": _granger,
    "lasso": _lasso,
    "stepwise_bic": _stepwise_bic,
    "mutual_info": _mutual_info,
}


def vote_selection(parts: Sequence[ExogSelection]) -> ExogSelection:
    """Columns chosen by most methods, each at the lag most methods chose (ties: smallest)."""

    votes: Counter[str] = Counter()
    lags: dict[str, Counter[int]] = {}
    for part in parts:
        columns = {f.column for f in part.features}
        votes.update(columns)
        for feature in part.features:
            lags.setdefault(feature.column, Counter())[feature.lag] += 1
    majority = len(parts) // 2 + 1
    chosen = tuple(
        sorted(
            ExogFeature(column, min(lags[column], key=lambda lag: (-lags[column][lag], lag)))
            for column, count in votes.items()
            if count >= majority
        )
    )
    table = pd.DataFrame(
        {
            "method": "vote",
            "column": sorted(votes),
            "lag": [min(lags[c], key=lambda lag: (-lags[c][lag], lag)) for c in sorted(votes)],
            "score": [votes[c] for c in sorted(votes)],
            "p_value": np.nan,
            "selected": [votes[c] >= majority for c in sorted(votes)],
        }
    )
    table["feature"] = [
        ExogFeature(c, int(lag)).name for c, lag in zip(table.column, table.lag, strict=True)
    ]
    return ExogSelection("vote", chosen, table)
