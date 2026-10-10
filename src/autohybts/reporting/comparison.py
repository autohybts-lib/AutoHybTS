"""Comparison of several models on the same evaluation units (§17, §30.2).

Tables and tests here are for *reporting*. They never guide the search or the selection,
whose rule is the protocol's (§9.3). Units are forecast origins of one series; when
origins overlap (step < horizon) their errors are dependent, which the Diebold-Mariano
variance accounts for with Newey-West lags and the other tests do not. Small samples give
coarse p-values; with few units "no significant difference" is not evidence of equivalence.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from autohybts.data.schema import SeriesFrame
from autohybts.domain.results import EvaluationResult
from autohybts.exceptions import ConfigurationError

__all__ = [
    "ComparisonResult",
    "diebold_mariano",
    "friedman_nemenyi",
    "holm",
    "paired_tests",
    "summary_table",
    "unit_table",
    "win_rate_matrix",
]


def _labels(results: Sequence[EvaluationResult]) -> list[str]:
    """Short name (``arima-mlp``) when unique, else the full label, else label plus id."""

    names = [r.candidate.name for r in results]
    labels = [r.candidate.label for r in results]
    out = []
    for name, label, r in zip(names, labels, results, strict=True):
        if names.count(name) == 1:
            out.append(name)
        elif labels.count(label) == 1:
            out.append(label)
        else:
            out.append(f"{label} #{r.candidate_id[:6]}")
    return out


def _check(results: Sequence[EvaluationResult]) -> None:
    if len(results) < 1:
        raise ConfigurationError("Nenhum modelo para comparar.")
    units = [r.unit_ids for r in results]
    if len(set(units)) != 1 or len({r.plan_fingerprint for r in results}) != 1:
        raise ConfigurationError("Modelos avaliados em unidades diferentes não são comparáveis.")


def unit_table(results: Sequence[EvaluationResult], metric: str) -> pd.DataFrame:
    """Units (rows) × models (columns); invalid units are NaN, never zero."""

    _check(results)
    return pd.DataFrame(
        {label: r.scores(metric) for label, r in zip(_labels(results), results, strict=True)}
    )


def summary_table(results: Sequence[EvaluationResult], metric: str) -> pd.DataFrame:
    """Mean, std, median, valid units and mean rank per model, best mean rank first."""

    table = unit_table(results, metric)
    complete = table.dropna()
    ranks = complete.rank(axis=1, method="average").mean() if len(complete) else np.nan
    summary = pd.DataFrame(
        {
            "mean": table.mean(),
            "std": table.std(ddof=1),
            "median": table.median(),
            "n_valid": table.notna().sum(),
            "n_units": len(table),
            "mean_rank": ranks,
        }
    )
    summary.index.name = "model"
    return summary.sort_values(["mean_rank", "mean"], na_position="last")


def win_rate_matrix(results: Sequence[EvaluationResult], metric: str) -> pd.DataFrame:
    """Share of common valid units where the row model beats (strictly) the column model."""

    table = unit_table(results, metric)
    matrix = pd.DataFrame(np.nan, index=table.columns, columns=table.columns)
    for row in table.columns:
        for column in table.columns:
            if row == column:
                continue
            both = table[[row, column]].dropna()
            if len(both):
                matrix.loc[row, column] = float((both[row] < both[column]).mean())
    return matrix


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (NaN kept as NaN)."""

    indexed = [(p, i) for i, p in enumerate(p_values) if math.isfinite(p)]
    adjusted = [math.nan] * len(p_values)
    running = 0.0
    m = len(indexed)
    for rank, (p, i) in enumerate(sorted(indexed)):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[i] = running
    return adjusted


def diebold_mariano(
    loss_a: np.ndarray, loss_b: np.ndarray, max_lag: int = 0
) -> tuple[float, float]:
    """DM statistic and two-sided p-value for equal expected loss (t with n-1 df).

    The long-run variance of ``d = loss_a - loss_b`` uses Newey-West weights up to
    ``max_lag``, to account for dependence between overlapping forecast units.
    """

    from scipy import stats

    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    n = d.size
    if n < 3:
        return math.nan, math.nan
    centered = d - d.mean()
    variance = float(np.dot(centered, centered)) / n
    for lag in range(1, min(max_lag, n - 1) + 1):
        weight = 1 - lag / (max_lag + 1)
        variance += 2 * weight * float(np.dot(centered[lag:], centered[:-lag])) / n
    if variance <= 0:
        return math.nan, math.nan
    statistic = float(d.mean() / math.sqrt(variance / n))
    p_value = float(2 * stats.t.sf(abs(statistic), df=n - 1))
    return statistic, p_value


def paired_tests(
    results: Sequence[EvaluationResult], metric: str, reference: int = 0, max_lag: int = 0
) -> pd.DataFrame:
    """Each model against ``results[reference]``: Wilcoxon and Diebold-Mariano, Holm-adjusted."""

    from scipy import stats

    table = unit_table(results, metric)
    labels = list(table.columns)
    ref = labels[reference]
    rows = []
    for label in labels:
        if label == ref:
            continue
        both = table[[ref, label]].dropna()
        diff = both[ref] - both[label]  # positive: the model beats the reference
        if len(both) == 0 or np.allclose(diff, 0):
            wilcoxon_p = 1.0 if len(both) else math.nan
        else:
            wilcoxon_p = float(stats.wilcoxon(both[ref], both[label]).pvalue)
        dm, dm_p = diebold_mariano(both[ref].to_numpy(), both[label].to_numpy(), max_lag)
        rows.append(
            {
                "model": label,
                "reference": ref,
                "n_units": len(both),
                "mean_difference": float(diff.mean()) if len(both) else math.nan,
                "median_difference": float(diff.median()) if len(both) else math.nan,
                "wins": int((diff > 0).sum()),
                "wilcoxon_p": wilcoxon_p,
                "dm_statistic": dm,
                "dm_p": dm_p,
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["wilcoxon_p_holm"] = holm(frame["wilcoxon_p"].tolist())
        frame["dm_p_holm"] = holm(frame["dm_p"].tolist())
    return frame


def friedman_nemenyi(
    results: Sequence[EvaluationResult], metric: str, alpha: float = 0.05
) -> dict[str, Any]:
    """Friedman test over complete units and Nemenyi critical difference of mean ranks."""

    from scipy import stats

    table = unit_table(results, metric).dropna()
    k, n = table.shape[1], table.shape[0]
    ranks = table.rank(axis=1, method="average").mean() if n else pd.Series(dtype=float)
    out: dict[str, Any] = {"n_units": n, "n_models": k, "mean_ranks": ranks, "alpha": alpha}
    if k < 3 or n < 2:
        out.update(statistic=math.nan, p_value=math.nan, critical_difference=math.nan)
        return out
    statistic, p_value = stats.friedmanchisquare(*(table[c] for c in table.columns))
    q = float(stats.studentized_range.ppf(1 - alpha, k, np.inf)) / math.sqrt(2)
    out.update(
        statistic=float(statistic),
        p_value=float(p_value),
        critical_difference=q * math.sqrt(k * (k + 1) / (6 * n)),
    )
    return out


@dataclass(frozen=True)
class ComparisonResult:
    """Several models scored on the same units, with tables, tests and plots."""

    results: tuple[EvaluationResult, ...]
    metric: str
    history: SeriesFrame
    source: str
    reference: int = 0
    max_lag: int = 0
    splits: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        _check(self.results)
        if self.metric not in self.results[0].metric_names:
            raise ConfigurationError(f"Métrica {self.metric!r} não foi avaliada.")

    @property
    def labels(self) -> list[str]:
        return _labels(self.results)

    @property
    def units(self) -> pd.DataFrame:
        return unit_table(self.results, self.metric)

    @property
    def summary(self) -> pd.DataFrame:
        return summary_table(self.results, self.metric)

    @property
    def win_rates(self) -> pd.DataFrame:
        return win_rate_matrix(self.results, self.metric)

    def tests(self, reference: int | None = None) -> pd.DataFrame:
        return paired_tests(
            self.results,
            self.metric,
            self.reference if reference is None else reference,
            self.max_lag,
        )

    def friedman(self, alpha: float = 0.05) -> dict[str, Any]:
        return friedman_nemenyi(self.results, self.metric, alpha)

    def plot(self, kind: str = "leaderboard") -> Any:
        from autohybts.reporting.plots import comparison_plot

        return comparison_plot(kind, self)

    def __repr__(self) -> str:
        return (
            f"ComparisonResult({len(self.results)} modelos, {len(self.units)} unidades, "
            f"{self.metric}, fonte {self.source!r})"
        )
