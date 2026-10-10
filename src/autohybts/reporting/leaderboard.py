"""Validation leaderboard (architecture §17.1).

This is a descriptive table, not the selection rule: the incremental selector decides
the champion with paired comparisons. ``mean_rank`` follows ranking convention v1 (one
series: the rank of the score, ties averaged); ``is_champion`` marks the selected model.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from autohybts.domain.results import EvaluationResult
from autohybts.exceptions import ConfigurationError

__all__ = ["leaderboard"]


def leaderboard(
    results: Sequence[EvaluationResult],
    sort: str | None = None,
    *,
    champion_id: str | None = None,
) -> pd.DataFrame:
    """One row per candidate; sorted by ``sort`` (default: first metric), then candidate id.

    Metric columns are means over valid units. ``coverage`` is computed for the sort
    metric, so a candidate that scored fewer units is visible as such rather than ranked
    on a partial, possibly easier, subset.
    """

    if not results:
        raise ConfigurationError("Nenhum resultado para o leaderboard.")
    metrics = results[0].metric_names
    if any(result.metric_names != metrics for result in results):
        raise ConfigurationError("Resultados com métricas diferentes não são comparáveis.")
    if len({result.plan_fingerprint for result in results}) != 1:
        raise ConfigurationError("Resultados de planos temporais diferentes não são comparáveis.")
    sort = sort or metrics[0]
    if sort not in metrics:
        raise ConfigurationError(f"Métrica de ordenação {sort!r} não avaliada; opções: {metrics}.")

    rows = []
    for result in results:
        candidate = result.candidate
        trials = result.trials
        rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "name": candidate.name,
                "model": candidate.label,
                "family": candidate.family,
                "strategy": candidate.strategy,
                "base": candidate.base.name,
                "corrector": None if candidate.corrector is None else candidate.corrector.name,
                "residual_source": candidate.residual_source,
                "window": candidate.window_size,
                "alpha": candidate.alpha,
                **{metric: result.mean(metric) for metric in metrics},
                "coverage": result.coverage(sort),
                "n_units": result.n_expected,
                "n_failed": sum(trial.status == "FAILED" for trial in trials),
                "n_skipped": sum(trial.status == "SKIPPED" for trial in trials),
                "fit_seconds": float(np.sum([trial.fit_seconds for trial in trials])),
                "predict_seconds": float(np.sum([trial.predict_seconds for trial in trials])),
            }
        )
    table = pd.DataFrame(rows)
    ranks = table[sort].rank(method="average", na_option="keep")
    table.insert(list(table.columns).index(sort), "mean_rank", ranks)
    table["is_champion"] = table["candidate_id"] == champion_id
    # NaN (no valid unit) sorts last; candidate_id makes ties deterministic.
    table = table.sort_values([sort, "candidate_id"], na_position="last", kind="mergesort")
    return table.set_index("candidate_id")
