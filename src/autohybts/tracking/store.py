"""SQLite store of experiments, evaluations and finalized models (ADR-0010).

``ExperimentStore`` writes what an experiment has already computed; it never evaluates a
candidate, never reads the sealed test and is never read by the search or the selection.
Series values are not written unless ``values=True`` (predictions and observations of the
external evaluation). Metrics are stored in long format (``metric``, ``value``) and an
undefined metric is ``NULL``, never 0.
"""

from __future__ import annotations

import enum
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import pandas as pd

import autohybts
from autohybts.domain.identity import stable_hash
from autohybts.exceptions import ArtifactCompatibilityError, ConfigurationError

if TYPE_CHECKING:
    import sqlite3

    from autohybts.experiment import AutoHybridExperiment
    from autohybts.forecasting.finalized import FinalizedForecaster, PanelForecaster

__all__ = ["STORE_SCHEMA", "ExperimentStore"]

STORE_SCHEMA: Final = "autohybts-store/1"

# table -> (columns, primary key). Every table except ``meta`` is keyed by ``run_id``.
_SCHEMA: Final = {
    "runs": (
        "run_id name logged_utc autohybts_version state variant fh profile budget "
        "session_id n_series n_points n_origins seasonal_period ranking_metric "
        "protocol_hash plan_fingerprint data_fingerprint champion_id champion_label "
        "params tags snapshot",
        "run_id",
    ),
    "series": (
        "run_id series_id frequency start end n_observations fingerprint",
        "run_id series_id",
    ),
    "splits": (
        "run_id series_id fold_id layer train_start origin first_target last_target "
        "n_train horizon",
        "run_id series_id fold_id",
    ),
    "candidates": (
        "run_id candidate_id label name family strategy variant base corrector "
        "residual_source window alpha spec",
        "run_id candidate_id",
    ),
    "evaluations": (
        "run_id candidate_id layer mean_rank coverage n_units n_failed n_skipped "
        "fit_seconds predict_seconds is_champion",
        "run_id candidate_id layer",
    ),
    "metrics": (
        "run_id candidate_id layer metric value",
        "run_id candidate_id layer metric",
    ),
    "trials": (
        "run_id candidate_id unit_id series_id fold_id layer origin status "
        "fit_seconds predict_seconds reason",
        "run_id candidate_id unit_id",
    ),
    "trial_metrics": (
        "run_id candidate_id unit_id metric value",
        "run_id candidate_id unit_id metric",
    ),
    "ledger": (
        "run_id seq candidate_id label family strategy stage status parent reason "
        "score coverage n_failed_units seconds cached",
        "run_id seq",
    ),
    "decisions": (
        "run_id step family champion_before_id champion_before_label challenger_id "
        "challenger_label accepted reason_codes median_gain win_rate "
        "variability_status variability_value p_value challenger_coverage",
        "run_id step",
    ),
    "events": (
        "run_id seq kind message at duration_ms",
        "run_id seq",
    ),
    "models": (
        "run_id model_id candidate_id label series_id variant training_cutoff "
        "n_observations seed manifest",
        "run_id model_id",
    ),
    "external": (
        "run_id model_id label series_id metric value",
        "run_id model_id series_id metric",
    ),
    "predictions": (
        "run_id model_id series_id timestamp origin horizon y_pred y_true",
        "run_id model_id series_id timestamp origin",
    ),
}
TABLES: Final[dict[str, tuple[tuple[str, ...], tuple[str, ...]]]] = {
    table: (tuple(columns.split()), tuple(key.split())) for table, (columns, key) in _SCHEMA.items()
}


class ExperimentStore:
    """Experiments, evaluations and finalized models in one SQLite file.

    The file and its tables are created on the first write. ``log`` records an experiment
    as a run; ``query`` runs SQL on a read-only connection and returns a ``DataFrame``.

    Args:
        path: SQLite file (created if needed on the first ``log``).
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def __repr__(self) -> str:
        return f"ExperimentStore({str(self.path)!r})"

    # ------------------------------------------------------------------ writing
    def log(
        self,
        experiment: AutoHybridExperiment,
        *,
        name: str = "run",
        tags: Mapping[str, Any] | None = None,
        models: Sequence[FinalizedForecaster | PanelForecaster] = (),
        values: bool = False,
    ) -> str:
        """Record ``experiment`` as a run and return its ``run_id``.

        Everything the experiment has computed so far is written: series identity (no
        values), splits, evaluated candidates with per-layer and per-origin metrics and,
        after ``compare_models``, the search ledger, the selection decisions and the
        champion. ``models`` adds finalized models; external evaluations already made are
        written as metrics and, with ``values=True``, as predictions with observations.

        The ``run_id`` depends on the experiment's settings, data, temporal plan and
        ``name``: logging the same experiment again replaces its rows (for instance after
        ``evaluate_external``) instead of duplicating them.
        """

        if getattr(experiment, "state_", "CREATED") == "CREATED":
            raise ConfigurationError("Chame fit antes de registrar o experimento.")
        if not isinstance(name, str) or not name.strip():
            raise ConfigurationError("name deve ser um texto não vazio.")
        exp = experiment
        params = exp._public_params()
        panel = exp.panel_
        run_id = stable_hash(
            {
                "params": params,
                "data": panel.fingerprint,
                "plan": exp.work_plan_.fingerprint,
                "name": name,
            }
        )
        rows = _experiment_rows(exp, run_id, name, dict(tags or {}), params)
        models_rows, external_rows, prediction_rows = _model_rows(exp, run_id, models, values)
        rows["models"] = models_rows
        rows["external"] = external_rows
        rows["predictions"] = prediction_rows
        with closing(self._connect()) as con, con:
            for table in TABLES:
                con.execute(f"DELETE FROM {table} WHERE run_id = ?", (run_id,))
            for table, table_rows in rows.items():
                _insert(con, table, table_rows)
        return run_id

    def delete_run(self, run_id: str) -> None:
        """Remove every row of ``run_id``."""

        with closing(self._connect()) as con, con:
            for table in TABLES:
                con.execute(f"DELETE FROM {table} WHERE run_id = ?", (run_id,))

    # ------------------------------------------------------------------ reading
    def query(self, sql: str, params: Sequence[Any] | Mapping[str, Any] = ()) -> pd.DataFrame:
        """Run ``sql`` on a read-only connection and return the result as a ``DataFrame``.

        Pass values through ``params`` (``?`` or ``:name`` placeholders), never by
        formatting them into the SQL text.
        """

        with closing(self._connect(read_only=True)) as con:
            return pd.read_sql_query(sql, con, params=params)

    def runs(self) -> pd.DataFrame:
        """One row per run, newest first, without the JSON columns."""

        columns = [c for c in TABLES["runs"][0] if c not in ("params", "tags", "snapshot")]
        return self.query(f"SELECT {', '.join(columns)} FROM runs ORDER BY logged_utc DESC, run_id")

    def tables(self) -> list[str]:
        """Names of the tables of the store's schema."""

        return list(TABLES)

    # ------------------------------------------------------------------ internals
    def _connect(self, *, read_only: bool = False) -> sqlite3.Connection:
        import sqlite3

        if read_only:
            if not self.path.is_file():
                raise ConfigurationError(f"O banco {self.path} ainda não existe; use log antes.")
            con = sqlite3.connect(f"{self.path.resolve().as_uri()}?mode=ro", uri=True)
            _check_schema(con)
            return con
        self.path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(self.path)
        try:
            with con:
                con.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
                found = con.execute("SELECT value FROM meta WHERE key = 'schema'").fetchone()
                if found is None:
                    con.execute("INSERT INTO meta VALUES ('schema', ?)", (STORE_SCHEMA,))
                for table, (columns, key) in TABLES.items():
                    con.execute(
                        f"CREATE TABLE IF NOT EXISTS {table} "
                        f"({', '.join(columns)}, PRIMARY KEY ({', '.join(key)}))"
                    )
            _check_schema(con)
        except BaseException:
            con.close()
            raise
        return con


def _check_schema(con: sqlite3.Connection) -> None:
    import sqlite3

    try:
        found = con.execute("SELECT value FROM meta WHERE key = 'schema'").fetchone()
    except sqlite3.DatabaseError as error:
        raise ArtifactCompatibilityError("O arquivo não é um banco do AutoHybTS.") from error
    if found is None or found[0] != STORE_SCHEMA:
        raise ArtifactCompatibilityError(
            f"Esquema {None if found is None else found[0]!r} não suportado; "
            f"esperado {STORE_SCHEMA!r}."
        )


def _insert(con: sqlite3.Connection, table: str, rows: Iterable[Mapping[str, Any]]) -> None:
    columns = TABLES[table][0]
    sql = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({', '.join('?' * len(columns))})"
    con.executemany(sql, [tuple(_value(row.get(c)) for c in columns) for row in rows])


def _value(value: Any) -> Any:
    """A Python value SQLite can store; NaN and missing become ``NULL``."""

    if value is None or value is pd.NaT:
        return None
    if isinstance(value, enum.Enum):
        value = value.value
    if isinstance(value, bool | np.bool_):
        return int(value)
    if isinstance(value, int | np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        return None if math.isnan(float(value)) else float(value)
    if isinstance(value, pd.Timestamp | datetime):
        return value.isoformat()
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping | list | tuple):
        return json.dumps(value, ensure_ascii=False, default=str, sort_keys=True)
    return str(value)


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [{str(k): v for k, v in row.items()} for row in frame.to_dict(orient="records")]


def _experiment_rows(
    exp: AutoHybridExperiment,
    run_id: str,
    name: str,
    tags: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    compare = exp.compare_
    metric = exp.evaluator_.metrics[0]
    panel = exp.panel_
    run: dict[str, Any] = {
        "run_id": run_id,
        "name": name,
        "logged_utc": datetime.now(UTC).isoformat(),
        "autohybts_version": autohybts.__version__,
        "state": exp.state_,
        "variant": exp.variant,
        "fh": exp.fh,
        "profile": exp.profile,
        "budget": exp.budget,
        "session_id": exp.session_id,
        "n_series": len(panel.series_ids),
        "n_points": sum(len(frame) for frame in panel),
        "n_origins": len(exp.splits),
        "seasonal_period": exp.seasonal_period_,
        "ranking_metric": metric,
        "plan_fingerprint": exp.work_plan_.fingerprint,
        "data_fingerprint": panel.fingerprint,
        "params": params,
        "tags": tags,
    }
    if compare is not None:
        run.update(
            protocol_hash=compare.audit.protocol_hash,
            champion_id=compare.best.candidate_id,
            champion_label=compare.best.label,
            snapshot=compare.snapshot,
        )
    series = [
        {
            "run_id": run_id,
            "series_id": frame.series_id,
            "frequency": frame.frequency,
            "start": frame.start,
            "end": frame.end,
            "n_observations": len(frame),
            "fingerprint": frame.fingerprint,
        }
        for frame in panel
    ]
    splits = exp.splits.reset_index()
    if "series_id" not in splits:
        splits["series_id"] = panel.series_ids[0]
    split_rows = [{"run_id": run_id, **row} for row in _records(splits)]

    evaluations = {key: result for key, result in exp._evaluations.items() if key[1] is not None}
    champion_id = None if compare is None else compare.best.candidate_id
    candidates: dict[str, dict[str, Any]] = {}
    evaluation_rows, metric_rows, trial_rows, trial_metric_rows = [], [], [], []
    by_layer: dict[str, list[Any]] = {}
    for (candidate_id, layer), result in evaluations.items():
        by_layer.setdefault(str(layer), []).append(result)
        spec = result.candidate
        candidates[candidate_id] = {
            "run_id": run_id,
            "candidate_id": candidate_id,
            "label": spec.label,
            "name": spec.name,
            "family": spec.family,
            "strategy": spec.strategy,
            "variant": spec.variant,
            "base": spec.base.name,
            "corrector": None if spec.corrector is None else spec.corrector.name,
            "residual_source": spec.residual_source,
            "window": spec.window_size,
            "alpha": spec.alpha,
            "spec": spec.describe(),
        }
        for trial in result.trials:
            trial_rows.append(
                {
                    "run_id": run_id,
                    "candidate_id": candidate_id,
                    "unit_id": trial.unit_id,
                    "series_id": trial.series_id,
                    "fold_id": trial.fold_id,
                    "layer": layer,
                    "origin": trial.origin_time,
                    "status": trial.status,
                    "fit_seconds": trial.fit_seconds,
                    "predict_seconds": trial.predict_seconds,
                    "reason": trial.reason,
                }
            )
            trial_metric_rows += [
                {
                    "run_id": run_id,
                    "candidate_id": candidate_id,
                    "unit_id": trial.unit_id,
                    "metric": key,
                    "value": value,
                }
                for key, value in (trial.metrics or {}).items()
            ]
    from autohybts.reporting.leaderboard import leaderboard

    for layer, results in by_layer.items():
        table = leaderboard(results, metric, champion_id=champion_id)
        names = results[0].metric_names
        for row in _records(table.reset_index()):
            evaluation_rows.append(
                {
                    "run_id": run_id,
                    "layer": layer,
                    **{c: row.get(c) for c in TABLES["evaluations"][0][3:]},
                    "candidate_id": row["candidate_id"],
                }
            )
            metric_rows += [
                {
                    "run_id": run_id,
                    "candidate_id": row["candidate_id"],
                    "layer": layer,
                    "metric": m,
                    "value": row[m],
                }
                for m in names
            ]

    ledger_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    if compare is not None:
        ledger_rows = [
            {"run_id": run_id, "seq": seq, **row}
            for seq, row in enumerate(_records(compare.search.ledger_frame()))
        ]
        decision_rows = [
            {
                "run_id": run_id,
                "step": step,
                "family": row.get("family"),
                "champion_before_id": row.get("champion_before_id"),
                "champion_before_label": row.get("champion_before"),
                "challenger_id": row.get("challenger_id"),
                "challenger_label": row.get("challenger_label"),
                "accepted": row.get("accepted"),
                "reason_codes": row.get("reason_codes"),
                "median_gain": row.get("median_gain"),
                "win_rate": row.get("win_rate"),
                "variability_status": row.get("variability_status"),
                "variability_value": row.get("variability_value"),
                "p_value": row.get("p_value"),
                "challenger_coverage": row.get("challenger_coverage"),
            }
            for step, row in enumerate(_records(compare.decisions.reset_index(drop=True)))
        ]
    event_rows = [
        {
            "run_id": run_id,
            "seq": seq,
            "kind": event.kind,
            "message": event.message,
            "at": event.at,
            "duration_ms": event.duration_ms,
        }
        for seq, event in enumerate(exp.events)
    ]
    return {
        "runs": [run],
        "series": series,
        "splits": split_rows,
        "candidates": list(candidates.values()),
        "evaluations": evaluation_rows,
        "metrics": metric_rows,
        "trials": trial_rows,
        "trial_metrics": trial_metric_rows,
        "ledger": ledger_rows,
        "decisions": decision_rows,
        "events": event_rows,
    }


def _model_rows(
    exp: AutoHybridExperiment,
    run_id: str,
    models: Sequence[FinalizedForecaster | PanelForecaster],
    values: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    from autohybts.forecasting.finalized import FinalizedForecaster, PanelForecaster

    model_rows = []
    for model in models:
        if not isinstance(model, FinalizedForecaster | PanelForecaster):
            raise ConfigurationError("models deve conter modelos finalizados.")
        manifest = dict(model.manifest)
        single = model if isinstance(model, FinalizedForecaster) else None
        model_rows.append(
            {
                "run_id": run_id,
                "model_id": model.model_id,
                "candidate_id": model.spec.candidate_id,
                "label": model.spec.label,
                "series_id": None if single is None else single.history.series_id,
                "variant": model.spec.variant,
                "training_cutoff": None if single is None else single.training_cutoff,
                "n_observations": manifest.get("n_observations"),
                "seed": manifest.get("seed"),
                "manifest": manifest,
            }
        )
    external_rows, prediction_rows = [], []
    for model_id, external in exp._external.items():
        table = external.metrics.reset_index()
        names = [c for c in table.columns if c not in ("model", "series_id")]
        for row in _records(table):
            # A panel adds a mean row over its series: it is stored with series_id NULL.
            series_id = row.get("series_id", external.predictions["series_id"].iloc[0])
            external_rows += [
                {
                    "run_id": run_id,
                    "model_id": model_id,
                    "label": row["model"],
                    "series_id": series_id if series_id in exp.panel_.series_ids else None,
                    "metric": metric,
                    "value": row[metric],
                }
                for metric in names
            ]
        if values:
            table = external.predictions.reset_index()
            prediction_rows += [
                {
                    "run_id": run_id,
                    "model_id": model_id,
                    **{c: row.get(c) for c in TABLES["predictions"][0][2:]},
                }
                for row in _records(table)
            ]
    return model_rows, external_rows, prediction_rows
