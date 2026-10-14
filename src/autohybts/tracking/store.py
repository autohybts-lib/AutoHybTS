"""SQLite store of experiments, evaluations, registered models and forecasts (ADR-0010).

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
    from autohybts.results import ExternalResult

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

# Operations: registered models, their stage history, forecasts made in production and the
# observations that arrived later. They outlive runs, so ``log`` and ``delete_run`` keep them.
_OPERATIONS: Final = {
    "registry": (
        "model_id name run_id candidate_id label variant series_id path sha256 stage "
        "reference_mae registered_utc",
        "model_id",
    ),
    "stage_history": ("model_id at stage reason", "model_id at stage"),
    "forecasts": (
        "model_id series_id origin timestamp horizon y_pred logged_utc",
        "model_id series_id timestamp horizon",
    ),
    "actuals": ("series_id timestamp y_true logged_utc", "series_id timestamp"),
}
OPERATION_TABLES: Final[dict[str, tuple[tuple[str, ...], tuple[str, ...]]]] = {
    table: (tuple(columns.split()), tuple(key.split()))
    for table, (columns, key) in _OPERATIONS.items()
}
STAGES: Final = ("candidate", "production", "archived")


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

        return [*TABLES, *OPERATION_TABLES]

    # ------------------------------------------------------------------ operations
    def register_model(
        self,
        model: FinalizedForecaster,
        path: str | Path,
        *,
        name: str,
        run_id: str | None = None,
        reference_mae: float | None = None,
    ) -> str:
        """Register a model saved with ``save_model`` under ``name``, as a ``candidate``.

        The registry keeps the artifact's path and the payload SHA-256 read from its
        manifest, never the model itself. ``reference_mae`` is the error the model is
        expected to keep in production (for instance the MAE of ``evaluate_external``);
        ``monitor`` compares the live error with it. Registering again keeps the stage.
        """

        from autohybts.forecasting.finalized import FinalizedForecaster
        from autohybts.persistence.model_store import read_manifest

        if not isinstance(model, FinalizedForecaster):
            raise ConfigurationError("register_model espera um FinalizedForecaster.")
        if not isinstance(name, str) or not name.strip():
            raise ConfigurationError("name deve ser um texto não vazio.")
        manifest = read_manifest(path)
        if manifest.get("model_id") != model.model_id:
            raise ArtifactCompatibilityError(
                f"O artefato em {path} não é deste modelo (model_id diferente)."
            )
        now = datetime.now(UTC).isoformat()
        with closing(self._connect()) as con, con:
            found = con.execute(
                "SELECT stage, registered_utc FROM registry WHERE model_id = ?", (model.model_id,)
            ).fetchone()
            row = {
                "model_id": model.model_id,
                "name": name,
                "run_id": run_id,
                "candidate_id": model.spec.candidate_id,
                "label": model.spec.label,
                "variant": model.spec.variant,
                "series_id": model.history.series_id,
                "path": str(Path(path)),
                "sha256": manifest["payload"]["sha256"],
                "stage": "candidate" if found is None else found[0],
                "reference_mae": reference_mae,
                "registered_utc": now if found is None else found[1],
            }
            _insert(con, "registry", [row], replace=True)
            if found is None:
                _insert(
                    con,
                    "stage_history",
                    [{"model_id": model.model_id, "at": now, "stage": "candidate"}],
                )
        return model.model_id

    def promote(self, model_id: str, stage: str = "production", reason: str = "") -> None:
        """Move a registered model to ``stage``; a new production model archives the old one.

        Every change is kept in ``stage_history`` with its ``reason``.
        """

        if stage not in STAGES:
            raise ConfigurationError(f"stage deve ser um de {STAGES}; recebido {stage!r}.")
        now = datetime.now(UTC).isoformat()
        with closing(self._connect()) as con, con:
            found = con.execute(
                "SELECT name, stage FROM registry WHERE model_id = ?", (model_id,)
            ).fetchone()
            if found is None:
                raise ConfigurationError(f"Modelo {model_id[:12]} não está registrado.")
            name, current = found
            if current == stage:
                return
            changes = [(model_id, stage, reason)]
            if stage == "production":
                changes += [
                    (old, "archived", f"substituído por {model_id[:12]}")
                    for (old,) in con.execute(
                        "SELECT model_id FROM registry WHERE name = ? AND stage = 'production'",
                        (name,),
                    ).fetchall()
                ]
            for changed, new_stage, why in changes:
                con.execute(
                    "UPDATE registry SET stage = ? WHERE model_id = ?", (new_stage, changed)
                )
                _insert(
                    con,
                    "stage_history",
                    [{"model_id": changed, "at": now, "stage": new_stage, "reason": why}],
                )

    def registry(self, name: str | None = None) -> pd.DataFrame:
        """Registered models (all, or those of ``name``), newest first."""

        sql = "SELECT * FROM registry"
        params: tuple[str, ...] = ()
        if name is not None:
            sql, params = sql + " WHERE name = ?", (name,)
        return self.query(sql + " ORDER BY registered_utc DESC, model_id", params)

    def load(
        self,
        name: str,
        stage: str = "production",
        *,
        trusted: bool = False,
        signing_key: bytes | None = None,
    ) -> FinalizedForecaster:
        """Load the model of ``name`` in ``stage`` with ``load_model``.

        The payload must match the SHA-256 recorded at registration. Like that hash, the
        database sits next to the artifacts, so it catches corruption, not tampering:
        pass ``signing_key`` for artifacts saved with one.
        """

        from autohybts.forecasting.finalized import FinalizedForecaster
        from autohybts.persistence.model_store import load_model

        found = self.query(
            "SELECT path, sha256 FROM registry WHERE name = ? AND stage = ? "
            "ORDER BY registered_utc DESC LIMIT 1",
            (name, stage),
        )
        if found.empty:
            raise ConfigurationError(f"Nenhum modelo {name!r} no estágio {stage!r}.")
        model = load_model(
            found.path[0], trusted=trusted, sha256=found.sha256[0], signing_key=signing_key
        )
        if not isinstance(model, FinalizedForecaster):
            raise ArtifactCompatibilityError("O registro só carrega modelos de uma série.")
        return model

    def log_forecasts(self, model_id: str, forecasts: pd.DataFrame) -> int:
        """Record forecasts of a registered model made in production; returns the row count.

        ``forecasts`` has ``y_pred`` and a ``timestamp`` column or index; ``origin``,
        ``horizon`` (default 1) and ``series_id`` (default: the model's series) are optional,
        as in the tables of ``predict_model`` and ``ForecastStream.log()``.
        """

        series_id = self._registered_series(model_id)
        table = forecasts
        if "timestamp" not in table and isinstance(table.index, pd.DatetimeIndex):
            table = table.rename_axis("timestamp").reset_index()
        if "y_pred" not in table or "timestamp" not in table:
            raise ConfigurationError("forecasts precisa das colunas timestamp e y_pred.")
        now = datetime.now(UTC).isoformat()
        rows = [
            {
                "model_id": model_id,
                "series_id": row.get("series_id", series_id),
                "origin": row.get("origin"),
                "timestamp": pd.Timestamp(row["timestamp"]),
                "horizon": row.get("horizon", 1),
                "y_pred": row["y_pred"],
                "logged_utc": now,
            }
            for row in _records(table)
        ]
        with closing(self._connect()) as con, con:
            _insert(con, "forecasts", rows, replace=True)
        return len(rows)

    def log_actuals(self, series_id: str, actuals: pd.Series) -> int:
        """Record observed values of ``series_id`` (indexed by timestamp); returns the count.

        A value logged again for the same timestamp replaces the previous one.
        """

        if not isinstance(actuals, pd.Series) or not isinstance(actuals.index, pd.DatetimeIndex):
            raise ConfigurationError("actuals deve ser uma pd.Series com DatetimeIndex.")
        now = datetime.now(UTC).isoformat()
        rows = [
            {"series_id": series_id, "timestamp": timestamp, "y_true": value, "logged_utc": now}
            for timestamp, value in actuals.items()
        ]
        with closing(self._connect()) as con, con:
            _insert(con, "actuals", rows, replace=True)
        return len(rows)

    def monitor(
        self, model_id: str, *, window: int = 30, threshold: float = 1.5, horizon: int = 1
    ) -> pd.DataFrame:
        """Live error of a registered model: rolling MAE against its ``reference_mae``.

        One row per forecast (of ``horizon``) whose observation has arrived, with the
        absolute error, the MAE over the last ``window`` such forecasts, the ratio to the
        reference and ``alert``: a full window whose MAE exceeds ``threshold`` times the
        reference. Without a reference, ``ratio`` is empty and ``alert`` is false.
        """

        if isinstance(window, bool) or not isinstance(window, int) or window < 1:
            raise ConfigurationError("window deve ser um inteiro positivo.")
        if not np.isfinite(threshold) or threshold <= 0:
            raise ConfigurationError("threshold deve ser finito e positivo.")
        self._registered_series(model_id)
        frame = self.query(
            f"""
            SELECT f.timestamp, f.y_pred, a.y_true,
                   ABS(a.y_true - f.y_pred) AS abs_error,
                   AVG(ABS(a.y_true - f.y_pred)) OVER w AS rolling_mae,
                   COUNT(*) OVER w AS n_window,
                   r.reference_mae
            FROM forecasts f
            JOIN actuals a ON a.series_id = f.series_id AND a.timestamp = f.timestamp
            JOIN registry r ON r.model_id = f.model_id
            WHERE f.model_id = ? AND f.horizon = ?
            WINDOW w AS (ORDER BY f.timestamp ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW)
            ORDER BY f.timestamp
            """,
            (model_id, horizon),
        )
        frame["ratio"] = frame.rolling_mae / frame.reference_mae
        frame["alert"] = (frame.n_window == window) & (frame.ratio > threshold)
        return frame.drop(columns="n_window")

    def _registered_series(self, model_id: str) -> str:
        found = self.query("SELECT series_id FROM registry WHERE model_id = ?", (model_id,))
        if found.empty:
            raise ConfigurationError(f"Modelo {model_id[:12]} não está registrado.")
        return str(found.series_id[0])

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
                for table, (columns, key) in {**TABLES, **OPERATION_TABLES}.items():
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


def _insert(
    con: sqlite3.Connection,
    table: str,
    rows: Iterable[Mapping[str, Any]],
    *,
    replace: bool = False,
) -> None:
    columns = {**TABLES, **OPERATION_TABLES}[table][0]
    verb = "INSERT OR REPLACE" if replace else "INSERT"
    marks = ", ".join("?" * len(columns))
    sql = f"{verb} INTO {table} ({', '.join(columns)}) VALUES ({marks})"
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
        external_rows += _external_metric_rows(exp, run_id, model_id, external)
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


def _external_metric_rows(
    exp: AutoHybridExperiment, run_id: str, model_id: str, external: ExternalResult
) -> list[dict[str, Any]]:
    table = external.metrics.reset_index()
    names = [c for c in table.columns if c not in ("model", "series_id")]
    rows = []
    for row in _records(table):
        # A panel adds a mean row over its series: it is stored with series_id NULL.
        series_id = row.get("series_id", external.predictions["series_id"].iloc[0])
        rows += [
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
    return rows
