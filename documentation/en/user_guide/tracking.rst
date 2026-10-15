Experiment tracking in SQLite
=============================

``ExperimentStore`` writes experiments to an SQLite file so that runs, series and settings
can be compared with SQL queries. It uses only the standard library's ``sqlite3``: there is
no new dependency and nothing is written unless asked.

.. code-block:: python

   from autohybts import AutoHybridExperiment, ExperimentStore

   store = ExperimentStore("artifacts/experiments.db")

   exp = AutoHybridExperiment(fh=14, session_id=42).fit(y)
   cmp = exp.compare_models()
   final = exp.finalize_model(cmp.best)
   exp.evaluate_external(final)

   run_id = store.log(exp, name="baseline", tags={"author": "ana"}, models=[final])
   store.runs()                                   # one row per run

``log`` writes what the experiment has already computed; it never evaluates candidates
again and never reads the sealed test. It also works without ``compare_models``: an
experiment with only ``create_model`` writes its evaluations, with no champion or
decisions.

What is written
---------------

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Table
     - Contents
   * - ``runs``
     - One row per run: name, version, variant, horizon, profile, budget, seed, number
       of series and origins, ranking metric (``ranking_metric``), protocol and data
       hashes, champion, parameters, tags and snapshot (JSON).
   * - ``series``
     - Identity of each series: frequency, start, end, length and fingerprint (no
       values).
   * - ``splits``
     - Origins of each layer (``search`` and ``select``) per series.
   * - ``candidates``
     - Each evaluated candidate: label, family, strategy, base, corrector, window,
       ``alpha`` and the full specification (JSON).
   * - ``evaluations`` and ``metrics``
     - Summary per candidate and layer (mean rank, coverage, cost, champion) and the mean
       metrics in long format (``metric``, ``value``).
   * - ``trials`` and ``trial_metrics``
     - Each origin of each candidate: status, timings, failure reason and metrics.
   * - ``ledger``
     - Every search proposal, with stage and status (``EVALUATED``, ``PRUNED_BUDGET`` …).
   * - ``decisions``
     - Each comparison of the incremental selection: challenger, acceptance, reasons,
       median gain, win rate and variability.
   * - ``events``
     - The experiment's events, with durations.
   * - ``models``, ``external`` and ``predictions``
     - Finalized models passed in ``models``, external evaluation metrics (per series; in
       a panel, the mean has a null ``series_id``) and, only with ``values=True``,
       forecasts with the observations.

Undefined metrics are ``NULL``, never 0. Series values are not written by default, as in
``export_evidence``; ``values=True`` writes the forecasts and observations of the external
test.

Runs and logging again
----------------------

The ``run_id`` depends on the experiment's parameters, data, temporal plan and ``name``.
Logging the same experiment again with the same name replaces the run's rows: ``log`` can
be called after ``compare_models`` and again after ``evaluate_external``. A different name
creates another run; ``store.delete_run(run_id)`` removes one.

Queries
-------

``store.query`` runs SQL on a read-only connection and returns a ``DataFrame``. Pass values
through ``params`` (``?`` placeholders), never by formatting them into the query text.

.. code-block:: python

   # error of each candidate in the selection layer of one run
   store.query(
       """
       SELECT c.label, c.family, m.value AS mase
       FROM metrics m JOIN candidates c USING (run_id, candidate_id)
       WHERE m.run_id = ? AND m.layer = 'select' AND m.metric = 'MASE'
       ORDER BY mase
       """,
       (run_id,),
   )

   # champion and external error of each run
   store.query(
       """
       SELECT r.name, r.champion_label, x.value AS external_mase
       FROM runs r JOIN external x USING (run_id)
       WHERE x.metric = 'MASE' AND x.series_id IS NOT NULL
       """
   )

   # why challengers were rejected
   store.query(
       "SELECT family, reason_codes, COUNT(*) AS n FROM decisions "
       "WHERE accepted = 0 GROUP BY family, reason_codes"
   )

The file is an ordinary SQLite database: it also opens in the ``sqlite3`` client, in DB
Browser for SQLite or with ``pandas.read_sql_query``.

Limits
------

* The database is a report: it never feeds the search, the selection or the thresholds.
* Models do not go into the database. Save them with ``save_model`` and keep the path.
* The schema is versioned (``autohybts-store/1``, table ``meta``); a file with another
  schema or that is not a database is refused.
* It is a local file, not meant for heavy concurrent access.

.. seealso::

   Tutorials: :doc:`../tutorials/notebooks/18_sql_analysis`,
   :doc:`../tutorials/notebooks/19_mlops_flow`, :doc:`../tutorials/notebooks/22_m4_development_pilot`.
   Models in production: :doc:`mlops`.
