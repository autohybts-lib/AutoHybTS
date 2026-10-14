Models in production (MLOps)
============================

The same ``ExperimentStore`` that logs experiments keeps the life cycle of a model in
operation: registration, promotion, forecasts made in production, observations that arrived
later and the real error over time. The library provides these pieces; when to train, serve
and retrain (scheduler, cron, Airflow …) is up to your code.

Registering and promoting
-------------------------

.. code-block:: python

   from autohybts import AutoHybridExperiment, ExperimentStore, save_model

   store = ExperimentStore("artifacts/mlops.db")

   exp = AutoHybridExperiment(fh=1, variant="B", test_size=60).fit(history)
   final = exp.finalize_model(exp.compare_models().best)
   mae = float(exp.evaluate_external(final).metrics["MAE"].iloc[0])

   run_id = store.log(exp, name="training", models=[final], values=True)
   path = save_model(final, "artifacts/models/v1")
   store.register_model(final, path, name="price", run_id=run_id, reference_mae=mae)
   store.promote(final.model_id, reason="first model")

``register_model`` keeps the artifact's path and the manifest's SHA-256, never the model,
and enters it as ``candidate``. ``promote`` moves the model to ``production`` (the previous
one with the same name goes to ``archived``) or to another stage; every change is kept in
``stage_history`` with its reason. ``reference_mae`` is the error expected in production,
usually the MAE of the sealed test.

Serving
-------

.. code-block:: python

   model = store.load("price", trusted=True)             # checks the registered SHA-256
   model = model.update_state(new_observations)          # catches up, no refit
   stream = model.stream()

   ticket = stream.predict_next()
   store.log_forecasts(model.model_id, pd.DataFrame(
       {"timestamp": [ticket.timestamp], "origin": [ticket.origin], "y_pred": [ticket.prediction]}
   ))
   stream.observe(value)                                  # when the value arrives
   store.log_actuals(model.history.series_id, observed)   # pd.Series indexed by date

``load`` requires ``trusted=True``, like ``load_model``. The registered hash sits in the same
database that points to the artifact, so it catches corruption, not tampering; for that,
save with ``signing_key`` and pass the same key to ``load``. ``update_state`` appends
observations without refitting the parameters. ``log_forecasts`` also accepts the tables of
``predict_model`` and of ``ForecastStream.log()``; an observation logged again for the same
date replaces the previous one.

Monitoring
----------

.. code-block:: python

   status = store.monitor(model.model_id, window=30, threshold=1.5)
   status.alert.iloc[-1]

One row per forecast already observed, with the absolute error, the MAE of the last
``window`` forecasts, the ratio to ``reference_mae`` and ``alert``: a full window with MAE
above ``threshold`` times the reference. Without a reference there is no alert.

Retraining with a gate
----------------------

Train a challenger on the current history, keeping the last days sealed, and compare its
error on those days with production's real error on the same dates, already in the
database:

.. code-block:: python

   live = store.query(
       """
       SELECT AVG(ABS(a.y_true - f.y_pred)) AS mae
       FROM forecasts f JOIN actuals a USING (series_id, timestamp)
       WHERE f.model_id = ?
         AND f.timestamp IN (SELECT timestamp FROM predictions WHERE run_id = ?)
       """,
       (production_id, challenger_run_id),
   ).mae[0]
   if challenger_mae <= live:
       store.promote(challenger.model_id, reason=f"MAE {challenger_mae:.3f} <= {live:.3f}")

Both measures are out of sample and on the same dates. The "not worse" rule replaces, on a
tie, the model with one trained on more recent data, which also refreshes the monitoring
reference after a regime change.

Operation tables
----------------

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Table
     - Contents
   * - ``registry``
     - One model per row: name, source run, candidate, label, variant, series, artifact
       path, SHA-256, stage, ``reference_mae`` and registration date.
   * - ``stage_history``
     - Each stage change, with date and reason.
   * - ``forecasts``
     - Forecasts made in production: model, series, origin, forecast date, horizon and value.
   * - ``actuals``
     - Observations of each series by date.

These tables do not belong to a run: ``log`` and ``delete_run`` leave them alone.

Limits
------

* Scheduling and deciding when to retrain stay in your code; the library records, loads and
  measures.
* ``monitor`` measures MAE. Other metrics can be computed with SQL over ``forecasts`` and
  ``actuals``.
* The database is a local file, not meant for heavy concurrent access.

.. seealso::

   Tutorial: :doc:`../tutorials/notebooks/19_mlops_flow`.
