Multiple series (panel)
=======================

``fit`` accepts a dict of series, a :class:`~autohybts.PanelFrame` or a long table:

.. code-block:: python

   exp = AutoHybridExperiment(fh=14).fit({"store_a": y_a, "store_b": y_b})
   exp = AutoHybridExperiment(fh=14).fit(
       long, series_col="series_id", date_col="timestamp", target="value"
   )

Each series has its own sealed test, origins and MASE scale. The comparison units are
``series:fold``.

* **Ranking**: best configuration of each structure in each series, ranks per series,
  mean of the ranks.
* **Selection**: median gain and wins over the units of all series together;
  variability is the median per series and then across series;
  ``result.selection.champion_per_series`` keeps the configuration of each series.
* **Finalization**: ``finalize_model`` returns a :class:`~autohybts.PanelForecaster`,
  one model per series, each with its own configuration.
* **External evaluation**: one row per series plus the mean.

.. code-block:: python

   model = exp.finalize_model(result.best)
   exp.predict_model(model).predictions
   exp.evaluate_external(model).metrics
   exp.plot_model(result.best, "cv_forecasts", series_id="store_b")

Nested CV also works on panels: each series is cut at its own outer origin, the inner
experiment runs on the truncated panel and the table has one row per fold and series.

.. code-block:: python

   nested = exp.nested_cross_validate(outer_folds=3)
   nested.folds        # index (fold_id, series_id)
   nested.summary
