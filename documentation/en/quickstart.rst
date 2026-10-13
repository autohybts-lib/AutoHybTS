Quickstart
==========

An AutoHybTS experiment follows the PyCaret workflow: create, fit, compare, finalize,
predict. Every method returns an immutable result.

1. Create the experiment and seal the test
------------------------------------------

.. code-block:: python

   from autohybts import AutoHybridExperiment
   from autohybts.datasets import get_data

   y = get_data("synthetic_daily")            # pd.Series with a regular DatetimeIndex
   exp = AutoHybridExperiment(fh=14, session_id=42).fit(y)
   exp.splits                                  # V_search and V_select origins

``fit`` validates the series, holds back the last ``test_size`` points (default:
``fh``) as the **sealed test** and freezes the validation origins. The sealed test is
read only by ``evaluate_external``. The terms (origin, ``V_search``, ``V_select``, base,
corrector …) are in :doc:`concepts`.

By default the forecast follows **variant A**: ``fh`` steps at once from each origin. To
forecast one step at a time, receiving each actual value before the next one, use
``variant="B"`` (:doc:`user_guide/variant_b`).

2. Explore the data
-------------------

.. code-block:: python

   exp.data_profile()
   exp.plot_data("seasonality")                # also: series, acf, distribution

3. Compare models and hybrids
-----------------------------

.. code-block:: python

   result = exp.compare_models()
   result.leaderboard        # every candidate of the search (short name, e.g. arima-mlp)
   result.decisions          # incremental chain: challenger, gain, wins, reason
   result.best               # the champion

The search runs on ``V_search``; the champion is chosen on ``V_select``. What to look
at:

* ``leaderboard``: one row per evaluated candidate, sorted by the first metric (MASE by
  default; lower is better), with coverage and cost;
* ``decisions``: one row per family that challenged the champion, with ``accepted``,
  ``median_gain`` (paired median gain), ``win_rate`` and the reason for a rejection.

.. note::

   With the defaults (``small`` budget and 7 origins per layer), ``compare_models`` can
   take minutes: the whole notebook 01, on the ``synthetic_daily`` series, took about
   1 min, and larger series, more models or a larger budget take longer. For a first
   contact,
   ``AutoHybridExperiment(fh=14, budget="smoke", fold=3)`` finishes in seconds, with the
   caveat that the acceptance rule was calibrated for 7 origins
   (:doc:`user_guide/selection`).

To include the other strategies for building the system:

.. code-block:: python

   result = exp.compare_models(
       include=("standalone", "residual", "multistage", "decomposition", "combination")
   )
   result.strategies
   result.plot("pareto")

4. Inspect a model
------------------

.. code-block:: python

   hybrid = exp.create_hybrid("arima", "mlp")
   hybrid.cv_results
   exp.plot_model(hybrid, "cv_forecasts")      # also: residuals, metric_by_horizon …

5. Finalize, predict and evaluate once
--------------------------------------

.. code-block:: python

   final = exp.finalize_model(result.best)     # refit on the history before the test
   exp.predict_model(final).predictions
   exp.evaluate_external(final).metrics        # a single evaluation per model

6. Save
-------

.. code-block:: python

   from autohybts import load_model, save_model

   path = save_model(final, "artifacts/champion")
   model = load_model(path, trusted=True)

Operational forecasting
-----------------------

Without a sealed test (``test_size=0``), or with the one-call estimator:

.. code-block:: python

   from autohybts import AutoHybridForecaster

   forecaster = AutoHybridForecaster(fh=14).fit(y)
   forecaster.predict()

Next steps
----------

* :doc:`concepts`: the vocabulary and the path of the data, from the split to the
  forecast.
* Tutorials :doc:`tutorials/notebooks/01_quickstart` to
  :doc:`tutorials/notebooks/04_rolling_one_step`: the same flow with figures, hybrid
  selection, finalization and variant B.
* :doc:`user_guide/diagnostics`: seasonality, stationarity and differences before
  choosing models.
