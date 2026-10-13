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
read only by ``evaluate_external``.

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

The search runs on ``V_search``; the champion is chosen on ``V_select``. To include
the other strategies for building the system:

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
