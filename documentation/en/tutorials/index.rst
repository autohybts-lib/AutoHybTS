Tutorials
=========

Short notebooks, one topic each, in the style of the PyCaret tutorials. They live in
``notebooks/en/`` in the repository, with their outputs, which are shown here.

* :doc:`notebooks/01_quickstart`: the whole flow on one series, with figures.
* :doc:`notebooks/02_hybrid_selection`: how the search builds hybrids and why a challenger is accepted or rejected.
* :doc:`notebooks/03_finalize_and_persist`: finalize, evaluate once on the sealed test, save, load and forecast in production.
* :doc:`notebooks/04_rolling_one_step`: variant B streaming, base and correction at each step, A × B on the same stretch.
* :doc:`notebooks/05_benchmark_m4`: M4, M3 and Rdatasets with the official test sealed (downloads data on the first run).
* :doc:`notebooks/06_compare_and_cross_validate`: statistical comparison of finalists and temporal cross-validation.
* :doc:`notebooks/07_panel`: several series in the same experiment.
* :doc:`notebooks/08_advanced_families_and_tuning`: stages, decomposition, combination and hyperparameter tuning.
* :doc:`notebooks/09_ablation`: ablation of the selection procedure (heavy: about 1 h).
* :doc:`notebooks/10_external_models`: PyTorch, TensorFlow/Keras and other libraries' models (heavy).
* :doc:`notebooks/11_calibration`: calibration of the acceptance rule on controls with a known answer.
* :doc:`notebooks/12_baselines_and_boosting`: baselines, boosting and optional correctors.
* :doc:`notebooks/13_ensembles`: ensembles of models and of hybrids.
* :doc:`notebooks/14_intermittent_demand`: intermittent demand: Croston, TSB, ADIDA, IMAPA.
* :doc:`notebooks/15_statsforecast`: StatsForecast's automatic models (environment with pandas 2.x).
* :doc:`notebooks/16_series_diagnostics`: seasonality, stationarity, differences and characterization.
* :doc:`notebooks/17_benchmark_study`: 17 classical series in three classes, five models (with the ARIMA–ANN hybrid), protocols A and B, 3, 6 and 12 lags and horizons up to 12 (downloads data on the first run).
* :doc:`notebooks/18_sql_analysis`: experiments stored in SQLite and analysed with SQL queries.

.. toctree::
   :hidden:

   notebooks/01_quickstart
   notebooks/02_hybrid_selection
   notebooks/03_finalize_and_persist
   notebooks/04_rolling_one_step
   notebooks/05_benchmark_m4
   notebooks/06_compare_and_cross_validate
   notebooks/07_panel
   notebooks/08_advanced_families_and_tuning
   notebooks/09_ablation
   notebooks/10_external_models
   notebooks/11_calibration
   notebooks/12_baselines_and_boosting
   notebooks/13_ensembles
   notebooks/14_intermittent_demand
   notebooks/15_statsforecast
   notebooks/16_series_diagnostics
   notebooks/17_benchmark_study
   notebooks/18_sql_analysis
