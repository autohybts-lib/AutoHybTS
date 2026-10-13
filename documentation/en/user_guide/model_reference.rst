Model reference
===============

For every registered model: the parameters it takes without configuration, what the
automatic search passes to it, the space ``tune_model`` explores and where it runs.
The tables are generated from the code at every documentation build, so they show what
the installed version does. The implementation of each model (source library) is in
:doc:`models`.

How to read
-----------

* **★** marks the models of the default search (``compare_models`` without
  ``base_models``).
* **Default parameters:** what ``exp.create_model(name)`` uses without arguments. In
  lag-based models, ``estimator_params`` shows what AutoHybTS changes relative to the
  library default (for example, one thread in the boosting models).
* **In the search:** what ``compare_models`` passes to the model when you give no
  grid; the seasonal period and the lags come from the experiment (here, a daily
  series: 7).
* ``tune_model`` **space:** distributions of ``exp.tune_model(name)``. ``[a, b]`` is a
  range (integer or continuous, ``(log)`` on a log scale) and ``{...}`` the choices. In
  lag-based models, ``lags`` goes from 2 to ``4 × period`` (at least 14, at most 60).
  "—" means the model chooses by itself (``arima``, StatsForecast) or has nothing to
  tune.
* **Variants:** A (several steps from one origin) and B (one step, frozen parameters).

Changing the parameters
-----------------------

.. code-block:: python

   # one model, one configuration
   exp.create_model("knn", lags=14, estimator_params={"n_neighbors": 3})

   # a grid in the search: each dict is a candidate
   exp = AutoHybridExperiment(
       fh=14,
       base_models=["knn", "lightgbm_regressor"],
       base_params={"knn": [{"lags": 7}, {"lags": 14, "estimator_params": {"n_neighbors": 3}}]},
   )

   # tuning on V_search, with the built-in space or your own
   exp.tune_model("lightgbm_regressor", n_trials=30)
   exp.tune_model("knn", search_space={"lags": (7, 28), "estimator_params.n_neighbors": [3, 5, 9]})

Base models
-----------

.. include:: _generated/base_models.rst

Residual correctors
-------------------

★ marks the correctors of the default search. **Window** correctors regress the next
residual on the previous ``window`` ones; **sequence** correctors model the residual
series and forecast the whole horizon (:doc:`models`). In the search, corrector
parameters vary through ``corrector_params`` (one grid per name), and ``tune_model``
tunes the base only.

.. include:: _generated/correctors.rst

Options shared by lag-based models and window correctors:

* ``scale``: ``True`` (mean and deviation of the series), ``"columns"`` (per column,
  like ``StandardScaler``), ``"minmax"`` (per column, to [0, 1]) or ``False``;
* ``differencing`` (bases only): differences fitted on each origin's training data;
* ``random_state``: seed; in the search, each origin gets a seed derived from the
  experiment's seed, unless you fix it;
* ``feature_order`` (correctors only): ``"chronological"`` or ``"lag"``.
