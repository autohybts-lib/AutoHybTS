StatsForecast (Nixtla)
======================

The automatic models of `StatsForecast <https://nixtlaverse.nixtla.io/statsforecast/>`_
are available as optional base models, with their own names (prefix ``sf_``) so they
are not confused with statsmodels' ``arima``.

Installation
------------

.. code-block:: bash

   python -m pip install "autohybts[nixtla-stats]==1.0.0rc4"

Current StatsForecast versions (2.1) require ``pandas<3``: the extra installs pandas
2.x, which AutoHybTS supports (floor 2.1). In an environment with pandas 3, keep a
second virtual environment for these models. Without the extra, a candidate using them
is ``SKIPPED`` and the reason says what to install.

Models
------

.. list-table::
   :header-rows: 1
   :widths: 18 22 60

   * - Name
     - Class
     - What it chooses by itself
   * - ``sf_auto_arima``
     - ``AutoARIMA``
     - order (p, d, q) and, with ``seasonal``, (P, D, Q) by information criterion
   * - ``sf_auto_ets``
     - ``AutoETS``
     - error, trend and seasonality (``model="ZZZ"``)
   * - ``sf_auto_theta``
     - ``AutoTheta``
     - variant of the Theta method and seasonal decomposition
   * - ``sf_auto_ces``
     - ``AutoCES``
     - complex exponential smoothing, with or without seasonality

Parameters: ``season_length`` and ``model_params`` (passed on to the StatsForecast
class). In the search, ``season_length`` comes from the experiment's seasonal period;
in ``create_model``, pass it explicitly:

.. code-block:: python

   exp.create_model("sf_auto_ets", season_length=7)
   exp.create_model("sf_auto_arima", season_length=7,
                    model_params={"seasonal": True, "max_D": 1})
   exp.create_hybrid("sf_auto_theta", "mlp", base_params={"season_length": 7})

Frozen parameters and variants A/B
----------------------------------

Each model is fitted on the origin's training vector and then frozen. Forecasting calls
StatsForecast's ``forward(y=context, h)``, which applies the fitted model (orders,
coefficients, smoothing constants) to the observed context without re-estimating it:
on the training data itself it reproduces ``predict``; on a longer context only the
states change. That is why these models work in variant B and as hybrid bases (the
out-of-sample residuals come from the same update), and pass the library's contract and
causality tests.

Cost
----

``AutoARIMA`` tries several orders at each fit (a few seconds on a daily series of a few
hundred points); ``AutoETS``, ``AutoTheta`` and ``AutoCES`` are much faster. A hybrid
fits the base twice per origin (training and residual prefix).

Parity with the replication project
-----------------------------------

The seven hybrids of the replication project are reproduced exactly by AutoHybTS
specifications (test ``tests/regression/test_legacy_parity.py``): four with a KNN
corrector and three statistical-ANN. An ETS–ANN, for example:

.. code-block:: python

   exp.create_hybrid(
       "sf_auto_ets", "mlp", window=7,
       base_params={"season_length": 7, "model_params": {"model": "ZZZ"}},
       corrector_params={
           "scale": "minmax", "feature_order": "lag", "random_state": 42,
           "estimator_params": {"hidden_layer_sizes": [5], "activation": "logistic",
                                "solver": "lbfgs", "max_iter": 2000},
       },
   )

``scale="minmax"`` (per-column scaling like ``MinMaxScaler``) and ``feature_order="lag"``
(window columns from lag 1 onwards) exist for this parity; the library default is still
``scale=True`` and chronological order. The project's fixed residual tail
(``residual_tail``) corresponds to ``ResidualSettings(max_tail=...)`` when building the
model (``strategies.base.build_model``); the experiment uses the default tail (half of
the training data).

Limitations
-----------

* Local point models only: StatsForecast's prediction intervals are not used.
* MLForecast and NeuralForecast are not integrated.
* No covariates (``target_mode="univariate"``).

.. seealso::

   Tutorials: :doc:`../tutorials/notebooks/15_statsforecast`.
