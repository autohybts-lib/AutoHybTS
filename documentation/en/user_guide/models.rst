Models and libraries
====================

Most models are a thin adapter over a well-known library (scikit-learn, statsmodels,
StatsForecast, LightGBM ...): the adapter handles what is specific to time series (lag
windows, differencing, recursive forecasting, temporal context) and delegates the
fitting to the source library. Simple methods without a suitable reference
implementation — baselines, harmonic regression and the intermittent-demand methods —
are implemented in numpy in AutoHybTS itself and marked "own (numpy)" in the tables.
The short name is what you pass to ``create_model`` and ``create_hybrid``, and it shows
up in the leaderboard.

.. code-block:: python

   exp.models()     # name, role (base/corrector), family, capabilities and required extra

Base models
-----------

.. list-table::
   :header-rows: 1
   :widths: 16 12 40 32

   * - Name
     - Family
     - Implementation
     - Installation
   * - ``naive``
     - statistical
     - own (numpy): repeats the last value
     - core
   * - ``seasonal_naive``
     - statistical
     - own (numpy): repeats the last season
     - core
   * - ``arima``
     - statistical
     - ``statsmodels.tsa.arima.model.ARIMA``; order chosen with the KPSS test
       (``statsmodels.tsa.stattools.kpss``) and AIC
     - ``stats`` extra (statsmodels)
   * - ``linear``
     - linear
     - ``sklearn.linear_model.LinearRegression`` on lags
     - core (scikit-learn)
   * - ``knn``
     - kernel
     - ``sklearn.neighbors.KNeighborsRegressor`` on lags
     - core (scikit-learn)
   * - ``svm``
     - kernel
     - ``sklearn.svm.SVR`` on lags
     - core (scikit-learn)
   * - ``tree``
     - tree
     - ``sklearn.tree.DecisionTreeRegressor`` on lags
     - core (scikit-learn)
   * - ``random_forest``
     - tree
     - ``sklearn.ensemble.RandomForestRegressor`` on lags
     - core (scikit-learn)
   * - ``mlp``
     - neural
     - ``sklearn.neural_network.MLPRegressor`` on lags (``hidden_layer_sizes=(32,)``,
       ``max_iter=1000``)
     - core (scikit-learn)

These nine make up the default search. The models below are registered but only join
when requested (``base_models=[...]``, ``create_model``): simple controls for comparing
composite structures, and boosting on lags.

.. list-table::
   :header-rows: 1
   :widths: 22 12 38 28

   * - Name
     - Family
     - Implementation
     - Installation
   * - ``historic_average``
     - statistical
     - own (numpy): mean of the training data, frozen
     - core
   * - ``window_average``
     - statistical
     - own (numpy): mean of the last ``window`` observations
     - core
   * - ``seasonal_window_average``
     - statistical
     - own (numpy): mean of each seasonal position over the last ``window`` seasons
     - core
   * - ``random_walk_drift``
     - statistical
     - own (numpy): last value + ``h`` · mean change in the training data
     - core
   * - ``exponential_smoothing``
     - statistical
     - additive ``statsmodels.tsa.holtwinters.ExponentialSmoothing`` (Holt,
       Holt-Winters, damped: ``trend``, ``damped``, ``season_length``); parameters
       estimated on the training data and frozen, states updated by the context
     - ``stats`` extra (statsmodels)
   * - ``sarima``
     - statistical
     - ``statsmodels.tsa.arima.model.ARIMA`` with a seasonal part ``(P, D, Q, m)``: ``D``
       from the seasonal strength, ``d`` by KPSS and ``(p, q, P, Q)`` by AICc in a stepwise
       search (:doc:`diagnostics`)
     - ``stats`` extra (statsmodels)
   * - ``harmonic_regression``
     - statistical
     - own (numpy): dynamic harmonic regression, with trend, Fourier pairs for each
       period in ``periods`` (accepts non-integers, such as 365.25) and ``ar_lags``
       lags; frozen coefficients. Represents series with several seasonalities (weekly +
       yearly in daily data; daily + weekly in hourly data) without one state per
       seasonal position. Default for daily data: ``periods=(7, 365.25)``,
       ``harmonics=(3, 5)``, ``ar_lags=1``
     - core
   * - ``ridge_regression``, ``lasso_regression``, ``elastic_net_regression``
     - linear
     - ``sklearn.linear_model.Ridge`` / ``Lasso`` / ``ElasticNet`` on lags
     - core (scikit-learn)
   * - ``extra_trees_regressor``
     - tree
     - ``sklearn.ensemble.ExtraTreesRegressor`` on lags
     - core (scikit-learn)
   * - ``hist_gradient_boosting``
     - tree
     - ``sklearn.ensemble.HistGradientBoostingRegressor`` on lags
     - core (scikit-learn)
   * - ``adaboost_r2``
     - tree
     - ``sklearn.ensemble.AdaBoostRegressor`` (AdaBoost.R2) on lags
     - core (scikit-learn)
   * - ``lightgbm_regressor``
     - tree
     - ``lightgbm.LGBMRegressor`` on lags (200 trees, 15 leaves, ``n_jobs=1``)
     - ``boosting`` extra (lightgbm)
   * - ``xgboost_regressor``
     - tree
     - ``xgboost.XGBRegressor`` on lags (200 trees, depth 4, ``n_jobs=1``)
     - ``boosting`` extra (xgboost)
   * - ``catboost_regressor``
     - tree
     - ``catboost.CatBoostRegressor`` on lags (200 iterations, depth 4,
       ``thread_count=1``, without writing ``catboost_info/``)
     - ``boosting`` extra (catboost)

StatsForecast's automatic models (``sf_auto_arima``, ``sf_auto_ets``,
``sf_auto_theta``, ``sf_auto_ces``; ``nixtla-stats`` extra, which brings pandas 2.x)
have their own page: :doc:`nixtla`.

The boosting models run with one thread per model: on series of a few hundred points,
OpenMP threads cost more than they save. The experiment's parallelism still happens
across candidates (``n_jobs``).

scikit-learn models become series forecasters through the same adapter
(``LagRegressionForecaster``): they learn ``y[t]`` from ``y[t-1], …, y[t-lags]``, with
optional differencing and standardization fitted on the training data only, and
forecast several steps recursively. The estimator's hyperparameters go in
``estimator_params``:

.. code-block:: python

   exp.create_model("knn", lags=14, estimator_params={"n_neighbors": 3})

Standardization (``scale``) is fitted on the training data: ``True`` (default) uses one
mean and one deviation of the series; ``"columns"`` standardizes each lag and the target
separately, like scikit-learn's ``StandardScaler`` on X and y; ``False`` does not
standardize. Linear models and trees do not change with the choice; KNN and SVR do,
because they depend on distances. Correctors accept the same options. With
``scale="columns"``, AutoHybTS exactly reproduces the four KNN-corrector hybrids of the
replication project.

Intermittent demand
-------------------

For series with many zeros (sparse sales, spare parts), four methods make up the
``intermittent`` family. They are base models, not combinations of candidates:

.. list-table::
   :header-rows: 1
   :widths: 16 56 28

   * - Name
     - Forecast (constant over the horizon)
     - Parameters
   * - ``croston_sba``
     - Croston with the Syntetos-Boylan correction: ``(1 − α/2) · z / p``, with ``z``
       the demand size and ``p`` the interval between demands, smoothed separately
     - ``alpha=0.1``
   * - ``tsb``
     - Teunter-Syntetos-Babai: demand probability (smoothed every period) × size
       (smoothed when there is demand); decays when demand stops
     - ``alpha_d=0.1``, ``alpha_p=0.1``
   * - ``adida``
     - sum in blocks of ``k`` periods (``k`` = mean interval between demands in the
       training data, frozen), smoothing of the blocks and division by ``k``
     - ``alpha=0.1``
   * - ``imapa``
     - mean of the ADIDA forecasts at levels ``1..k``
     - ``alpha=0.1``

All of them are implemented in numpy, run in variants A and B (the aggregation level is
learned on the training data; the recursions read the context) and never forecast
negative values. A series with negative values is not demand: fitting is refused and
the trial is ``SKIPPED``. The diagnostics of each trial include the profile of the
series in the training data: fraction of zeros, average demand interval (ADI) and the
CV² of the sizes.

The family joins the selection chain only when one of these models is in
``base_models``, right before the residual hybrids; without them, the default protocol
does not change. For intermittent demand, prefer ``metrics=["RMSSE", ...]``: metrics
based on absolute error (MASE, MAE) favour forecasting zero.

.. code-block:: python

   y = get_data("intermittent_daily")
   exp = AutoHybridExperiment(
       fh=14, metrics=["RMSSE", "MASE"], base_models=["naive", "croston_sba", "tsb", "imapa"]
   ).fit(y)

Residual correctors
-------------------

.. list-table::
   :header-rows: 1
   :widths: 16 52 32

   * - Name
     - Implementation
     - Installation
   * - ``linear``
     - ``sklearn.linear_model.LinearRegression`` on residual windows
     - core (scikit-learn)
   * - ``knn``
     - ``sklearn.neighbors.KNeighborsRegressor`` on residual windows
     - core (scikit-learn)
   * - ``mlp``
     - ``sklearn.neural_network.MLPRegressor`` on residual windows
     - core (scikit-learn)

Optional correctors (outside the default grid; use ``correctors=[...]`` or
``create_hybrid``): ``svm``, ``tree``, ``random_forest``, ``hist_gradient_boosting``,
``adaboost_r2``, ``lightgbm_regressor``, ``xgboost_regressor`` and
``catboost_regressor`` (the last three in the ``boosting`` extra), with the same
implementations as the base models, now on residual windows.

Residual sequence correctors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A window corrector regresses the next residual on the previous ``window`` ones and is
iterated step by step. A **sequence** corrector is a time series model of the residual
series itself: it is fitted once on the contiguous out-of-sample residuals of the
training data and forecasts the residual vector for the whole horizon from the residual
history available at the origin. State models (ARIMA, exponential smoothing) need the
full residual history for their states, which a fixed window does not provide.

.. list-table::
   :header-rows: 1
   :widths: 30 46 24

   * - Name
     - Model on the residuals
     - Installation
   * - ``arima_sequence``
     - ``arima`` (automatic order), frozen parameters
     - ``stats`` extra
   * - ``exponential_smoothing_sequence``
     - ``exponential_smoothing`` without trend (simple smoothing)
     - ``stats`` extra

.. code-block:: python

   exp.create_hybrid("seasonal_naive", "arima_sequence")
   # parameters of the model on the residuals:
   exp.create_hybrid("linear", "arima_sequence",
                     corrector_params={"model_params": {"order": (1, 0, 0)}})

They work in the additive hybrid (and in stages), in variants A and B: in B, new
observations extend the residual series without refitting. The window does not apply to
them: in the search, each one is proposed with the first window of
``residual_windows`` only. Strategies that require a window corrector (decomposition,
corrector combination, dynamic selection) do not propose them, and a hand-built
candidate using them is ``SKIPPED``.

Capabilities
------------

``exp.models()`` shows what each model declares. Besides ``multi_step`` and
``online_state_update`` (variants A and B), ``training_scope`` (``local``,
``global``, ``pretrained``), ``output_type`` (``point``, ``quantile``, ``samples``,
``distribution``), ``target_mode`` (``univariate``, ``multivariate``, ``exogenous``),
``horizon_strategy`` and ``max_horizon`` describe the model. The search is univariate,
local, point-valued and without covariates: a model trained globally on a panel, with a
non-point output, that forecasts several targets together, that needs covariates or
whose maximum horizon is shorter than requested is ``SKIPPED`` without being fitted.

Metrics
-------

MASE (default for ranking), MAE, RMSE, sMAPE, WAPE, RRMSE, RMAE (``MAE / mean of the
observations``, as in the replication protocol) and RMSSE (root mean squared error
scaled by the squared seasonal difference of the training data, with the same ``m`` as
MASE; M5 uses ``m = 1``). Choose with ``metrics=[...]``; the first one ranks and
selects.

Other dependencies
------------------

Parts outside the models also use external libraries, always imported only when used
(``import autohybts`` loads only numpy, pandas and the package itself):

.. list-table::
   :header-rows: 1
   :widths: 34 40 26

   * - Feature
     - Library
     - Installation
   * - ``stl(...)`` decomposition
     - ``statsmodels.tsa.seasonal.STL``
     - ``stats`` extra
   * - statistical tests in comparison and selection
     - ``scipy.stats`` (comes with scikit-learn)
     - core
   * - ``tune_model(sampler="optuna")``
     - ``optuna`` (TPE); with ``sampler="auto"`` and Optuna not installed, the search is
       random with a seed
     - ``optuna`` extra
   * - plots
     - ``matplotlib``
     - ``plot`` extra
   * - reading M3
     - ``rdata``
     - ``datasets`` extra

Without the required extra, the model or feature fails with
``UnsupportedCapabilityError`` saying what to install; in the automatic search, the
candidate becomes ``SKIPPED`` and the others go on. The origin of the *data*
(Rdatasets, M4, M3) is in :doc:`datasets`.

Your own models
---------------

Models from PyTorch, TensorFlow/Keras or any other library enter through registration
(``exp.register_model``) and then work across the whole experiment: see
:doc:`custom_models`.
