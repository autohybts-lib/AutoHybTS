Series diagnostics
==================

Before choosing models, it is worth knowing whether the series has a regular cycle.
Seasonality decides which models make sense: ``seasonal_naive``, seasonal
``exponential_smoothing``, the ``stl`` decomposition and ``sarima`` only help when the
cycle exists. This page shows the exploratory diagnostic and how the search can use it
without leakage.
The notebook :doc:`../tutorials/notebooks/16_series_diagnostics` goes through all of
it with figures.

Seasonality
-----------

.. code-block:: python

   exp = AutoHybridExperiment(fh=14).fit(y)
   exp.seasonality_report()              # the experiment's period
   exp.seasonality_report(periods=[7, 30])

One row per series and candidate period, computed on the development history (never on
the sealed test):

``n_cycles``
   complete cycles available. With fewer than 3, there is no test.
``strength``
   seasonal strength, ``F_s = max(0, 1 − Var(R)/Var(S + R))``, from a classical
   decomposition (centred moving average of one cycle for the trend, means by position
   for the seasonal part; Wang, Smith and Hyndman, 2006). Near 1: a dominant cycle; near
   0: none.
``autocorrelation``
   autocorrelation of the detrended series at the period lag.
``p_value``
   Friedman test of the detrended series: each cycle is a block and each position in
   the cycle a treatment, so slow changes between cycles do not count as seasonality.
``seasonal``
   the rule's decision: ``p_value < 0.05`` with at least 3 cycles.
``seasonal_differences``
   suggested seasonal differences: 1 when there is seasonality and ``strength ≥ 0.64``
   (the rule of ``nsdiffs`` in the R package ``forecast``).

On the library's synthetic series (730 days, period 7):

.. list-table::
   :header-rows: 1
   :widths: 28 14 14 18 14 12

   * - Series
     - ``n_cycles``
     - ``strength``
     - ``p_value``
     - ``seasonal``
     - ``D``
   * - ``synthetic_daily``
     - 102.3
     - 0.87
     - 1.6 × 10⁻¹¹⁴
     - yes
     - 1
   * - ``white_noise_daily``
     - 102.3
     - 0.01
     - 0.81
     - no
     - 0

On 200 simulated series per scenario without seasonality, the rule declared a cycle in
2–9 % of the series with independent noise (with or without a linear trend) and in up to
11 % with persistent noise (AR(0.8) or random walk), against the nominal 5 %. With a
cycle whose amplitude equals the noise standard deviation, it detected 68–100 % of the
series from 6 cycles on and almost none with 3. Power depends on the number of cycles:
with few, not rejecting is no evidence that the cycle is absent.

Seasonality in the search: ``seasonal_period="auto"``
-----------------------------------------------------

By default, the period comes from the frequency (daily: 7, monthly: 12, yearly: 1) and
seasonal models always receive it. With ``seasonal_period="auto"``, the search tests
first:

.. code-block:: python

   exp = AutoHybridExperiment(fh=14, seasonal_period="auto").fit(y)
   exp.seasonal_period_    # 7: period of the metrics (MASE scale), always the frequency's
   exp.search_period_      # 7 with a cycle; 1 without evidence of a cycle
   exp.seasonality_        # the test that decided

The test uses only the training data of the **first** validation origin: data every
origin has already seen, so the decision depends on no evaluated target. If that
training data has at least 6 cycles and the test shows no cycle, ``search_period_`` is
1: ``seasonal_naive``, ``sarima`` and the STL are ``SKIPPED`` with the reason in the
ledger, and the other models get their non-seasonal version. With fewer than 6 cycles
the test has almost no power, and the frequency's period stays (column ``keep_period``
of ``seasonality_``). The MASE scale stays the frequency's, so errors are comparable with
and without ``"auto"``. In a panel, the cycle holds when at least half of the series
show it.

With the data above, ``synthetic_daily`` kept period 7 (p ≈ 3 × 10⁻⁸² on the first
origin's training data) and ``white_noise_daily`` moved to 1 (p = 0.59).

.. warning::

   The test is decided on the first origin's training data, which can be short. On
   ``air_passengers`` (monthly, ``fh=12``), that training data has only 3 cycles: the
   test gave p = 0.039 and period 12 would stay anyway, by the 6-cycle rule. On short
   series, ``"auto"`` almost always keeps the frequency's period; prefer an explicit
   ``seasonal_period`` chosen from domain knowledge.

Characterization (rules of the replication project)
---------------------------------------------------

``exp.characterize()`` applies the two rules of the replication project to the
development history, with the same result (parity test against the legacy code):

1. **Stationarity (ADF)**, with a constant and lags by AIC: the series is stationary when
   ``p ≤ 0.05`` and the statistic is below the 5 % critical value. If it is not,
   ``diff_order`` is the smallest order up to 2 that passes. The KPSS of each order is
   reported too, without entering the decision.
2. **Seasonality**, for non-stationary series only: strength ``F_S`` of a robust STL on
   the Box-Cox transformed series (λ by maximum likelihood between −1 and 2), for each
   candidate period with at least two cycles. The strongest is the dominant period, and
   the series is seasonal when ``F_S > 0.5``.

.. code-block:: python

   exp.characterize()                  # the experiment's period
   exp.characterize(periods=[7, 365])  # daily: week and year, as in the replication

The category sums up both rules: ``stationary``, ``nonseasonal`` or ``seasonal``. On the
synthetic series (716-day history):

.. list-table::
   :header-rows: 1
   :widths: 24 14 14 14 12 12 14

   * - Series
     - ADF (level)
     - ADF p d=0
     - ADF p d=1
     - ``diff_order``
     - ``fs_7``
     - category
   * - ``synthetic_daily``
     - −0.75
     - 0.83
     - < 0.001
     - 1
     - 0.84
     - ``seasonal``
   * - ``white_noise_daily``
     - −25.0
     - < 0.001
     - —
     - 0
     - —
     - ``stationary``

With ``seasonal_period="auto"``, ``seasonality_test="strength"`` replaces the Friedman
test with rule 2: a series that ADF finds stationary has no seasonality, and the others
need ``F_S > 0.5``. Without two cycles the strength is undefined and the frequency's
period stays.

Plots
-----

All of them use the development history, never the sealed test (``plot`` and ``stats``
extras):

``exp.plot_data("stationarity")``
   the series in level, with one and with two differences, with the ADF and KPSS
   p-values in each panel and the order chosen by ADF highlighted.
``exp.plot_data("acf_pacf")``
   ACF and PACF at the chosen order, with the ±1.96/√n band. The PACF is Yule-Walker
   with biased autocovariances (statsmodels' ``method="ywm"``).
``exp.plot_data("decomposition")``
   trend, seasonal and remainder of the robust STL of the Box-Cox series, at the
   experiment's period.
``exp.plot_characterization()``
   ADF statistic against ``F_S``, one point per series, coloured by category; useful in
   a panel.

Stationarity: ADF and KPSS
--------------------------

``exp.stationarity_report()`` tests the level and the differences up to ``max_d``
(default 2) with both tests, whose null hypotheses are opposite: ADF rejects a unit root
when ``p ≤ 0.05`` and the statistic is below the 5 % critical value; KPSS says stationary
when it does not reject (``p ≥ 0.05``). ``consensus`` marks the orders where both agree,
and ``attrs`` holds each rule's smallest order (``d_adf``, ``d_kpss``, ``d_consensus``).
On ``synthetic_daily``:

.. code-block:: text

   d   adf_stat  adf_pvalue  adf_stationary  kpss_pvalue  kpss_stationary  consensus
   0     -0.749       0.834           False         0.01            False      False
   1    -11.128     < 0.001            True         0.10             True       True
   2    -12.048     < 0.001            True         0.10             True       True

   {'d_adf': 1, 'd_kpss': 1, 'd_consensus': 1}

KPSS reports p-values only between 0.01 and 0.10 (beyond that, the table's bound); with
few observations, both tests have little power.

Decomposition as a table
------------------------

``exp.decompose("stl")`` (robust STL) or ``exp.decompose("classical")`` (centred moving
average and mean seasonal indices) return ``observed``, ``trend``, ``seasonal`` and
``resid`` indexed by date, at the experiment's period (or ``period=``). In the classical
one, the trend is ``NaN`` at the edges. It is exploratory: the ``decomposition`` strategy
redoes the decomposition on each origin's training data.

Residuals of a base: Ljung-Box
------------------------------

``exp.residual_diagnostics("arima")`` fits the base on the first half of the development
history and forecasts the other half one step at a time, with frozen parameters: these
are the out-of-sample residuals correctors learn from. The Ljung-Box test says whether
autocorrelation is left in them; a small ``p`` signals room for a residual corrector.
Default lags: ``min(10, n/5)`` and twice the seasonal period. On ``synthetic_daily`` (358
residuals):

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Base
     - p (lag 10)
     - p (lag 14)
   * - ``naive``
     - < 0.001
     - < 0.001
   * - ``seasonal_naive``
     - < 0.001
     - < 0.001
   * - ``arima``
     - 0.070
     - 0.110

The residuals of ``naive`` and ``seasonal_naive`` carry the structure the base did not
model; those of ``arima``, almost none. A test without rejection does not guarantee that
a corrector is useless: the acceptance rule decides on the validation origins.

Automatic differences
---------------------

Lag models (linear, KNN, SVR, trees, MLP, boosting and registered networks) learn better
on a series without trend or unit root. With ``differencing="auto"``, each origin
chooses, on its own training data:

1. ``D``: one seasonal difference (``y[t] − y[t − m]``) when the test above finds a
   significant cycle with ``strength ≥ 0.64``;
2. ``d``: the smallest order from 0 to 2 at which the series (already seasonally
   differenced) passes the ``differencing_test``:

   * ``"kpss"`` (default): KPSS does not reject stationarity at 5 %, like ``ndiffs`` in
     ``auto.arima`` and this library's ``arima``;
   * ``"adf"``: ADF rejects a unit root (``p ≤ 0.05`` and the statistic below the 5 %
     critical value), the rule of the replication project.

The chosen orders stay frozen: the forecast integrates back from the values observed in
the context, without redoing the tests (variant B included). The tests come from
statsmodels (``stats`` extra); without it, the candidate is ``SKIPPED`` with the reason.

.. code-block:: python

   # one model
   exp.create_model("knn", differencing="auto", season_length=7)
   exp.create_model("linear", differencing=1, seasonal_differencing=1, season_length=12)

   # every built-in lag model of the search
   exp = AutoHybridExperiment(fh=14, differencing="auto", differencing_test="adf")

The experiment option applies to the built-in models of the search (``compare_models``)
and uses the search period (``search_period_``); ``create_model`` uses only the
parameters you pass, and a registered network gets ``differencing="auto"`` at
registration (``torch_model(LSTMNet, differencing="auto")``). The
default (``differencing=None``) does not difference, and candidate IDs do not change.

Optional preprocessing: Box-Cox and outliers
--------------------------------------------

Off by default; the user turns them on per model or for every lag model of the search.
Everything is fitted on each origin's training data, and the metric targets remain the
observed values.

``transform="boxcox"`` or ``"log"``
   λ by maximum likelihood on the training data (with a shift when there are values
   ≤ 0), frozen; the forecast context goes through the same transform and the forecast
   returns to the original scale. Applied before the differences.
``outliers="winsorize"`` or ``"interpolate"``
   points with a robust z above 4 (in absolute value) against a rolling median are clipped at the threshold or
   replaced by interpolation, only in the values the model learns from. The forecast
   context is not changed.

.. code-block:: python

   exp.create_model("linear", lags=12, transform="boxcox", differencing="auto", season_length=12)
   AutoHybridExperiment(fh=12, transform="boxcox", outliers="winsorize")   # the whole search

On ``air_passengers`` (monthly, 4 origins), the linear model on 12 lags had MASE 0.822;
with ``transform="log"``, 0.881; with ``"boxcox"``, 0.927; with Box-Cox and
``differencing="auto"``, 0.773. The transform alone did not help: combined with the
differences, it did. Use it as one more option to validate, not as a default.

Seasonal ARIMA
--------------

``sarima`` (``stats`` extra) is statsmodels' seasonal ARIMA with automatic choices on the
training data of each origin, like ``auto.arima`` (Hyndman and Khandakar, 2008):

1. ``D`` (0 or 1) by the rule above: a significant cycle and ``strength ≥ 0.64``;
2. ``d`` (up to 2) by successive KPSS tests on the seasonally differenced series;
3. ``(p, q, P, Q)`` by AICc in a stepwise search from four starting models, with
   ``p, q ≤ 2`` and ``P, Q ≤ 1``.

.. code-block:: python

   exp = AutoHybridExperiment(fh=12, base_models=["seasonal_naive", "arima", "sarima"])
   exp.create_model("sarima", season_length=12)
   exp.create_model("sarima", season_length=12, order=(1, 1, 0), seasonal_order=(0, 1, 1))

Orders and parameters stay frozen after fitting: a longer context (variant B,
out-of-sample residuals) only updates the Kalman filter state. ``sarima`` is outside the
default search (it enters through ``base_models``) and refuses periods above 24, too slow
for a search; for long cycles use ``harmonic_regression``. ``arima`` remains
non-seasonal.

Measured on 13 October 2026 with ``create_model`` and the experiment defaults (mean MASE
over the validation origins and total time):

.. list-table::
   :header-rows: 1
   :widths: 28 18 18 18 18

   * - Series
     - ``seasonal_naive``
     - ``arima``
     - ``exponential_smoothing``
     - ``sarima``
   * - ``air_passengers`` (8 origins)
     - 1.27 · 0 s
     - 1.30 · 6 s
     - 0.68 · 0.5 s
     - 0.60 · 12 s
   * - ``synthetic_daily`` (14 origins)
     - 1.13 · 0 s
     - 1.13 · 26 s
     - 1.06 · 1.5 s
     - 0.76 · 310 s

The gain came at a cost: on a daily series of 730 points, each origin took about 22 s.
That is why ``sarima`` stays out of the default search until it is measured on more
series.
