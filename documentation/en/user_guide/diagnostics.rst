Series diagnostics
==================

Before choosing models, it is worth knowing whether the series has a regular cycle.
Seasonality decides which models make sense: ``seasonal_naive``, seasonal
``exponential_smoothing``, the ``stl`` decomposition and ``sarima`` only help when the
cycle exists. This page shows the exploratory diagnostic and how the search can use it
without leakage.

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
