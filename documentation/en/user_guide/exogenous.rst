Exogenous variables
===================

A series may depend on others: electricity demand on temperature and working days, sales on
a leading indicator, road deaths on a law. With ``fit(y, X)`` the experiment tries models
with those covariates — ARIMAX, SARIMAX, the lag models and the hybrids built on them — and
the selection decides, under the same acceptance rule, whether they enter.

.. code-block:: python

   from autohybts import AutoHybridExperiment
   from autohybts.datasets import load_exog_dataset

   y, X = load_exog_dataset("bjsales")             # sales and a leading indicator
   exp = AutoHybridExperiment(fh=3, exog_selection="auto").fit(y, X)
   cmp = exp.compare_models()                      # "exogenous" family included
   cmp.decisions                                   # accepted? with what gain?
   exp.compare_exog().tests()                      # Wilcoxon and Diebold–Mariano
   final = exp.finalize_model(cmp.best)
   exp.evaluate_external(final)

``X`` is a ``DataFrame`` indexed by date, with every date of ``y`` (the sealed test's
included) and, to forecast beyond the data, the next dates of the ``future`` columns.

What each column may see
------------------------

The risk of covariates is leakage: using at the forecast date a value only known later. So
each column has a kind.

.. list-table::
   :header-rows: 1
   :widths: 14 40 46

   * - Kind
     - When to use
     - How it enters
   * - ``past`` (default)
     - values observed with the series: prices, indicators, measured temperature
     - lagged by at least the steps forecast from one origin: ``max(fh, test_size)`` in
       variant A, 1 in B. The value used at date ``t`` is ``x[t - lag]``, already known at
       the origin.
   * - ``future``
     - values known in advance: calendar, holidays, planned price or promotion, a law
       already passed
     - at the forecast date itself (lag 0)

Declare ``future`` columns in ``exog_future=[...]``; extra lags go in
``exog_lags={"column": [7, 14]}``. A ``past`` column with a lag below the minimum is refused.
The temporal context and the finalized model refuse any row beyond what was known at the
origin (``TemporalLeakageError``).

.. warning::

   Declaring as ``future`` a value only observed later — the day's temperature, tomorrow's
   price — turns the evaluation into an **oracle**: the measured error is that of someone
   who knew the future. Use ``future`` that way only to measure the potential of a good
   forecast of the covariate; in production, the column must come from that forecast.

Choosing the variables automatically
------------------------------------

``exog_selection`` sets the feature sets the search tries:

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Value
     - Sets
   * - ``None`` (default)
     - every column at its smallest causal lag
   * - ``"ccf"``
     - prewhitened cross-correlation (Box–Jenkins), with Holm
   * - ``"granger"``
     - Granger F test per column, with Holm; the lag with the largest ``|t|``
   * - ``"lasso"``
     - Lasso with the penalty of the one-standard-error rule on a temporal validation
   * - ``"stepwise_bic"``
     - forward selection by BIC
   * - ``"mutual_info"``
     - mutual information against 30 permutations (null of the maximum over all features)
   * - ``"vote"``
     - columns chosen by most of the five methods
   * - ``"auto"``
     - one set per method and the set with all columns; the validation decides
   * - list of methods
     - one set per method in the list

The methods use only the data before the first ``V_select`` origin, as hyper-parameter
tuning uses ``V_search``; ``V_select`` and the sealed test never reach them.
``exp.exog_report()`` shows, per method, each candidate feature, its score, its p-value
(when there is one) and whether it was chosen.

Measured on synthetic series with one relevant covariate (lag 7), a holiday and three noise
columns (10 series with a relation and 10 without, 300 points):

.. list-table::
   :header-rows: 1
   :widths: 20 26 26 28

   * - Method
     - noise chosen (series without a relation, AR noise / random walk)
     - relevant found (of 20)
     - extras on series with a relation
   * - ``vote``
     - 0 / 0
     - 17 / 19
     - 0 / 0
   * - ``granger``
     - 0 / 1
     - 17 / 17
     - 0 / 0
   * - ``ccf``
     - 0 / 0
     - 10 / 16
     - 1 / 0
   * - ``mutual_info``
     - 1 / 0
     - 10 / 10
     - 0 / 0
   * - ``lasso``
     - 0 / 0
     - 20 / 20
     - 12 / 16
   * - ``stepwise_bic``
     - 3 / 10
     - 20 / 20
     - 17 / 14

``lasso`` and ``stepwise_bic`` find everything but bring neighbouring lags along; ``vote`` is
the best balance. Under ``"auto"`` this matters little: a bad set loses in validation.

With × without covariates
-------------------------

The decision is the incremental selection's: the ``exogenous`` family challenges the
univariate champion after the residual structures and enters only if it passes the
calibrated acceptance rule (median gain ≥ 5 %, wins ≥ 60 %). A model with covariates is
never the initial champion.

``exp.compare_exog()`` compares the best finalist with and the best without covariates on the
same ``V_select`` units: ``.summary``, ``.tests()`` (Wilcoxon and Diebold–Mariano, with Holm)
and ``.plot()``. The tests are reporting; with 7 units, the smallest possible Wilcoxon
p-value is 0.016. For the sealed test, finalize both and use ``compare_external``.

Results on the benchmarks (horizon and selection of notebook 20):

.. list-table::
   :header-rows: 1
   :widths: 30 14 14 14 28

   * - Dataset
     - accepted
     - median gain
     - p (Wilcoxon)
     - test MASE (without → with)
   * - ``bjsales`` (fh = 3, leading indicator)
     - yes
     - 69 %
     - 0.016
     - 0.110 → 0.029
   * - ``elecdaily`` with ``past`` temperature
     - no
     - −4 %
     - 0.58
     - 1.954 → 1.954
   * - ``elecdaily`` with ``future`` temperature (oracle)
     - yes
     - 28 %
     - 0.031
     - 1.954 → 0.689
   * - ``seatbelts``, ``uschange`` (fh = 12 and 8)
     - no
     - < 0
     - > 0.1
     - same

``past`` covariates lagged by the whole horizon rarely help at long horizons; an indicator
that leads the series by the horizon, or covariates truly known in advance, do.

Forecasting, updating and serving
---------------------------------

.. code-block:: python

   final = exp.finalize_model(cmp.best)
   exp.predict_model(final)                    # future columns from the fit's X...
   final.predict(3, X_future=new_rows)         # ...or from X_future

   newer = final.update_state(y_new, X_new=X_new)    # no refit
   stream = newer.stream()                            # variant B
   ticket = stream.predict_next(exog={"feriado": 1.0})
   stream.observe(value, exog={"indicador": 12.3})

The finalized model keeps the covariates up to its last observation; ``save_model`` and
``load_model`` preserve them. A horizon longer than the lag of the ``past`` columns is
refused.

Datasets with covariates
------------------------

``list_exog_datasets()`` lists and ``load_exog_dataset(name)`` returns ``(y, X)``, downloaded
from Rdatasets the first time:

.. list-table::
   :header-rows: 1
   :widths: 18 58 24

   * - Name
     - Series
     - ``future`` columns
   * - ``seatbelts``
     - drivers killed or injured in the UK ~ km driven, petrol price, seat belt law
       (monthly, 1969–1984)
     - ``lei_cinto``
   * - ``bjsales``
     - sales ~ leading indicator (Box–Jenkins)
     - —
   * - ``uschange``
     - US consumption ~ income, production, savings, unemployment (quarterly)
     - —
   * - ``insurance``
     - insurance quotes ~ TV advertising (monthly, 40 points)
     - ``publicidade_tv``
   * - ``elecdaily``
     - daily electricity demand in Victoria ~ temperature (and its square) and working day,
       2014
     - ``dia_util``
   * - ``vic_elec_daily``
     - daily demand in Victoria in GWh ~ maximum temperature (and its square), holiday and
       weekend, 2012–2014
     - ``feriado``, ``fim_de_semana``

Limits
------

* One series per experiment: panels with covariates come later.
* Decomposition, combination, ensemble and dynamic selection do not take covariates.
* In hybrids, with covariates only the ``historical_oof_1step`` residual source is used (the
  multi-step one would read ``past`` columns beyond their lag).
* In lag models, covariates enter as given (differencing applies to the series only).

.. seealso::

   Tutorial: :doc:`../tutorials/notebooks/20_exogenous`.
