Reading the results
===================

``compare_models`` returns a ``CompareResult``. This page says what each table holds and
how to read a decision.

Leaderboard
-----------

``result.leaderboard``: one row per candidate evaluated in the search (``V_search``),
sorted by the first metric (lower is better).

``name``, ``model``
   short name of the structure (``arima-mlp``) and full label, with window, α and
   residual source.
``family``, ``strategy``, ``base``, ``corrector``, ``window``, ``alpha``, ``residual_source``
   the configuration.
``mean_rank``
   rank of the candidate by the sort metric (ties get the average rank).
metrics (``MASE``, ``MAE`` …)
   means over the valid units.
``coverage``, ``n_units``, ``n_failed``, ``n_skipped``
   share of units with a valid metric, expected units, and units that failed or were
   skipped.
``fit_seconds``, ``predict_seconds``
   cost summed over the origins.
``is_champion``
   marks the final champion.

``result.selection_leaderboard`` has the same columns for the finalists on
``V_select``.

Decisions
---------

``result.decisions``: one row per family that challenged the champion, in chain order.

``champion_before``, ``challenger``
   the ``candidate_id`` of the champion at that moment and of the challenger.
``accepted``
   whether the challenger became the champion.
``median_gain``
   paired median gain, ``(E_champion − E_challenger) / |E_champion|``, on the same units;
   positive when the challenger errs less.
``win_rate``
   share of units where the challenger erred less (ties do not count).
``variability_status``, ``variability_value``
   the corrector variability criterion: ``passed``, ``failed``, ``undefined``,
   ``not_applicable`` (challenger without a corrector) or ``disabled`` (``quickstart``
   profile).
``p_value``
   the Wilcoxon test, only with ``significance_alpha``.
``reason_codes``
   why the decision was made:

   .. list-table::
      :header-rows: 1
      :widths: 34 66

      * - Code
        - Meaning
      * - ``accepted``
        - every criterion passed
      * - ``median_gain_below_threshold``
        - median gain below 5 %
      * - ``win_rate_below_threshold``
        - wins below 60 % of the units
      * - ``coverage_incomplete``
        - the challenger has no valid metric on every unit
      * - ``different_units`` / ``no_comparable_units``
        - the units do not match or none is comparable
      * - ``variability_below_threshold``
        - the corrector barely moves the forecast (``correction_std_ratio`` < 0.05)
      * - ``variability_undefined``
        - the variability criterion could not be computed
      * - ``not_significant``
        - the Wilcoxon test did not reject (with ``significance_alpha``)

A real example, from notebook 16 (``smoke`` budget, ``fold=3``):

.. code-block:: text

   family    champion_before  challenger    accepted  median_gain  win_rate
   linear    arima            linear        False     -0.400       0.000
   kernel    arima            knn           False     -0.357       0.000
   residual  arima            arima-linear  False      0.028       0.667

``linear`` and ``knn`` lost on every unit. ``arima-linear`` won 67 % of the units, above
the required 60 %, but with a median gain of 2.8 %, below 5 %: rejected for
``median_gain_below_threshold``, and ``arima`` stayed the champion.

``result.selection.decision_units_frame()`` goes down to each unit: both errors, the gain
and the win, to recompute the rule or resample outside the library.

Ledger and audit
----------------

``result.search.ledger_frame()``: every proposal of the search, evaluated or not, with the
stage (``S0``–``S4``), the status and the reason.

``EVALUATED``
   evaluated on every ``V_search`` origin.
``PRUNED_BUDGET``
   cut by the budget before being evaluated.
``SKIPPED_CAPABILITY``
   the model cannot run in that configuration (missing extra, incompatible variant,
   series without seasonality …); the reason says which.
``DUPLICATE``
   same ``candidate_id`` as an earlier proposal.
``STOPPED``
   did not start because the ``max_wall_time_s`` deadline was over.

``result.audit`` sums up how adaptive the selection was (trials, pruning, finalists,
reasons). ``result.audit.selection_gap()`` compares the winner's score on ``V_search``
with its score on ``V_select``: a large gap signals optimism from choosing where it was
searched.

Metrics
-------

Every metric is "lower is better" and is computed per origin, over steps ``1..H``. A
metric undefined on a unit is ``NaN``: the unit becomes invalid for it and lowers the
coverage, instead of entering the ranking with an arbitrary value. Scales below
``1e-12`` count as zero.

.. list-table::
   :header-rows: 1
   :widths: 14 56 30

   * - Metric
     - Formula
     - Undefined when
   * - ``MAE``
     - ``mean(|y − ŷ|)``
     - —
   * - ``RMSE``
     - ``sqrt(mean((y − ŷ)²))``
     - —
   * - ``sMAPE``
     - ``mean(2|y − ŷ| / (|y| + |ŷ|))``, fraction in [0, 2]; 0/0 counts as 0
     - —
   * - ``MASE`` (default)
     - ``MAE / mean(|y_t − y_{t−m}|)`` on the origin's training data, ``m`` = seasonal
       period
     - zero scale or training with at most ``m`` points
   * - ``RMSSE``
     - ``sqrt(mean((y − ŷ)²) / mean((y_t − y_{t−m})²))`` on the origin's training data
     - zero scale or training with at most ``m`` points
   * - ``WAPE``
     - ``sum(|y − ŷ|) / sum(|y|)``
     - ``sum(|y|)`` zero
   * - ``RRMSE``
     - ``RMSE / mean(y)`` over steps ``1..H``
     - ``mean(y)`` zero
   * - ``RMAE``
     - ``MAE / mean(y)`` over steps ``1..H``
     - ``mean(y)`` zero

* The MASE and RMSSE scale comes only from each origin's training data, never from the
  targets.
* For intermittent demand, prefer ``RMSSE``: absolute-error metrics favour forecasting
  zero.
* ``metrics=[...]`` chooses the metrics; the first one ranks and decides.

.. seealso::

   :doc:`selection` (the acceptance rule) and :doc:`search_space` (stages and ledger).
