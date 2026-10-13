Search space and budget
=======================

``compare_models`` proposes candidates in stages, evaluates each one on every origin of
``V_search`` and takes the best of each family to the selection on ``V_select``
(:doc:`selection`). This page says what is proposed, how much the budget lets through
and how to widen or narrow the space. The models and their parameters are in
:doc:`model_reference`.

Stages
------

``S0`` — base models
   each base in ``base_models`` with the search parameters, or each item of its
   ``base_params`` grid.
``S1`` — retention
   the two best configurations of each family, ``arima`` and the best standalone model
   go on as hybrid bases (without duplicates).
``S2`` — residual hybrids
   each retained base × corrector × window × α × residual source, breadth first: every
   base gets the first combination before the second, so a budget cut keeps the bases
   varied.
``S3`` — refinement
   around the two best residual structures, one axis at a time: another residual
   source, the ``corrector_params`` alternatives and, with ``"multistage"`` in
   ``include``, extra stages.
``S4`` — advanced families
   only with ``include``: decomposition, combination, dynamic selection (variant B) and
   ensembles, on the leftover budget, with one reserved slot per family.

Defaults
--------

.. include:: _generated/search_defaults.rst

Budget
------

The budget counts **candidates**: a candidate is evaluated on every origin of
``V_search``. The limits apply to the ordered list of proposals before each stage, so
the same series and the same seed give the same search on any machine, with
``n_jobs`` sequential or parallel. Proposals that are cut show up in the ledger as
``PRUNED_BUDGET``.

.. include:: _generated/budgets.rst

``max_trials`` replaces the total; ``max_wall_time_s`` adds an optional wall-clock
deadline (:doc:`strategies`). The number of origins in each layer is ``fold``, not the
budget.

Example: ``smoke`` and ``small``
--------------------------------

Series ``synthetic_daily`` (730 days), ``fh=14``, ``session_id=42``, other defaults
(``fold="auto"`` used 7 origins per layer). Ledger count per stage and status:

.. list-table::
   :header-rows: 1
   :widths: 14 12 16 18 20 20

   * - Budget
     - Stage
     - Evaluated
     - Cut by the budget
     - Skipped for capability
     - Proposed
   * - ``smoke``
     - S0
     - 6
     - 3
     - 0
     - 9
   * -
     - S2
     - 2
     - 70
     - 0
     - 72
   * -
     - S3
     - 2
     - 0
     - 0
     - 2
   * - ``small``
     - S0
     - 9
     - 0
     - 0
     - 9
   * -
     - S2
     - 11
     - 85
     - 0
     - 96
   * -
     - S3
     - 2
     - 0
     - 2
     - 4

* ``smoke``: 10 candidates, 70 units (candidate × origin), 140 fits, 22 s; only 6 of
  the 9 bases fit within the per-stage limit. Champion: ``linear``.
* ``small``: 22 candidates, 154 units, 350 fits, about 5 min on this machine (with
  another load in parallel). All bases and 11 of the 96 possible hybrids were
  evaluated; the **per-family** limit (15) stopped the hybrids before the total (50).
  Champion: ``random_forest``, accepted against ``linear`` with a median gain of 12 %
  and 6 wins out of 7.
* The 2 proposals skipped in S3 were the ``in_sample`` source with bases that do not
  expose fitted values.

S2 proposes 96 hybrids: 8 retained bases × 3 correctors × 2 windows × 2 values of α.
To evaluate more of them, increase the budget or reduce the space:

.. code-block:: python

   exp = AutoHybridExperiment(
       fh=14,
       budget="medium",                 # 200 candidates, 40 per family
       base_models=["seasonal_naive", "arima", "linear", "knn"],
       base_params={"knn": [{"lags": 7}, {"lags": 14}, {"lags": 28}]},
       correctors=["linear", "knn"],
       corrector_params={"knn": [{"estimator_params": {"n_neighbors": 3}}]},
       residual_windows=[7, 14],
       alphas=[1.0],
   )
   result = exp.compare_models()
   result.search.ledger_frame()        # every proposal, evaluated or not, and why

Cost
----

The cost of a search is proportional to candidates × origins × fits per candidate. With
``fold="auto"`` (7 origins per layer when the series allows, ADR-0008), each candidate
is evaluated on more origins than with the old default of 3: in the example above,
``small`` took about 3.5 times as long. A hybrid fits the base twice per origin
(training and residual prefix) plus the corrector. For a quick exploration,
``fold=3`` and ``budget="smoke"`` are still available, with the less reliable
acceptance rule that implies (:doc:`selection`).
