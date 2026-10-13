Incremental selection
=====================

Selection starts from the best standalone model and visits the families in
``family_order``: statistical, linear, kernel, tree, neural, residual, decomposition,
combination and dynamic selection. Each family's finalist — its best structure in the
search — challenges the **current champion** on ``V_select``.

Some families join the chain only when they are used, and the default protocol does not
change without them:

* ``intermittent`` (``croston_sba``, ``tsb``, ``adida``, ``imapa``), when one of those
  models is in ``base_models``, and the families of models registered with
  ``register_model``, as soon as they are registered: right before ``residual``;
* ``ensemble``, when included in ``compare_models(include=...)``: last, before
  ``dynamic``.

Families without a finalist (not requested in ``include`` or without a valid candidate)
are skipped.

Acceptance rule
---------------

The challenger replaces the champion only if, on the same units (origins), it:

* has full coverage and finite metrics;
* has a paired **median gain** ``(E_champion − E_challenger) / |E_champion|`` ≥ 5 %;
* has a **win rate** ≥ 60 % (ties do not count);
* has **variability** ``correction_std_ratio`` ≥ 0.05 when there is a corrector — a
  corrector that barely moves the forecast is inert. The value is the median per series
  (over the compared units) and then the median across series; a series with no finite
  diagnostic shows up as ``NaN`` in the decision's ``variability_by_series`` and is
  left out of the median (version 2 of the criterion).

Optionally, ``AutoHybridExperiment(significance_alpha=0.05)`` adds a statistical
criterion: the challenger must also win a one-sided paired Wilcoxon test on the same
units, ``p < alpha`` (reason ``not_significant``; the p-value appears in the
``p_value`` column of the decisions). Units with equal errors leave the test; with
``n`` remaining units (all series together) the smallest possible p-value is
``1/2ⁿ``: at least 5 are needed for ``alpha = 0.05``. With ``fold=3`` on one series,
the criterion **never** passes. The test controls each comparison, not the whole
chain: several families challenge a champion that changes with the previous decisions,
and neighbouring origins are not independent.

Calibration of the rule
~~~~~~~~~~~~~~~~~~~~~~~

The thresholds and the number of origins come from a calibration with a known answer
(``autohybts.benchmarks.calibration``): 50 white-noise series, where any
composite structure is a false acceptance, and 50 with an AR(0.8) residual, where the
correction is useful. The target, set before measuring, was false acceptance ≤ 10 % and
useful gain accepted ≥ 50 %, per chain.

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Rule
     - False acceptance
     - Useful gain accepted
   * - gain ≥ 1 %, ``fold=3`` (previous default)
     - 38 %
     - 66 %
   * - gain ≥ 5 %, ``fold=3``
     - 16 %
     - 60 %
   * - gain ≥ 5 %, ``fold=5``
     - 12 %
     - 72 %
   * - **gain ≥ 5 %, ``fold=7``** (default)
     - **2 %** (confirmation: 4 %)
     - **54 %** (confirmation: 56 %)

That is why ``fold="auto"`` (the experiment default) uses 7 origins per layer when the
history allows. On a short series it uses the largest number that fits and warns: the
rule is still more conservative than the previous one, but the measured false
acceptance rate holds only for 7 units. An integer ``fold`` is honoured as given; the
number used is stored in ``exp.fold_``. On a panel, the units of all series enter the
comparison together.

Rejection keeps the champion. Every decision is logged, accepted or not
(``result.selection.decision_units_frame()`` gives, per decision and unit, both errors,
the gain and the win, to recompute or resample outside the library):

.. code-block:: python

   result.decisions
   #   family    champion_before  challenger   accepted  median_gain  win_rate ...
   #   residual  arima            arima-mlp    False     -0.07        0.33

Profiles
--------

``quickstart``
   variability criterion off (and stated explicitly).
``standard``
   full rule (default).
``research_strict``
   full rule and mandatory full coverage.
``exploratory``
   results flagged as exploratory in the audit.

Ranking
-------

In the search and across series, each structure is represented by its best
configuration; structures are ordered by rank (ties get the mean rank), then aggregated
error, coverage, cost and ``candidate_id``. On a panel, ranks are computed per series
and averaged.

.. warning::

   Without ``significance_alpha``, the rule is a protocol, not a hypothesis test: on
   white noise it accepted spurious hybrids for some of the seeds tested. The number of
   origins in ``V_select`` matters a lot: with 2 units, a 60 % win rate requires winning
   both, which happens by chance in about 1 of every 4 comparisons. Measure the
   behaviour of the chain with ``autohybts.benchmarks.calibration`` and use the
   comparison and nested CV to report the uncertainty.

.. seealso::

   Tutorials: :doc:`../tutorials/notebooks/02_hybrid_selection`, :doc:`../tutorials/notebooks/09_ablation`, :doc:`../tutorials/notebooks/11_calibration`.
