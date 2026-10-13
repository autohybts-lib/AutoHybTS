Evaluating the procedure itself
===============================

``autohybts.benchmarks`` measures the selection procedure itself. Nothing here is
imported by ``import autohybts``.

Ablation
--------

Four procedures with the same budget and the same external tests.

=======================  ============================================================
Procedure                What it does
=======================  ============================================================
``lowest_error``         the finalist with the lowest error on ``V_select``, no
                         thresholds
``incremental``          the guarded incremental chain, all families
``incremental_simple``   the same chain, only structures with up to 2 models
``guarded_nested``       like ``incremental``; if nested CV (same families) says the
                         procedure does not beat the best base of the inner
                         selection, the best base on ``V_select`` is kept
=======================  ============================================================

.. code-block:: python

   from autohybts.benchmarks import (
       ablation_report, ablation_uncertainty, run_ablation, synthetic_cases,
   )

   table = run_ablation(synthetic_cases(12), seeds=(0, 1, 2))
   ablation_report(table)                  # per procedure
   ablation_report(table, by="stratum")    # per stratum (predictability, subset)
   ablation_uncertainty(table)             # interval resampling series

.. code-block:: bash

   python -m autohybts.benchmarks.ablation --dataset synthetic --series 12 --seeds 0 1 2
   python -m autohybts.benchmarks.ablation --dataset m4-daily --series 20 --out ablation.csv

With ``--out``, the resolved protocol (environment, commit, budget, space, seeds and
series) goes to ``<out>.protocol.json``. Uncommitted local changes are not detected:
record them separately.

Each row (series × seed × procedure) holds the external error of the chosen model, that
of the best base on ``V_select`` and those of the ``naive`` and ``seasonal_naive``
references, and the full cost of the procedure:

=====================  ==============================================================
Column                 Meaning
=====================  ==============================================================
``search``             search used (``full`` is shared by three procedures)
``candidates``         candidates evaluated, including the inner searches of nested
                       CV in ``guarded_nested``
``units``              units (origin × series) evaluated in the search, the
                       selection and nested CV; each unit is a fresh model
``seconds``            cost of the procedure: search + selection (+ nested CV)
``seconds_*``          per phase; ``seconds_refit`` (refit before the test) is
                       reported only
``stop_reason``        why the search stopped
``max_trials_total``   configured budget
=====================  ==============================================================

The report gives, per procedure: median gain against the best base on the test,
fractions of cases better than, equal to and worse than it, fraction of hybrids and
mean complexity, validation–test gap, mean rank, spread across seeds, coverage and
cost. ``ablation_uncertainty`` summarizes each series (median across seeds) before
resampling series: seeds of the same series are not independent units.

.. warning::

   The external test is for reporting only: do not pick the "winning" procedure with
   it.

.. note::

   Budget presets limit only the number of candidates, so the same series and the same
   seed give the same search on any machine. A wall-clock deadline
   (``max_wall_time_s``) is optional; with it, the evaluated candidates can change
   between machines, which is why the ablation does not use it.

The ``09_ablation`` notebook has a complete run (12 synthetic series × 3 seeds,
``small`` budget) with tables and figures.

Calibration of the acceptance rule
----------------------------------

``autohybts.benchmarks.calibration`` measures the rule on controls with a known
answer:

* ``white_noise`` (negative control): level plus white noise; a hybrid champion is a
  **false acceptance** of complexity;
* ``ar_residual`` (positive control): trend, weekly seasonality and AR(0.8) noise, with
  bases that do not model the AR; a standalone champion is a **rejected useful gain**.

For each series and number of origins in ``V_select`` (``fold``), one search and one
selection run once; each policy (the rule, the rule with Wilcoxon …) is applied to the
same finalists. Rates are given per chain (the final champion) and per comparison, with
intervals that resample series. The ``calibration`` and ``confirmation`` sets use
disjoint seeds: tune on one, check once on the other. The external test is not read.

.. code-block:: bash

   python -m autohybts.benchmarks.calibration --series 50 --folds 2 3 5 --out cal.csv
   python -m autohybts.benchmarks.calibration --series 50 --split confirmation --out conf.csv

The ``11_calibration`` notebook runs both controls on both sets, with tables and a plot
of the rates per number of origins.

.. seealso::

   Tutorials: :doc:`../tutorials/notebooks/05_benchmark_m4`, :doc:`../tutorials/notebooks/09_ablation`.
