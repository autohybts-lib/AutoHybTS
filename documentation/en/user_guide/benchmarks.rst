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

Non-finite gains (``NaN`` or infinity) do not enter the better, tied or worse shares.
The report includes ``n_rows``, ``n_paired``, ``n_invalid_gain`` and ``paired_coverage``
for the supplied rows; these counts do not include planned cases absent from the table.
The existing ``coverage`` column still describes validation coverage. In uncertainty
reports, ``n_series`` counts the series present, while ``n_paired_series``,
``n_invalid_series`` and ``paired_series_coverage`` identify series with a finite gain
after aggregating valid seeds. Without valid pairs, estimates and intervals remain
undefined instead of being reported as zero.

``ablation_protocol`` records metrics, the initial policy and budget limits from the
configuration supplied through ``space``, including profile, significance and budget
overrides. Per-series resolved origins and effective family order belong to comparison
snapshots; generating this protocol performs no searches and does not open the external
test. The report's median gain aggregates series × seed rows, whereas the interval
aggregates seeds within each series before resampling.

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

   Tutorials: :doc:`../tutorials/notebooks/05_benchmark_m4`, :doc:`../tutorials/notebooks/09_ablation`,
   :doc:`../tutorials/notebooks/17_benchmark_study`.

Standalone reference and procedure version
------------------------------------------------------------

Schema ``ablation/3`` uses ``lowest_select_error`` for both the nested CV reference
and the ``guarded_nested`` fallback: the non-exogenous standalone finalist with the
lowest valid ``V_select`` error and sufficient coverage. Rows record
``standalone_reference_rule``, ``best_base_id`` and ``initial_champion_id``.
The initial champion follows search ranking and may differ from this reference.
Previous schema ``ablation/2`` used the initial champion within nested CV and another
rule for fallback. The schemas evaluate different procedures and must be identified
separately; relabeling old result files is not sufficient.

Configurable protocol and resume
------------------------------------------------------------

The CLI accepts JSON schema ``autohybts-ablation-config/1``. The supplied
``tools/protocols/ablation_demo.json`` uses synthetic data and a demonstration budget:

.. code-block:: bash

   python -m autohybts.benchmarks.ablation --config tools/protocols/ablation_demo.json --checkpoint artifacts/demo.sqlite --out artifacts/demo.csv
   python -m autohybts.benchmarks.ablation --config tools/protocols/ablation_demo.json --checkpoint artifacts/demo.sqlite --resume --out artifacts/demo.csv

``preset="demo"`` keeps two origins per layer and the ``smoke`` budget. ``research``
uses ``fold="auto"`` and ``small``; it is a configuration starting point, not an approved
scientific protocol or a guarantee of seven origins on short series. ``space`` accepts
experiment parameters such as models, residual sources, grids, metrics, profile and
origins. Horizons, test sizes and dataset identities come from cases and cannot be
overridden in ``space``. Official benchmark tests are preserved. Descriptor seasonality
is carried into each case; root-level ``seasonal_period`` provides an explicit override
(``null`` requests inference).

For benchmarks, ``series_ids`` fixes an explicit list, or ``series`` and ``sampling_seed``
define a sample. Sampling uses a different seed from model ``seeds``. Do not combine
explicit IDs with a count, or ``--config`` with scientific CLI overrides. Without JSON,
``--fold``, ``--sampling-seed`` and ``--series-ids`` are also available. Requesting more
series than available raises an error rather than silently shrinking the sample.

The Python API accepts the same cases, seeds and parameters:

.. code-block:: python

   table = run_ablation(
       synthetic_cases(2), seeds=(0, 1), space={"fold": "auto"},
       checkpoint="artifacts/study.sqlite", on_error="record",
   )
   table = run_ablation(
       synthetic_cases(2), seeds=(0, 1), space={"fold": "auto"},
       checkpoint="artifacts/study.sqlite", resume=True, on_error="record",
   )
   table.attrs["coverage"]   # planned denominator and case states
   table.attrs["cases"]      # attempts, reasons, timings and case snapshots
   table.attrs["protocol"]   # configuration, data, environment and code identity

The SQLite checkpoint atomically saves all four procedures of each series × seed.
``SUCCEEDED`` cases are never rerun. Interrupted cases restart from the beginning; ``FAILED``
and ``SKIPPED`` cases require ``--retry-failed`` or ``retry_failed=True`` together with
explicit resume. Errors still stop by default; ``--on-error record`` continues after
case failures. Configuration and temporal leakage errors always stop execution.

The study hash includes configuration, data values and dates, versions and installed
Python sources, including uncommitted edits. Incompatible studies cannot resume.
An OS lock admits one writer per file and is released on process exit; the ``.lock``
file may remain on disk. Raw series values are not stored. Checkpoints never supply
trials to model search or selection.

``full`` and ``simple`` phases retain resolved snapshots; ``nested`` retains inner-search
snapshots, actual splits, errors and costs by outer fold. Interrupted attempts keep
their history; an unknown duration after abrupt process exit is ``null``. CSV contains
completed results; failed cases remain in the checkpoint and ``<out>.protocol.json``,
alongside coverage and snapshots. This JSON is required to reconstruct the planned
denominator, which a success-only CSV does not contain.
``coverage.procedures`` distinguishes planned, completed, metric-valid and paired-gain
cases for each procedure. Completing a case does not imply that every metric is defined.
Undefined values are ``null`` in JSON and ``NaN`` in reconstructed tables.
``n_attempts`` and ``known_seconds_total`` include previous attempts; durations lost to
abrupt process termination are counted in ``n_unknown_duration_attempts``. Partial
snapshots from earlier attempts are preserved when a case is retried.
