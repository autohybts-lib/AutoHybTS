Concepts
========

The path of the data
--------------------

.. code-block:: text

   series ──► fit: validates, seals the test and freezes the origins
                │
                ├─ development history
                │     ├─ V_search: staged search S0–S4 (bases, retention, hybrids,
                │     │            refinement, advanced families)
                │     └─ V_select: incremental selection (a challenger replaces the
                │                  champion only if it passes the acceptance rule)
                │
                └─ sealed test: read a single time by evaluate_external
                                      │
   finalize_model: refits the champion on the history before the test
                                      │
   predict_model / stream: forecast in variant A (several steps) or B (one step)

The library's principle: **extra complexity is accepted only when it passes
validation**. A hybrid that does not improve on the champion on the same origins, with
enough gain and wins, is rejected and the simpler model stays. Every decision, accepted
or not, is recorded with its reason (``result.decisions``).

Glossary
--------

.. glossary::

   base
      The model that forecasts the series (``arima``, ``knn``, ``linear`` …). On its own,
      it is a ``standalone`` candidate.

   corrector
      The model that learns the error the base leaves, from a window of past residuals. In
      the additive hybrid, ``ŷ = b + α·r̂``.

   residual
      The error of the base, ``e = y − b``. It is generated **inside the training data**
      of each origin by the configured residual source (``historical_oof_1step`` by
      default: out-of-sample one-step forecasts), never on the whole series.

   candidate
      A complete configuration to evaluate: strategy, base, corrector, window, α, residual
      source, variant. It has a stable ``candidate_id`` (hash of the configuration).

   structure
      The shape of a candidate, without configuration details: ``arima-mlp`` is the
      structure of every candidate with an ARIMA base and an MLP corrector, whatever the
      window or α.

   family
      The group of models used in the incremental selection: statistical, linear,
      kernel, tree, neural, residual, decomposition, combination, dynamic selection (and
      those that join when used). Each family brings one finalist to the challenge.

   champion
      The winning candidate so far. Selection starts from the best standalone model.

   challenger
      The finalist of a family that tries to replace the champion on ``V_select``.

   origin
      The point of the series where a model is fitted and from which it forecasts.
      Nothing after the origin enters the fit of that forecast.

   unit
      One origin of one series (``series:fold`` in a panel). Comparisons are paired by
      unit: both models are evaluated on exactly the same ones.

   V_search
      The origins used by the search: evaluating, retaining and refining candidates, and
      tuning hyperparameters (``tune_model``).

   V_select
      The most recent origins of the development history, used only to decide the
      champion. Separating search and decision reduces the optimism of picking the winner
      where it was searched for.

   sealed test
      The last ``test_size`` points (default ``fh``). Neither the search nor the selection
      can read them; ``evaluate_external`` reads them once per finalized model.

   coverage
      The share of the expected units in which the candidate produced a valid metric.
      The acceptance rule requires full coverage.

   ledger
      The record of every proposal of the search, evaluated or not (pruned by the
      budget, skipped for capability, duplicated, stopped), with its reason:
      ``result.search.ledger_frame()``.

   variant
      How the forecasts meet the future: **A**, ``fh`` steps at once from the origin
      (default); **B**, one step at a time, with the actual value revealed before the
      next one and frozen parameters (:doc:`user_guide/variant_b`).

   profile
      The set of selection rules: ``standard`` (default), ``quickstart`` (without the
      variability criterion), ``research_strict`` and ``exploratory``.

   budget
      How many candidates the search may evaluate: ``smoke``, ``small`` (default),
      ``medium`` or ``large``, in total, per family and per stage.

Where each concept is detailed
------------------------------

* Temporal split, origins and residuals: :doc:`user_guide/temporal_validation`.
* Acceptance rule, families and profiles: :doc:`user_guide/selection`.
* Stages, budget and ledger: :doc:`user_guide/search_space`.
* Strategies for building the hybrid system: :doc:`user_guide/strategies`.
* Variants A and B: :doc:`user_guide/variant_b`.
