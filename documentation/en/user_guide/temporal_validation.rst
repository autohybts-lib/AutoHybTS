Leak-free temporal validation
=============================

Everything in AutoHybTS starts from a temporal split frozen in ``fit``:

.. code-block:: text

   |------------- development history -------------|-- sealed test --|
   |  train  ... | V_search (search) | V_select (selection) | fh points |

* **Sealed test**: the last ``test_size`` points (default ``fh``). It is an opaque
  object (``SealedTest``): it cannot be iterated, indexed or converted. Only
  ``evaluate_external`` reads it, once per finalized model.
* **V_search**: origins used by the search (fitting structures, retention, refinement,
  ``tune_model``).
* **V_select**: the most recent origins, used only to decide the champion. Keeping the
  two layers apart reduces the optimism of picking the winner in the same place where
  it was searched for.

At each origin ``o``, the model is fitted on ``y[:o+1]`` and forecasts ``fh`` steps.
Nothing after the origin enters fitting, transforms, the MASE scale, corrector windows,
the moving-average window or combination weights.

Parameters
----------

.. code-block:: python

   AutoHybridExperiment(
       fh=14,                     # horizon per origin
       fold="auto",               # origins per layer: 7 if they fit (ADR-0008)
       fold_strategy="expanding", # or "rolling" with window_size
       test_size=None,            # None = fh; 0 turns the sealed test off
       seasonal_period=None,      # inferred from the frequency (daily: 7, monthly: 12 ...)
   )

Corrector residuals
-------------------

The corrector learns from residuals generated **inside** each origin's training vector:

``historical_oof_1step`` (default)
   a base fitted on the prefix forecasts each point of the tail one step ahead, with
   frozen parameters.
``historical_oof_multistep``
   the tail is cut into blocks; each block is forecast by a base refitted before it,
   with errors at horizons 1..block, like those of variant A.
``in_sample``
   the base's fitted values on its own training data. Experimental: it can be
   optimistic.

Every residual stores its origin, target, horizon and ``available_at``; a window that
uses a residual later than the origin raises ``TemporalLeakageError``.

Temporal cross-validation
-------------------------

``exp.cross_validate(folds=8, step=7)`` creates its own folds (rolling origin, anchored
at the end of the development history) and evaluates several models on the same
origins. Nested CV, ``exp.nested_cross_validate``, repeats the whole procedure
(search + selection) in each outer fold and measures how well the automatic choice
generalizes.

.. code-block:: python

   result = exp.compare_models(include=("standalone", "residual", "decomposition"))
   nested = exp.nested_cross_validate(outer_folds=3)
   nested.include    # the families of the last compare_models (or include=...)
   nested.summary    # autohybts, best_standalone and the fixed references
   nested.cost       # per fold: candidates, units, budget, stop, seconds

Without ``include``, the inner searches repeat the families of the last
``compare_models``: that is the procedure being estimated. The ``best_standalone`` row
is the best base **of the inner selection** of each fold, chosen without the outer
block; it is the fair reference for saying whether hybridization pays off.
``nested.total_candidates`` and ``nested.total_seconds`` add up the work of all folds.
