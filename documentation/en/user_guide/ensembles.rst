Ensembles and compositions
==========================

An *ensemble* combines **forecasts of complete candidates**: standalone models,
residual hybrids or other ensembles. A *corrected composition* applies a residual
corrector on top of a composite candidate. Both forms allow nesting up to three levels,
for example an ensemble whose member is a hybrid on top of another ensemble.

.. code-block:: python

   linear = exp.create_model("linear")
   ensemble = exp.create_ensemble(["seasonal_naive", linear, "knn"], method="nnls")
   corrected = exp.create_hybrid(ensemble, "linear")      # ensemble + residual correction
   of_hybrids = exp.create_ensemble(
       [exp.create_hybrid("naive", "linear"), exp.create_hybrid("seasonal_naive", "knn")]
   )

Aggregations
------------

==================  ===========================================================================
``method``          What it does
==================  ===========================================================================
``equal``           mean; weights ``1/K``
``median``          median per step (3 or more members)
``trimmed_mean``    drops ``max(1, round(trim·K))`` forecasts at each end per step and averages
                    the rest (``trim``, default 0.2; 3 or more members)
``inverse_mse``     weights ``∝ 1/MSE`` of the one-step out-of-sample errors
``nnls``            non-negative least squares on the out-of-sample forecasts, normalized and
                    shrunk towards equal weights:
                    ``w = (1 − shrinkage)·w_nnls + shrinkage/K`` (default 0.5)
==================  ===========================================================================

Learned weights use only each origin's training vector: every member is refitted on the
prefix and forecasts the tail one step at a time; then all of them are fitted on the
whole vector. With few origins, free weights tend to do worse than the simple mean —
hence the shrinkage. The weights appear in the diagnostics of each trial
(``weight_i``, in the order of the members in the label).

In the search
-------------

.. code-block:: python

   result = exp.compare_models(include=("standalone", "residual", "ensemble"))

With ``"ensemble"``, stage S4 proposes (family ``ensemble``, with
``ensemble_methods``):

* ensembles of the best base of each family (up to 3 members);
* ensembles of the two best residual structures (weighted aggregations);
* the equal-weight ensemble corrected by each configured corrector.

The equal-weight version of each kind comes first, so that a budget cut keeps the
simplest control. The ``ensemble`` family challenges the champion last (before
``dynamic``), only after the simpler structures, and goes through the same acceptance
rule: an ensemble that does not beat the champion is rejected with its reasons logged.

.. warning::

   ``"ensemble"`` stays out of the default ``include`` until its spurious acceptance
   rate is calibrated (:doc:`benchmarks`, ``--include standalone residual ensemble``).
   Every new family is one more challenger in the chain and increases the chance of
   accepting complexity by chance.

Cost
----

Each trial counts its leaf-model fits (``TrialResult.n_fits``;
``result.search.fit_count``). An ``nnls`` ensemble of ``K`` members fits ``2K`` models
per origin (prefix and whole vector); a hybrid, three (base, base on the prefix and
corrector). Always weigh gains against cost.
