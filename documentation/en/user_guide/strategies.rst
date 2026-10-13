Hybrid system strategies
========================

Each candidate has a **short name** that describes its structure (``arima-mlp``) and a
full label with the configuration (window, alpha, residual source). The source papers
of each strategy are in :doc:`../references`.

==========================  ========================  ============================================
Strategy                    Name                      Idea
==========================  ========================  ============================================
``standalone``              ``arima``                 one model
``residual_additive``       ``arima-mlp``             ``ŷ = b + α·r̂``: the corrector forecasts
                                                      the error
``residual_multistage``     ``arima-mlp-linear``      each stage corrects the previous one's error
``decomposition``           ``ma(linear-mlp)``        base on the smooth part, corrector on the
                                                      volatile one
``fixed_combination``       ``comb(arima,knn)``       weighted mean of bases
``residual_combination``    ``arima-comb(mlp,knn)``   base + weighted mean of correctors
``dynamic_selection``       ``dref(arima-mlp,knn)``   picks the corrector at each step (B only)
==========================  ========================  ============================================

Additive correction
-------------------

The base model ``b`` is fitted; the residuals ``e = y - b`` are generated inside the
training data by the configured source; the corrector learns ``e[t]`` from the last
``w`` entries. In variant A (multi-step), the corrector window is fed with its own
**predicted** residuals after the origin — never with residuals observed in the future.

.. code-block:: python

   exp.create_hybrid("arima", "mlp", window=7, alpha=1.0,
                     residual_source="historical_oof_1step")

Multiple stages
---------------

Stage *k* is a residual hybrid whose base is stage *k−1*; the target of stage *k* is
the error of the whole previous model, generated inside the training data. Up to 3
stages.

.. code-block:: python

   exp.create_hybrid("arima", "mlp", extra_stages=["linear"])

By default, extra stages repeat the window, alpha and residual source of the first one.
``stage_params`` changes that per stage; in the search,
``AutoHybridExperiment(stage_params="per_stage")`` also varies the window and alpha of
the extra stage during refinement.

.. code-block:: python

   exp.create_hybrid("arima", "mlp", extra_stages=["linear"],
                     stage_params=[{"window_size": 28, "alpha": 0.5}])

Decomposition
-------------

``y = s + v``. The base forecasts the smooth part ``s`` and the corrector the volatile
part ``v``.

``ma``
   **causal** moving average (``s[t]`` uses ``y`` up to ``t``), with the window chosen
   by the kurtosis of the volatile part closest to 3, computed on the training data
   only. Recomputed causally on any context: valid in variants A and B.
``stl``
   robust STL fitted on the training data only; variant A only.

.. code-block:: python

   exp.create_decomposition("linear", "mlp", method="ma")       # window by kurtosis
   exp.create_decomposition("linear", "mlp", method="ma", ma_window=14)
   exp.create_decomposition("linear", "mlp", method="stl")

Fixed combination
-----------------

Bases combined with weights fixed at fitting time: ``equal`` or ``inverse_mse``
(inverse of the one-step out-of-sample error, measured on the training data itself).

.. code-block:: python

   exp.create_combination(["arima", "knn"], weights="inverse_mse")

Corrector combination: several correctors on the same residual,
``ŷ = b + α·Σ wⱼ r̂ⱼ``. With ``inverse_mse``, the weights come from each corrector's
error on the last rows of its own training data, fitted on the earlier ones.

.. code-block:: python

   exp.create_hybrid("arima", "mlp", combine_correctors=["knn", "linear"],
                     combination_weights="inverse_mse")

Dynamic selection (variant B only)
----------------------------------

A pool of correctors (plus the option of **not correcting**). At each step, the
residuals right before the origin are compared with a library of past situations (the
DSEL, made of one-step **out-of-sample** errors on a tail of the training data, never on
the scoring window) and the ``k`` most similar neighbours decide:

``dref``
   uses the corrector with the lowest error on those neighbours — or none.
``local``
   fits a small Kernel Ridge on the neighbours, from (base forecast, correction) to the
   observed value, for that step only.

In the search, the grid is the legacy one (Silva et al., 2020): ``k ∈ {10, 5, 20}``,
``lag ∈ {7, 14}`` and, for ``local``, ``alpha ∈ {1, 0.1}``; the default values come
first.

Without an observation between steps there is nothing to select: in variant A the
strategy is refused.

.. code-block:: python

   exp = AutoHybridExperiment(fh=7, variant="B").fit(y)
   result = exp.compare_models(include=("standalone", "residual", "dynamic"))

In the search
-------------

.. code-block:: python

   result = exp.compare_models(
       include=("standalone", "residual", "multistage", "decomposition", "combination")
   )

``include`` accepts ``"standalone"`` (required), ``"residual"`` (with it, the default),
``"multistage"``, ``"decomposition"``, ``"combination"``, ``"ensemble"``
(:doc:`ensembles`) and ``"dynamic"`` (variant B only). ``"multistage"`` and
``"dynamic"`` build on the residual structures and require ``"residual"``.

The search stages (S0–S4), the budget, the wall-clock deadline and the ledger are in
:doc:`search_space`.

.. seealso::

   Tutorials: :doc:`../tutorials/notebooks/02_hybrid_selection`, :doc:`../tutorials/notebooks/08_advanced_families_and_tuning`.
