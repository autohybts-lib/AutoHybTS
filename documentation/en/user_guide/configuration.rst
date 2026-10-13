Choosing the configuration
==========================

The ``AutoHybridExperiment`` parameters that change the result the most, with the
default, when to change them and what it costs. The full list is in the
:doc:`../api/index`.

.. list-table::
   :header-rows: 1
   :widths: 20 16 40 24

   * - Parameter
     - Default
     - When to change
     - Effect and cost
   * - ``fh``
     - 14
     - always: it is the horizon the decision needs (7 for a week of daily data, 12 for a
       year of monthly data)
     - long horizons accumulate error in variant A
   * - ``test_size``
     - ``fh``
     - ``0`` to forecast the future without a sealed test (operation); a larger block for
       a steadier final evaluation
     - sealed points leave the development history
   * - ``fold``
     - ``"auto"`` (7)
     - a smaller integer only to explore quickly
     - the acceptance rule was calibrated for 7 origins; with fewer, it accepts spurious
       structures more often; the cost grows with the number of origins
   * - ``seasonal_period``
     - from the frequency
     - an integer when the cycle is known; ``"auto"`` to test the cycle before the
       search (:doc:`diagnostics`)
     - sets the seasonal models, the STL and the MASE scale
   * - ``variant``
     - ``"A"``
     - ``"B"`` when each value arrives before the next forecast (:doc:`variant_b`)
     - the errors of the two variants are not comparable
   * - ``budget``
     - ``"small"``
     - ``"smoke"`` to test; ``"medium"``/``"large"`` to explore more hybrids
     - cost proportional to candidates × origins; ``max_trials`` fixes the total
   * - ``base_models``, ``correctors``
     - 9 bases, 3 correctors
     - to include optional models (``sarima``, boosting …) or shrink the search
     - each extra model multiplies the hybrid combinations
   * - ``profile``
     - ``"standard"``
     - ``"research_strict"`` to require full coverage; ``"quickstart"`` turns off the
       variability criterion
     - changes what the acceptance rule requires
   * - ``significance_alpha``
     - ``None``
     - to also require a paired Wilcoxon test
     - almost never accepts spurious hybrids, but rejects most useful gains
       (:doc:`selection`)
   * - ``metrics``
     - MASE, MAE, RMSE, sMAPE
     - ``RMSSE`` first for intermittent demand (:doc:`results`)
     - the first metric ranks and decides
   * - ``differencing``
     - ``None``
     - ``"auto"`` for the lag models to choose their differences
     - chosen on each origin's training data; needs the ``stats`` extra
   * - ``max_wall_time_s``
     - ``None``
     - to cap the search time
     - results may change between machines; to reproduce, limit by count
   * - ``n_jobs``
     - 1
     - to evaluate candidates in parallel
     - does not change the result (the limits apply to the ordered list of proposals)
   * - ``session_id``
     - 42
     - to repeat an experiment with another seed
     - the same seed and the same data give the same search

A starting point per goal:

.. code-block:: python

   # quick exploration (indicative results)
   AutoHybridExperiment(fh=14, budget="smoke", fold=3)

   # a decision for a report
   AutoHybridExperiment(fh=14)                         # calibrated defaults

   # daily operation, one step at a time
   AutoHybridExperiment(fh=1, variant="B", test_size=0)

   # series with an uncertain cycle and a trend
   AutoHybridExperiment(fh=14, seasonal_period="auto", differencing="auto")
