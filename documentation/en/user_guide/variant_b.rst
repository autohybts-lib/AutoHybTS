Variants A and B: how the forecast meets the future
===================================================

The variant says **when the actual value becomes available** while the model forecasts.
In both, the model is fitted once at the origin; what changes is how the next ``fh``
forecasts are produced. The variant is chosen in the experiment and applies to the search,
the selection, finalization and forecasting:

.. code-block:: python

   AutoHybridExperiment(fh=14)                 # variant A (default)
   AutoHybridExperiment(fh=7, variant="B")     # variant B

Variant A: several steps at once (default)
------------------------------------------

At origin *t*, the model forecasts *t+1, …, t+fh* **without seeing any actual value after
t**. To forecast *t+3*, the values of *t+1* and *t+2* do not exist yet: the model uses its
own forecasts of them.

In a hybrid (``ŷ = b + α·r̂``), the corrector needs the last ``w`` residuals. After the
origin there is no observed residual, so the window is fed with the **predicted
residuals** ``r̂``. The error tends to accumulate along the horizon, and the corrector's
gain is usually strongest at the first step. When the dominant pattern is regular and
well modelled, the difference from variant B can be small: in notebook 04, on a 28-day
stretch, variant A even had a lower error.

**When to use:** when the whole plan is needed in advance, for example the demand of the
next 14 days computed today.

Variant B: one step at a time, frozen parameters
------------------------------------------------

The model forecasts **only the next step**. The actual value arrives, enters the context,
and the model forecasts the following step, until ``fh`` steps are done. The parameters
**do not change**: nothing is refitted; only the state (the observed context) moves on.

In a hybrid, the observed residual ``y − b`` enters the corrector window **only after**
the observation: the corrector works with true errors, not with errors it predicted
itself.

**When to use:** in continuous operation, when each value arrives before the next
forecast, for example monitoring or a next-day forecast updated every day. The choice
follows the operation, not a guaranteed accuracy advantage.

Example: ``fh = 3``, origin at day 10
-------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 16 42 42

   * -
     - Variant A
     - Variant B
   * - Day 11
     - forecasts with data up to day 10
     - forecasts with data up to day 10
   * - Day 12
     - forecasts with data up to day 10 (and the forecast of day 11)
     - forecasts with data up to the **observed day 11**
   * - Day 13
     - forecasts with data up to day 10 (and the forecasts of days 11 and 12)
     - forecasts with data up to the **observed day 12**
   * - Fits
     - 1, at the origin
     - 1, at the origin; none afterwards

Effect on validation
--------------------

Temporal validation simulates the chosen variant. In A, each origin produces one
``fh``-step forecast. In B, each origin produces ``fh`` one-step forecasts, and each
target is revealed only after its forecast. The errors of the two variants **are not
comparable**: in B every error is one step ahead, in A they cover horizons 1 to ``fh``.
Always compare models within the same variant.

Compatibility
-------------

.. list-table::
   :header-rows: 1
   :widths: 46 18 18

   * - Feature
     - A
     - B
   * - Base models that take new observations without refitting
       (``online_state_update``)
     - yes
     - yes
   * - Base models without that capability
     - yes
     - ``SKIPPED``
   * - ``ma`` decomposition (recomputed causally)
     - yes
     - yes
   * - ``stl`` decomposition (fitted on the training data only)
     - yes
     - no
   * - Dynamic selection of the corrector (``dref``, ``local``)
     - no
     - yes
   * - Residual sequence correctors (``arima_sequence`` …)
     - yes
     - yes
   * - Multi-step ``predict_model``
     - yes
     - only ``fh=1``; use ``stream()``

In production (variant B)
-------------------------

.. code-block:: python

   exp = AutoHybridExperiment(fh=7, variant="B").fit(y)
   result = exp.compare_models()
   final = exp.finalize_model(result.best)

   stream = final.stream()
   ticket = stream.predict_next()      # forecast, base and correction of the next step
   stream.observe(observed_value)      # reveals the value; nothing is refitted

* ``predict_next`` and ``observe`` strictly alternate: forecasting twice without observing
  is refused.
* ``ForecastTicket`` carries the forecast, the base part and the correction of that step.
* ``FinalizedForecaster.update_state`` returns a **new** object with the new observations
  and refuses ``refit=True``. ``training_cutoff`` (end of training) differs from
  ``last_observation`` (last value received).

The notebook :doc:`../tutorials/notebooks/04_rolling_one_step` shows the streaming, the
base and correction at each step and both variants on the same stretch, with figures.
