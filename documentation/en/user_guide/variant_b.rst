Variant B: one step with frozen parameters
==========================================

In variant A, the model forecasts ``fh`` steps at once from the origin. In
**variant B**, it forecasts one step, receives the actual observation and forecasts the
next — without refitting: the parameters stay frozen and only the state (context)
moves forward.

.. code-block:: python

   exp = AutoHybridExperiment(fh=7, variant="B").fit(y)
   result = exp.compare_models()
   final = exp.finalize_model(result.best)

   stream = final.stream()
   ticket = stream.predict_next()
   stream.observe(observed_value)          # reveals the value; nothing is refitted

* The observed residual ``y − b`` enters the corrector window only after the
  observation.
* ``FinalizedForecaster.update_state`` returns a new object and refuses ``refit=True``.
* The ``ma`` decomposition works in B (it is recomputed causally); ``stl`` does not.
* Dynamic corrector selection (``dref``, ``local``) exists only here: see
  :doc:`strategies`.
