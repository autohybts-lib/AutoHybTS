Troubleshooting
===============

"fold='auto': o histórico comporta N origem(ns) por camada"
-----------------------------------------------------------

The warning (in Portuguese, like every message of the library) says the series is too
short for 7 origins per layer. The experiment uses the largest number that fits
(``exp.fold_``) and warns because the acceptance rule was calibrated for 7: with fewer
units, it accepts spurious structures more often. Use a smaller ``fh``, a longer series
or read the decisions with caution; an integer ``fold`` silences the warning but does not
change the risk.

A model shows up as ``SKIPPED``
-------------------------------

The reason is in the ledger (``result.search.ledger_frame()``, column ``reason``) or in
the message of ``create_model``. The common cases:

* missing extra: the message carries the ``pip install "autohybts[...]"`` command;
* model incompatible with variant B (cannot take observations without refitting);
* series without a declared seasonality (``seasonal_naive``, ``sarima``, STL with
  period 1);
* capabilities outside the search: global model, non-point output or a maximum horizon
  shorter than requested.

The search goes on with the other models; only "no valid candidate left" stops the run.

``DataValidationError`` in ``fit``
----------------------------------

The library does not silently reorder, resample, drop duplicates or fill missing values.
The message says what to fix:

* missing values (``NaN``): handle them first, with your domain's criterion;
* duplicate or unordered dates: aggregate or sort the series;
* irregular frequency or gaps in the dates: reindex the series at the right frequency
  and decide how to fill; or pass ``frequency``.

My hybrid was rejected
----------------------

Look at ``result.decisions``: ``reason_codes`` says which criterion failed and
``median_gain``/``win_rate`` show by how much (:doc:`results`). Rejecting is the expected
behaviour when the gain is not robust: the simpler model stays. Do not tune thresholds to
make a hybrid pass; to measure the rule, use ``autohybts.benchmarks.calibration``
(:doc:`benchmarks`).

The search is slow
------------------

The cost is proportional to candidates × origins × fits per candidate
(:doc:`search_space`). To explore quickly: ``budget="smoke"``, ``fold=3`` (with the
calibration caveat) and fewer ``base_models``/``correctors``. ``n_jobs`` evaluates
candidates in parallel without changing the result. ``max_wall_time_s`` caps the time but
makes the result machine-dependent. Expensive models: ``sarima`` on long daily series,
``AutoARIMA``, networks on the CPU (``MambaNet`` especially).

The result changed from one machine to another
----------------------------------------------

With the same series, the same ``session_id`` and the same parameters, the search is the
same. Differences come from: ``max_wall_time_s`` (the deadline cuts different
proposals), GPU or TensorFlow (residual non-determinism) and different library versions.
The manifest of a saved model (``read_manifest``) records the versions used.

``SealedTestAccessError`` when evaluating again
-----------------------------------------------

``evaluate_external`` reads the sealed test a single time per finalized model. Reading it
again would turn the test into validation. Keep the result of the first call
(``external = exp.evaluate_external(final)``); to compare several models on the test,
use ``exp.compare_external()``, which evaluates each one once.

``predict_model`` refuses several steps in variant B
----------------------------------------------------

A variant B model forecasts one step at a time. Use ``fh=1`` or ``final.stream()``, which
alternates ``predict_next`` and ``observe`` (:doc:`variant_b`).

``load_model`` requires ``trusted=True``
----------------------------------------

Loading a model unpickles it, which runs code. ``trusted=True`` declares that the file
comes from a trusted source; to check integrity, pass ``sha256=`` (the hash received
through a trusted channel) or ``signing_key=`` (:doc:`persistence`).

StatsForecast and pandas 3
--------------------------

Current StatsForecast releases require pandas < 3. Install the ``nixtla-stats`` extra in
a separate environment (:doc:`nixtla`); without it, the ``sf_*`` models are ``SKIPPED``.
