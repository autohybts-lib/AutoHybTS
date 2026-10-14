Version 1 and publishing
========================================

1.0.0rc4 — SQLite tracking, MLOps and exogenous variables, October 14, 2026
---------------------------------------------------------------------------

Compatible additions: without ``X`` and without a database, candidate IDs, the default
protocol hash and the model persistence schema do not change.

* **SQLite tracking:** ``ExperimentStore`` writes experiments (candidates, per-layer and
  per-origin metrics, decisions, champion, external evaluation) and answers read-only SQL
  queries; series values only with ``values=True``.
* **MLOps in the same database:** model registry with the artifact's SHA-256, promotion with
  stage history, ``load`` with ``trusted=True``, logged forecasts, observations and
  covariates, ``monitor`` (rolling MAE against the reference, with alerts); synthetic series
  ``commodity_price_daily``.
* **Exogenous variables:** ``fit(y, X)`` with ``past`` columns (lagged by the horizon) and
  declared ``future`` columns, ARIMAX, SARIMAX, lag models and hybrids with covariates, an
  ``exogenous`` family decided by the acceptance rule, forecasting with ``X_future``,
  streaming, persistence and panels of series.
* **Automatic choice of covariates:** ``exog_selection`` (``ccf``, ``granger``, ``lasso``,
  ``stepwise_bic``, ``mutual_info``, ``vote``, ``auto``), ``exp.exog_report()`` and
  ``exp.compare_exog()`` (Wilcoxon and Diebold–Mariano).
* **Datasets with covariates:** ``list_exog_datasets()`` and ``load_exog_dataset()``
  (seatbelts, bjsales, uschange, insurance, elecdaily, vic_elec_daily).
* **Documentation:** guides on SQLite tracking, MLOps and covariates; notebooks 18 (analysis
  with SQL), 19 (MLOps flow), 20 (covariates on benchmarks) and 21 (MLOps with covariates on
  real electricity demand); co-authorship with AI (Claude and Codex) on the *About* page.

.. code-block:: bash

   python -m pip install "autohybts[stats,plot]==1.0.0rc4"

1.0.0rc3 — series diagnostics and preprocessing, October 14, 2026
-----------------------------------------------------------------

Compatible additions: candidate IDs, the default protocol hash and the persistence schema
do not change, and every new feature acts only when requested.

* **Seasonality:** ``exp.seasonality_report()`` (seasonal strength and a Friedman test with
  cycles as blocks) and ``seasonal_period="auto"``, which tests the cycle on the first
  origin's training data before passing the period to the models.
* **SARIMA:** the ``sarima`` model (``stats`` extra), outside the default search.
* **Automatic differences:** ``differencing="auto"`` on lag models chooses the seasonal
  difference and up to two differences on each origin's training data, by KPSS or by the
  ADF rule of the replication project.
* **Characterization:** ``exp.characterize()`` with the two rules of the replication
  project and the ``stationarity``, ``acf_pacf``, ``decomposition`` and
  ``plot_characterization`` plots.
* **Reports:** ``exp.stationarity_report()`` (ADF, KPSS and consensus),
  ``exp.decompose()``, ``exp.residual_diagnostics()`` (Ljung-Box on a base's residuals),
  ``exp.outlier_report()`` (AO/LS/TC candidates, exploratory only), the ``MAPE`` and
  ``MdAPE`` metrics and ``cmp.kruskal()`` (Kruskal-Wallis and Dunn).
* **Optional preprocessing:** ``transform="boxcox"``/``"log"`` and
  ``outliers="winsorize"``/``"interpolate"`` on lag models, fitted on each origin's
  training data; ``missing="ffill"`` fills gaps with the last observed value. All off by
  default.
* **Documentation:** variants A and B, concepts, reading the results, choosing the
  configuration, troubleshooting, maintenance, the complete API reference, figures in
  every tutorial and notebooks 16 (diagnostics) and 17 (17 classical series in three
  classes, with five models in protocols A and B, 3, 6 and 12 lags and horizons up to 12).

.. code-block:: bash

   python -m pip install "autohybts[stats,plot]==1.0.0rc3"

1.0.0rc2 — PyPI documentation, October 13, 2026
------------------------------------------------------------

This candidate includes the README updated after the first PyPI publication.
PyPI is the main installation path, commands pin ``1.0.0rc2`` and both language
versions cover extras, Jupyter and the publishing workflow. Algorithms, public
signatures, candidate IDs and the persistence schema retain ``1.0.0rc1`` behavior.
This remains a release candidate; the criteria for stable ``1.0.0`` still apply
as described below.

.. code-block:: bash

   python -m pip install "autohybts[stats,plot]==1.0.0rc2"

Version: `1.0.0rc2 on PyPI <https://pypi.org/project/autohybts/1.0.0rc2/>`_.

1.0.0rc1 — candidate, October 13, 2026
------------------------------------------------------------

This prerelease prepares the first stable AutoHybTS release. Its scope includes
standalone forecasters, residual hybrids, multiple stages, decomposition, optional
combinations and ensembles, panels of local models, search and hyperparameter
tuning, temporal validation, reports and persistence. Variant A forecasts multiple
steps; variant B forecasts one step per update with frozen parameters.

The candidate was `published on PyPI on October 13, 2026
<https://pypi.org/project/autohybts/1.0.0rc1/>`_, with a wheel and source
distribution. Publication through ``publish.yml`` was approved in the GitHub
``pypi`` environment.

The default rule requires a median gain of at least 5%, wins on at least 60% of
paired units and the variability criterion. ``fold="auto"`` uses 7 origins per
layer when history permits; shorter series receive fewer origins and a warning.
Calibration and confirmation on synthetic controls measured 2–4% false acceptance
and 54–56% useful gains accepted. These rates apply to the measured conditions;
they are not guarantees for real series or all optional families.

Delivery changes in this candidate:

* version ``1.0.0rc1`` and beta classification in package metadata;
* validation workflow with reusable CI on the tagged commit;
* exact agreement checked between tag, version and wheel/sdist metadata;
* installed-wheel test without extras, including A/B forecasting and persistence;
* Sphinx builds for both languages and StatsForecast contracts in CI;
* complete slow/heavy notebook and download checks in the validation workflow;
* candidate available on PyPI; TestPyPI reserved for the separate testing workflow.

Compatibility
-------------

The candidate v1 public API is documented in :doc:`api/index`, including experiment
parameters, returned results, exceptions, datasets and model-registration
interfaces. Private modules and attributes, cache details, execution times and
diagnostic message text are not compatibility contracts.

Before ``1.0.0``, necessary fixes may change the API between candidates; changes
will be recorded in release notes. Starting with the stable release:

* compatible fixes use patch versions;
* compatible additions use minor versions;
* removals and incompatible changes require a major version; APIs scheduled for
  removal will be documented and emit ``DeprecationWarning`` in a minor version
  before removal in the next major version.

Candidate IDs and the ``autohybts-model/1`` schema are versioned contracts. Changes
that invalidate IDs or artifacts require migration notes. The manifest records
environment versions, but pickle also depends on libraries and external models:
loading a model across arbitrary dependency versions is not guaranteed. Preserve
the training environment and inspect ``read_manifest`` before loading. See
:doc:`user_guide/persistence`.

Migration from ``0.0.1`` does not change signatures or IDs in this candidate. The
calibrated rule was already part of the code before the candidate. To reproduce
experiments, pin versions, seeds, models, grids, ``fold`` and thresholds; a
compatible API does not imply identical forecasts after changing dependencies.

Criteria for stable 1.0.0
----------------------------------------

The M3/M4 study has no results yet. Before the final version, freeze its protocol
with the calibrated rule, samples, seeds and budget; execute without using test
data to retune the rule and report gains, deteriorations, coverage and cost against
baselines. Also review installation through PyPI, public contracts and
limitations found while using the candidate.

Composite families remain optional until they pass chain-level calibration. New
model families, exogenous inputs, probabilistic forecasts and global models are
not requirements for this release. Synthetic evidence does not replace the study
on real series. Demonstration notebooks do not constitute that study.

The publishing flow and the checklist before the tag are in :doc:`maintainers`.
