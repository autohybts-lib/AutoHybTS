Version 1 and publishing
========================================

1.0.0rc1 — candidate, October 13, 2026
------------------------------------------------------------

This prerelease prepares the first stable AutoHybTS release. Its scope includes
standalone forecasters, residual hybrids, multiple stages, decomposition, optional
combinations and ensembles, panels of local models, search and hyperparameter
tuning, temporal validation, reports and persistence. Variant A forecasts multiple
steps; variant B forecasts one step per update with frozen parameters.

The default rule requires a median gain of at least 5%, wins on at least 60% of
paired units and the variability criterion. ``fold="auto"`` uses 7 origins per
layer when history permits; shorter series receive fewer origins and a warning.
Calibration and confirmation on synthetic controls measured 2–4% false acceptance
and 54–56% useful gains accepted. These rates apply to the measured conditions;
they are not guarantees for real series or all optional families.

Delivery changes in this candidate:

* version ``1.0.0rc1`` and beta classification in package metadata;
* CI reused on the tagged commit before any publication;
* exact agreement checked between tag, version and wheel/sdist metadata;
* installed-wheel test without extras, including A/B forecasting and persistence;
* Sphinx builds for both languages and StatsForecast contracts in CI;
* slow notebooks, including heavy ones, and downloads checked before publication;
* ``rc`` candidates routed to TestPyPI and final versions to PyPI.

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
baselines. Also review installation through TestPyPI, public contracts and
limitations found while using the candidate.

Composite families remain optional until they pass chain-level calibration. New
model families, exogenous inputs, probabilistic forecasts and global models are
not requirements for this release. Synthetic evidence does not replace the study
on real series. Demonstration notebooks do not constitute that study.

Publishing and installing the candidate
---------------------------------------

Merging the PR prepares the code; it does not publish packages. After review, tag
``v1.0.0rc1`` triggers ``release.yml`` on that same commit. The workflow validates
the tag, runs full CI and release tests, builds and checks both distributions, and
publishes to TestPyPI. A final tag such as ``v1.0.0`` is routed to PyPI.

Register a Trusted Publisher on each index for repository
``jvataidee/AutoHybTS``, workflow file ``release.yml`` and the corresponding
environment: ``testpypi`` or ``pypi``. Create those GitHub environments with required
reviewers. External configuration must be ready before tagging; the PR does not
create accounts, publishers or reviewers.

Once the candidate is published, test in a clean virtual environment:

.. code-block:: bash

   python -m pip install numpy pandas scikit-learn
   python -m pip install --index-url https://test.pypi.org/simple/ --no-deps autohybts==1.0.0rc1

Install dependencies from PyPI and only the package from TestPyPI; avoid combining
indices with ``--extra-index-url``. This test installs the core. Install
``statsmodels`` and ``matplotlib`` from PyPI to use statistical examples and plots.
Each uploaded version is immutable on its index; use ``1.0.0rc2`` if the candidate
needs a fix.

Local checklist
---------------

.. code-block:: bash

   ruff check .
   ruff format --check .
   mypy
   pytest -m "not slow" --cov
   pytest -m slow --run-network  # AUTOHYBTS_RUN_HEAVY=1; all tutorial extras
   pytest -m network --run-network
   sphinx-build -W documentation documentation/_build/html
   sphinx-build -W documentation/en documentation/_build/html-en
   python -m build --outdir dist/1.0.0rc1
   python -m twine check dist/1.0.0rc1/*
   python tools/check_release.py --tag v1.0.0rc1 --dist dist/1.0.0rc1

After installing the wheel in a clean environment without extras, run
``python tools/smoke_wheel.py`` using that environment's Python. CI also validates
dependency floors, Python 3.11–3.13, Windows, optional frameworks and auditing of
the dependencies selected by the ``audit`` job. Heavy and network notebooks run
in the tag workflow; per-PR CI does not execute this long-running work.
You can also run ``ci`` manually with ``release-checks=true`` before tagging,
without publishing. In that mode, missing notebook requirements fail rather than
skip a test.

References: `GitHub reusable workflows
<https://docs.github.com/en/actions/how-tos/reuse-automations/reuse-workflows>`_ and
`publishing with a Trusted Publisher
<https://docs.pypi.org/trusted-publishers/using-a-publisher/>`_.
