Version 1 and publishing
========================================

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

Publishing and installing the candidate
---------------------------------------

Install the published version directly from PyPI:

.. code-block:: bash

   python -m pip install "autohybts[stats,plot]==1.0.0rc1"

The :doc:`installation` guide covers the core, extras, notebooks and the GitHub
development option.

The ``publish.yml`` workflow runs when a ``v*`` tag is pushed. It builds the wheel
and source distribution, checks them with Twine and waits for approval of the
``pypi`` environment before uploading. The Trusted Publisher registered on PyPI
must match:

* project: ``autohybts``;
* owner and repository: ``jvataidee/AutoHybTS``;
* workflow filename: ``publish.yml``;
* environment: ``pypi``.

On GitHub, open the run, use **Review deployments**, select **pypi** and click
**Approve and deploy**. Registering a publisher does not start an upload; pushing
the tag and approving the environment are steps in the workflow.

``release.yml`` retains a separate validation and TestPyPI workflow, which depends
on its own Trusted Publisher. For future versions, run complete checks before
tagging and choose a single publishing workflow for each index.

Each version uploaded to PyPI is immutable, including the README embedded in its
metadata. Changes to the GitHub README and Read the Docs appear on PyPI only in
the next published version, such as ``1.0.0rc2``. Do not re-upload ``1.0.0rc1``.

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
