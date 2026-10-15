Maintenance
===========

Tasks for whoever publishes the library and the documentation: PyPI versions, the tag
and the publishing workflows, the checklist before the tag and the Read the Docs
configuration. Contributors of code or documentation find what they need in
:doc:`contributing`.

Publishing and installing the candidate
---------------------------------------

Install the published version directly from PyPI:

.. code-block:: bash

   python -m pip install "autohybts[stats,plot]==1.0.0rc4"

The :doc:`installation` guide covers the core, extras, notebooks and the GitHub
development option.

The canonical ``release.yml`` workflow runs for ``v*`` tags, both RC and stable.
It verifies the tag/version match, runs the reusable CI with all release checks
(including tutorials and downloads), and only then builds and checks distributions.
Uploading to PyPI requires approval of the ``pypi`` environment. No parallel workflow
publishes tags and there is no automatic TestPyPI upload.

Before the next tag, update the PyPI Trusted Publisher to match:

* project: ``autohybts``;
* owner and repository: ``jvataidee/AutoHybTS``;
* workflow filename: ``release.yml`` (previously ``publish.yml``);
* environment: ``pypi``.

Keep required reviewers on the GitHub ``pypi`` environment. After validation, use
**Review deployments**, select **pypi** and **Approve and deploy**. This repository
change does not update the external PyPI registration or publish a version. For
validation without publication, dispatch ``ci`` with ``release-checks=true``.

Each version uploaded to PyPI is immutable, including the README embedded in its
metadata. Changes to the GitHub README and Read the Docs appear on PyPI only in
the next published version. ``1.0.0rc4`` includes this update; later changes require
another version. Do not re-upload an already published version.

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
   python -m build --outdir dist/1.0.0rc4
   python -m twine check dist/1.0.0rc4/*
   python tools/check_release.py --tag v1.0.0rc4 --dist dist/1.0.0rc4

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

Publishing on Read the Docs
---------------------------

The language configured in the dashboard sets the URL segment (``pt-br`` or ``en``),
but does not select the folder that Sphinx builds. Set the language and build
configuration file according to the same row in this table:

.. list-table::
   :header-rows: 1

   * - Dashboard language (Language)
     - Build configuration file
     - Sphinx configuration in the YAML
   * - Portuguese (Brazil)
     - ``documentation/.readthedocs.yaml``
     - ``documentation/conf.py``
   * - English
     - ``.readthedocs.yaml``
     - ``documentation/en/conf.py``

To publish the Portuguese documentation at
``https://autohybts.readthedocs.io/pt-br/latest/``:

1. Open the project that publishes Portuguese in the Read the Docs dashboard and
   go to **Admin → Settings** (or **Settings**).
2. Check the repository: ``https://github.com/jvataidee/AutoHybTS``.
3. Set **Language** to **Portuguese (Brazil)**.
4. Set **Build configuration file** to ``documentation/.readthedocs.yaml``.
   This field takes the YAML path; the ``conf.py`` path is already inside it.
5. Set **Default branch** to ``main`` and check that this branch contains the
   documentation files. Changes on another branch do not reach ``latest``;
   merge them into ``main`` for the final publication. For a preview, activate
   the desired branch under **Versions** and build that version.
6. Save and check that ``latest`` is active under **Versions**. Under **Builds**,
   trigger **Build version** for ``latest``.
7. Check the built commit in the log and that the Sphinx command uses
   ``documentation/conf.py``. Once the build succeeds, open the URL above.

Both YAML files already install the package with the ``docs`` extra on Python 3.12
and Ubuntu 24.04. All paths inside them are relative to the repository root,
including those in the YAML under ``documentation/``. Notebooks are copied during
the build and rendered with their saved outputs, without execution.

To offer both languages on the same domain, create one Read the Docs project per
language, using the same repository and the values in the table. Choose one as the
parent and add the other under **Translations** in the parent's dashboard. If
``autohybts`` remains the Portuguese parent, the English project can be registered
as its translation. If English is the parent, the Portuguese project is its
translation. Keep the same versions active in both projects and build both.

If English appears under ``/pt-br/``, check **Build configuration file**: the root
YAML builds English even when the dashboard language is Portuguese. If the content
is outdated, check the branch and commit in the last successful build.

References: `configuration file in a subfolder
<https://docs.readthedocs.com/platform/stable/guides/setup/monorepo.html>`_,
`languages and translations
<https://docs.readthedocs.com/platform/stable/localization.html>`_ and
`versions and branches
<https://docs.readthedocs.com/platform/stable/versions.html>`_.
