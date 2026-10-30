Contributing
============

How to add a model, a strategy or a tutorial without losing the library's guarantees:
causality, an untouched sealed test and auditable decisions.

Tutorials (notebooks)
---------------------

The tutorials in ``notebooks/`` (Portuguese) and ``notebooks/en/`` (English) are the
executable contract of the public API and are kept in the repository **with their
outputs**, so they can be read without running them.

Lesson model
~~~~~~~~~~~~

A new tutorial follows this order:

1. **Goal and requirements**: topic, scientific reference, related issue or PR, supported A/B
   variants, dependencies.
2. **Data and protocol**: source, frequency, ``fh``, ``test_size``, ``fold``, seeds and
   budget. The search never touches the sealed test.
3. **Causal exploration**: series, seasonality and scales, using only what exists at
   the origin.
4. **Controls and proposed models**: same metric, same horizon, same origins; the best
   standalone model and a simple combination are the minimum controls.
5. **Search and selection**: public API, ``V_search`` for the search, ``V_select`` for
   the decision, cost in candidates, units and fits.
6. **Results and plots**: forecasts against observations, errors, weights, gains,
   acceptances and rejections.
7. **Diagnostics and limits**: leakage risks, coverage, failures, cost.
8. **External evaluation**: a single reading of the sealed test, after freezing the
   choice.
9. **Reading**: written after the run, with its numbers; what won, against which
   controls, with how many units and with which limits.

The skeleton comes ready from the generator (``--lang en`` writes the English version
to ``notebooks/en/``):

.. code-block:: bash

   python notebooks/new_notebook.py 16_temporal_bagging "Temporal bagging" --lang en \
       --guide user_guide/ensembles --extras plot \
       --requires autohybts.AutoHybridExperiment.compare_models

Metadata
~~~~~~~~

Each notebook declares ``metadata.autohybts`` with these fields, checked by
``tests/contracts/test_notebook_metadata.py``:

.. list-table::
   :header-rows: 1
   :widths: 14 86

   * - Field
     - Meaning
   * - ``requires``
     - public names the notebook calls (``autohybts.AutoHybridExperiment.create_model``
       ...); while any of them is missing, the execution test skips the notebook
   * - ``network``
     - downloads data on the first run (tests only run it with ``--run-network``)
   * - ``heavy``
     - takes many minutes; tests only run it with ``AUTOHYBTS_RUN_HEAVY=1``
   * - ``extras``
     - ``pyproject.toml`` extras it needs (``plot``, ``stats``, ``boosting`` ...);
       without them the test skips and says which one is missing; a skip never counts
       as a run
   * - ``seeds``
     - seeds that make the run reproducible
   * - ``guide``
     - pages of this documentation the notebook illustrates (they must exist)
   * - ``template``
     - follows the lesson model: opens with a title and ends with the reading section
       ("Leitura" in Portuguese, "Reading" in English)

Execution and outputs
~~~~~~~~~~~~~~~~~~~~~

Run the notebook from its folder (``nbclient`` with
``resources={"metadata": {"path": "notebooks"}}``, or ``"notebooks/en"`` for the
English ones) and save the outputs. Before saving, check figures and tables; the
outputs must not contain local paths (``tests/contracts/test_notebook_outputs.py``),
keys, private data or large artifacts. The documentation build does not execute
notebooks: it shows the saved outputs.

Language of the code
--------------------

Code, names and docstrings in English; error messages, warnings and texts shown to the
user in Portuguese. The API reference shows the English docstrings in both languages
of the documentation.

Translations
------------

The documentation exists in Portuguese (``documentation/``) and English
(``documentation/en/``), with the same file names page by page; the tutorials, in
``notebooks/`` and ``notebooks/en/``. A change to a page or notebook goes into both
languages in the same PR. ``tests/contracts/test_docs_translation.py`` checks that the
two trees have the same pages and the same notebooks with the same metadata. Build both
before opening the PR:

.. code-block:: bash

   sphinx-build -W documentation documentation/_build/html
   sphinx-build -W documentation/en documentation/_build/html-en

Environment and tests
---------------------

.. code-block:: bash

   python -m venv .venv
   .venv/bin/python -m pip install -e ".[dev,stats,notebook]"   # Windows: .venv\Scripts\python
   ruff check . && ruff format --check .
   mypy                                  # strict, the whole package
   pytest -m "not slow" --cov            # minimum coverage: 85 %
   pytest -m leakage                     # one suite

* The suites (``unit``, ``integration``, ``contracts``, ``leakage``, ``regression``) come
  from the folder under ``tests/``. ``slow`` stays out of each PR's CI; ``network`` runs
  only with ``--run-network``; ``heavy`` notebooks only with ``AUTOHYBTS_RUN_HEAVY=1``.
* ``import autohybts`` must not load statsmodels, torch, optuna, matplotlib, plotly, yaml
  or scikit-learn (``tests/contracts/test_import_minimal.py``): import heavy libraries
  inside functions.
* StatsForecast models require pandas < 3: test them in a second environment with the
  ``nixtla-stats`` extra.

Adding a built-in model
-----------------------

1. Subclass ``ForecasterBase`` (``autohybts.adapters.base``) and store every ``__init__``
   argument in an attribute of the same name: that is how ``get_params`` and ``clone``
   work.
2. Implement ``_fit(y)`` and ``_predict(horizon, history)``. The forecast starts from the
   observed context with frozen parameters: never refit inside the forecast (variant B
   and out-of-sample residuals require it).
3. Import the library inside the methods and declare the extra in
   ``Capabilities(requires_extra=...)``.
4. Register it in ``registry.default_registry()`` with a descriptive name; short names
   collide with models registered by users (check notebooks and tests).
5. The contract tests (``tests/contracts/test_base_forecaster.py``) and causality tests
   (``tests/leakage/test_base_causality.py``) cover every registered model
   automatically; if the defaults are slow, add fast parameters to ``FAST_PARAMS``.
6. Keep the model out of the default search until it is measured and describe it in
   :doc:`user_guide/models` (the tables of :doc:`user_guide/model_reference` are
   generated).

Documenting a strategy
----------------------

The guide page of a new strategy has:

* definition and formula or pseudocode;
* scientific reference and how it differs from similar approaches;
* parameters and the space the search explores;
* compatible models and data, variants A and B, horizon;
* how it is fitted, where the out-of-sample residuals or forecasts come from and when
  the weights change;
* cost, behaviour on failures and known limits;
* a runnable example with a real result, never an invented one;
* links to the notebook, the causality test and the benchmarks section.

AI-assisted contributions
-------------------------

Contributions made with AI assistants are welcome on the same terms as any other:

* whoever sends it reviews and understands every line, and answers for it in the pull
  request;
* the change goes through the same checks (``ruff``, ``mypy``, contract, causality and
  leakage tests) and no invariant is loosened to fit generated code: untouched sealed test,
  causal residuals, stable identities;
* the pull request description says what was done with assistance and how it was checked;
* private data, credentials and keys never go to an AI service.

Delivery
--------

A new strategy is ready when:

* the algorithm is registered with explicit capabilities and incompatibilities;
* it passes the unit, temporal contract, integration and leakage tests;
* it was compared with the best standalone model, the corresponding hybrid and a
  simple combination, on the same origins;
* it has a guide page, a notebook with reviewed outputs, API reference and updated
  navigation, in both languages, and the documentation builds without warnings
  (``sphinx-build -W documentation documentation/_build/html`` and the same for
  ``documentation/en``);
* costs, versions, seeds and results are recorded, and the decisions (acceptances and
  rejections, with reasons) show up in the reports;
* new composite families stay out of the default ``include`` until they pass the
  calibration gate (``autohybts.benchmarks.calibration``).
