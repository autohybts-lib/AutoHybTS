AutoHybTS
=========

.. image:: ../../figures/AutoHybTS.png
   :alt: AutoHybTS
   :width: 220px
   :align: center

**Automated Hybrid Time Series Forecasting.** AutoHybTS is an open-source Python
library that automates the comparison of forecasting models and the discovery of
**residual hybrid systems**: a base model forecasts the series and a corrector learns
the error left over. Extra complexity is accepted only when it passes user-defined
temporal validation criteria; otherwise, the simpler model remains the champion.

The interface follows the ergonomics of `PyCaret <https://pycaret.org>`_ — one
experiment, short methods, results ready for tables and plots — and brings the
strategies for building hybrid systems together in a single workflow:

* **additive correction** ``ŷ = b + α·r̂`` (e.g. ``arima-mlp``);
* **multiple stages**, the correction of the correction (``arima-mlp-linear``);
* **decomposition** into a smooth and a volatile part (``ma(linear-mlp)``,
  ``stl(...)``);
* **fixed combination** of bases (``comb(arima,knn)``) or of correctors
  (``arima-comb(mlp,knn)``);
* **dynamic selection** of the corrector at each step, in rolling forecasting
  (``dref(arima-mlp,knn)``).

.. code-block:: python

   from autohybts import AutoHybridExperiment
   from autohybts.datasets import get_data

   exp = AutoHybridExperiment(fh=14, session_id=42).fit(get_data("synthetic_daily"))
   result = exp.compare_models()
   result.decisions          # arima-mlp: accepted or not, and why
   result.plot("pareto")     # error × cost of each strategy
   final = exp.finalize_model(result.best)
   exp.evaluate_external(final).metrics

.. note::

   **v1 release candidate** (|release|), published on PyPI: see :doc:`installation`.
   This is a prerelease for validation. See
   :doc:`releasing` for scope, compatibility and the stable-release criteria.

.. toctree::
   :maxdepth: 1
   :caption: Getting started

   installation
   quickstart
   concepts

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorials/index

.. toctree::
   :maxdepth: 1
   :caption: Guide: fundamentals

   user_guide/configuration
   user_guide/temporal_validation
   user_guide/variant_b
   user_guide/selection
   user_guide/diagnostics
   user_guide/troubleshooting

.. toctree::
   :maxdepth: 1
   :caption: Guide: data

   user_guide/datasets
   user_guide/panels
   user_guide/exogenous

.. toctree::
   :maxdepth: 1
   :caption: Guide: models

   user_guide/models
   user_guide/model_reference
   user_guide/nixtla
   user_guide/custom_models

.. toctree::
   :maxdepth: 1
   :caption: Guide: search

   user_guide/search_space
   user_guide/strategies
   user_guide/ensembles
   user_guide/tuning

.. toctree::
   :maxdepth: 1
   :caption: Guide: results and production

   user_guide/results
   user_guide/comparison
   user_guide/persistence
   user_guide/tracking
   user_guide/mlops

.. toctree::
   :maxdepth: 1
   :caption: Guide: evaluating the procedure

   user_guide/benchmarks

.. toctree::
   :maxdepth: 1
   :caption: Reference

   api/index
   references

.. toctree::
   :maxdepth: 1
   :caption: Project

   contributing
   maintainers
   releasing
