Installation
============

AutoHybTS requires **Python 3.11** or newer.

From PyPI
---------

The `1.0.0rc1 candidate is published on PyPI
<https://pypi.org/project/autohybts/1.0.0rc1/>`_. Install with the extras used in the
statistical-model and plotting examples:

.. code-block:: bash

   python -m pip install "autohybts[stats,plot]==1.0.0rc1"

To install only the core (NumPy, pandas and scikit-learn):

.. code-block:: bash

   python -m pip install "autohybts==1.0.0rc1"

These commands pin the candidate; ``--pre`` is not needed for an explicitly
requested prerelease version. To get the latest version including candidates,
use ``python -m pip install --upgrade --pre "autohybts[stats,plot]"``. Publication
on PyPI preserves its prerelease status; see :doc:`releasing`.

Check the installation:

.. code-block:: bash

   python -c "import autohybts; print(autohybts.__version__)"

Notebooks
---------

In a Jupyter cell, ``%pip`` installs into the current kernel's environment:

.. code-block:: ipython

   %pip install "autohybts[stats,plot,notebook]==1.0.0rc1"

Extras
------

Combine extras inside brackets and keep the version. For example:

.. code-block:: bash

   python -m pip install "autohybts[stats,plot,torch]==1.0.0rc1"

.. list-table::
   :header-rows: 1

   * - Extra
     - Purpose
   * - ``stats``
     - ARIMA, ETS and smoothing with statsmodels
   * - ``plot``
     - Plots with matplotlib
   * - ``notebook``
     - Kernel and execution of Jupyter tutorials
   * - ``torch`` / ``tensorflow``
     - Custom models using these frameworks
   * - ``boosting``
     - LightGBM, XGBoost and CatBoost
   * - ``nixtla-stats``
     - StatsForecast; requires pandas 2.x (see :doc:`user_guide/nixtla`)
   * - ``optuna``
     - Hyperparameter search with Optuna
   * - ``datasets``
     - Reading the M3 benchmark
   * - ``yaml``
     - PyYAML for working with YAML configurations

Development
-----------

To test code from the ``main`` branch before its next publication, install from
GitHub (requires Git):

.. code-block:: bash

   python -m pip install "autohybts[stats,plot] @ git+https://github.com/jvataidee/AutoHybTS.git"

To contribute from a local clone, use
``python -m pip install -e ".[dev,stats,notebook]"`` at the repository root.

References: `pip prereleases
<https://pip.pypa.io/en/stable/cli/pip_install/#pre-release-versions>`_ and
`IPython %pip command
<https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-pip>`_.
