Instalação
==========

AutoHybTS requer **Python 3.11** ou mais novo.

Pelo PyPI
---------

A candidata `1.0.0rc2 está publicada no PyPI
<https://pypi.org/project/autohybts/1.0.0rc2/>`_. Instale com os extras usados nos
exemplos de modelos estatísticos e gráficos:

.. code-block:: bash

   python -m pip install "autohybts[stats,plot]==1.0.0rc2"

Para instalar apenas o núcleo (NumPy, pandas e scikit-learn):

.. code-block:: bash

   python -m pip install "autohybts==1.0.0rc2"

Os comandos fixam a candidata; não é necessário adicionar ``--pre`` a uma versão
explicitamente solicitada. Para buscar a versão mais recente incluindo candidatas,
use ``python -m pip install --upgrade --pre "autohybts[stats,plot]"``. A publicação
no PyPI mantém o status de pré-versão; veja :doc:`releasing`.

Confira a instalação:

.. code-block:: bash

   python -c "import autohybts; print(autohybts.__version__)"

Notebooks
---------

Em uma célula Jupyter, ``%pip`` instala no ambiente do kernel atual:

.. code-block:: ipython

   %pip install "autohybts[stats,plot,notebook]==1.0.0rc2"

Extras
------

Combine os extras entre colchetes, preservando a versão. Por exemplo:

.. code-block:: bash

   python -m pip install "autohybts[stats,plot,torch]==1.0.0rc2"

.. list-table::
   :header-rows: 1

   * - Extra
     - Uso
   * - ``stats``
     - ARIMA, ETS e suavização com statsmodels
   * - ``plot``
     - Gráficos com matplotlib
   * - ``notebook``
     - Kernel e execução dos tutoriais Jupyter
   * - ``torch`` / ``tensorflow``
     - Modelos próprios nesses frameworks
   * - ``boosting``
     - LightGBM, XGBoost e CatBoost
   * - ``nixtla-stats``
     - StatsForecast; requer pandas 2.x (veja :doc:`user_guide/nixtla`)
   * - ``optuna``
     - Busca de hiperparâmetros com Optuna
   * - ``datasets``
     - Leitura do benchmark M3
   * - ``yaml``
     - PyYAML; reservado: nenhum recurso desta versão depende dele

Desenvolvimento
---------------

Para testar o código da branch ``main`` antes de sua próxima publicação, instale
do GitHub (requer Git):

.. code-block:: bash

   python -m pip install "autohybts[stats,plot] @ git+https://github.com/jvataidee/AutoHybTS.git"

Para contribuir a partir de um clone local, use
``python -m pip install -e ".[dev,stats,notebook]"`` na raiz do repositório.

Referências: `pré-versões no pip
<https://pip.pypa.io/en/stable/cli/pip_install/#pre-release-versions>`_ e
`comando %pip do IPython
<https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-pip>`_.
