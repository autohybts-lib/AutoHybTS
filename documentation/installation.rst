Instalação
==========

AutoHybTS requer **Python 3.11** ou mais novo. O núcleo depende só de NumPy, pandas e
scikit-learn; o resto vem em extras opcionais.

Pelo pip
--------

Quando a versão estiver publicada no PyPI:

.. code-block:: bash

   pip install autohybts
   pip install "autohybts[stats,plot]"          # ARIMA/STL e gráficos
   pip install "autohybts[stats,plot,optuna]"   # + tune_model com Optuna

Pelo git
--------

Para a versão em desenvolvimento, direto do repositório:

.. code-block:: bash

   pip install "autohybts[stats,plot] @ git+https://github.com/jvataidee/AutoHybTS.git"

   # uma tag ou um commit específico
   pip install "autohybts[stats,plot] @ git+https://github.com/jvataidee/AutoHybTS.git@v0.1.0"

A partir do código-fonte
------------------------

.. code-block:: bash

   git clone https://github.com/jvataidee/AutoHybTS.git
   cd AutoHybTS
   python -m venv .venv
   .venv/Scripts/python -m pip install -e ".[dev,stats,notebook]"   # Windows
   .venv/bin/python -m pip install -e ".[dev,stats,notebook]"       # Linux/macOS

Extras
------

==============  =====================================================================
Extra           Para quê
==============  =====================================================================
``stats``       statsmodels: ARIMA e decomposição STL
``plot``        matplotlib: todos os gráficos (``plot_data``, ``plot_model``, Pareto …)
``optuna``      sampler TPE do ``tune_model`` (sem ele, busca aleatória com semente)
``datasets``    ler o benchmark M3 (arquivo R do pacote Mcomp)
``yaml``        configuração em YAML
``notebook``    rodar os tutoriais (ipykernel, nbclient)
``docs``        gerar esta documentação (Sphinx, Furo, myst-nb)
``dev``         testes, lint e tipos
==============  =====================================================================

Sem um extra, o recurso correspondente levanta ``UnsupportedCapabilityError`` com o
comando de instalação; ``import autohybts`` nunca carrega statsmodels, matplotlib ou
Optuna.

Gerar esta documentação
-----------------------

.. code-block:: bash

   pip install -e ".[docs]"
   sphinx-build documentation documentation/_build/html
