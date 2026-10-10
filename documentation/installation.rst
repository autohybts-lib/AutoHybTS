Instalação
==========

AutoHybTS requer **Python 3.11** ou mais novo. O núcleo depende só de NumPy, pandas e
scikit-learn; o resto vem em extras opcionais.

.. code-block:: ipython

   !pip install "autohybts[stats,plot] @ git+https://github.com/jvataidee/AutoHybTS.git"

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
