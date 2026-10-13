AutoHybTS
=========

.. image:: ../figures/AutoHybTS.png
   :alt: AutoHybTS
   :width: 220px
   :align: center

**Automated Hybrid Time Series Forecasting.** AutoHybTS é uma biblioteca Python de
código aberto que automatiza a comparação de modelos de previsão e a descoberta de
**sistemas híbridos residuais**: um modelo-base prevê a série e um corretor aprende o
erro que sobra. Complexidade extra só é aceita quando passa por critérios de validação
temporal definidos pelo usuário; caso contrário, o modelo mais simples continua campeão.

A interface segue a ergonomia do `PyCaret <https://pycaret.org>`_ — um experimento,
métodos curtos, resultados prontos para tabela e gráfico — e reúne num só fluxo as
estratégias de montar sistemas híbridos:

* **correção aditiva** ``ŷ = b + α·r̂`` (ex.: ``arima-mlp``);
* **vários estágios**, a correção da correção (``arima-mlp-linear``);
* **decomposição** em parte suave e volátil (``ma(linear-mlp)``, ``stl(...)``);
* **combinação fixa** de bases (``comb(arima,knn)``) ou de corretores
  (``arima-comb(mlp,knn)``);
* **seleção dinâmica** do corretor a cada passo, na previsão rolling
  (``dref(arima-mlp,knn)``).

.. code-block:: python

   from autohybts import AutoHybridExperiment
   from autohybts.datasets import get_data

   exp = AutoHybridExperiment(fh=14, session_id=42).fit(get_data("synthetic_daily"))
   result = exp.compare_models()
   result.decisions          # arima-mlp: aceito ou não, e por quê
   result.plot("pareto")     # erro × custo de cada estratégia
   final = exp.finalize_model(result.best)
   exp.evaluate_external(final).metrics

.. note::

   **Candidata à v1** (|release|): pré-versão para validação. Consulte
   :doc:`releasing` para o escopo, a compatibilidade e os critérios da versão estável.

.. toctree::
   :maxdepth: 1
   :caption: Primeiros passos

   installation
   quickstart

.. toctree::
   :maxdepth: 1
   :caption: Tutoriais

   tutorials/index

.. toctree::
   :maxdepth: 1
   :caption: Guia do usuário

   user_guide/temporal_validation
   user_guide/datasets
   user_guide/models
   user_guide/model_reference
   user_guide/nixtla
   user_guide/custom_models
   user_guide/search_space
   user_guide/strategies
   user_guide/ensembles
   user_guide/selection
   user_guide/comparison
   user_guide/panels
   user_guide/variant_b
   user_guide/tuning
   user_guide/persistence
   user_guide/benchmarks

.. toctree::
   :maxdepth: 1
   :caption: Referência

   api/index
   references

.. toctree::
   :maxdepth: 1
   :caption: Projeto

   contributing
   releasing
   about
