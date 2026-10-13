StatsForecast (Nixtla)
======================

Os modelos automáticos do `StatsForecast <https://nixtlaverse.nixtla.io/statsforecast/>`_
entram como modelos-base opcionais, com nomes próprios (prefixo ``sf_``) para não se
confundirem com o ``arima`` do statsmodels.

Instalação
----------

.. code-block:: bash

   python -m pip install "autohybts[nixtla-stats]==1.0.0rc2"

As versões atuais do StatsForecast (2.1) exigem ``pandas<3``: o extra instala o pandas
2.x, que a AutoHybTS suporta (piso 2.1). Num ambiente com pandas 3, mantenha um segundo
ambiente virtual para esses modelos. Sem o extra, uma candidata com eles fica
``SKIPPED`` e o motivo diz o que instalar.

Modelos
-------

.. list-table::
   :header-rows: 1
   :widths: 18 22 60

   * - Nome
     - Classe
     - O que escolhe sozinho
   * - ``sf_auto_arima``
     - ``AutoARIMA``
     - ordem (p, d, q) e, com ``seasonal``, (P, D, Q) por critério de informação
   * - ``sf_auto_ets``
     - ``AutoETS``
     - erro, tendência e sazonalidade (``model="ZZZ"``)
   * - ``sf_auto_theta``
     - ``AutoTheta``
     - variante do método Theta e decomposição sazonal
   * - ``sf_auto_ces``
     - ``AutoCES``
     - suavização exponencial complexa, com ou sem sazonalidade

Parâmetros: ``season_length`` e ``model_params`` (repassados à classe do StatsForecast).
Na busca, ``season_length`` vem do período sazonal do experimento; em ``create_model``,
passe-o explicitamente:

.. code-block:: python

   exp.create_model("sf_auto_ets", season_length=7)
   exp.create_model("sf_auto_arima", season_length=7,
                    model_params={"seasonal": True, "max_D": 1})
   exp.create_hybrid("sf_auto_theta", "mlp", base_params={"season_length": 7})

Parâmetros congelados e variantes A/B
-------------------------------------

Cada modelo é ajustado no vetor de treino da origem e depois congelado. A previsão chama
``forward(y=contexto, h)`` do StatsForecast, que aplica o modelo ajustado (ordens,
coeficientes, constantes de suavização) ao contexto observado sem reestimá-lo: no próprio
treino ela reproduz ``predict``; num contexto mais longo só os estados mudam. Por isso os
modelos funcionam na variante B, como bases de híbridos (os resíduos fora da amostra vêm
da mesma atualização) e passam nos testes de contrato e de causalidade da biblioteca.

Custo
-----

O ``AutoARIMA`` testa várias ordens a cada ajuste (alguns segundos numa série diária de
algumas centenas de pontos); ``AutoETS``, ``AutoTheta`` e ``AutoCES`` são bem mais
rápidos. Um híbrido ajusta a base duas vezes por origem (treino e prefixo dos resíduos).

Paridade com o projeto de replicação
------------------------------------

Os sete híbridos do projeto de replicação são reproduzidos exatamente pelas
especificações da AutoHybTS (teste ``tests/regression/test_legacy_parity.py``): quatro com
corretor KNN e três estatístico-ANN. Um ETS–ANN, por exemplo:

.. code-block:: python

   exp.create_hybrid(
       "sf_auto_ets", "mlp", window=7,
       base_params={"season_length": 7, "model_params": {"model": "ZZZ"}},
       corrector_params={
           "scale": "minmax", "feature_order": "lag", "random_state": 42,
           "estimator_params": {"hidden_layer_sizes": [5], "activation": "logistic",
                                "solver": "lbfgs", "max_iter": 2000},
       },
   )

``scale="minmax"`` (escala por coluna como o ``MinMaxScaler``) e ``feature_order="lag"``
(colunas da janela da defasagem 1 em diante) existem para essa paridade; o padrão da
biblioteca continua ``scale=True`` e ordem cronológica. A cauda de resíduos fixa do projeto
(``residual_tail``) corresponde a ``ResidualSettings(max_tail=...)`` ao montar o modelo
(``strategies.base.build_model``); o experimento usa a cauda padrão (metade do treino).

Limites
-------

* Só modelos locais e pontuais: intervalos de previsão do StatsForecast não são usados.
* MLForecast e NeuralForecast não estão integrados.
* Sem covariáveis (``target_mode="univariate"``).

.. seealso::

   Tutoriais: :doc:`../tutorials/notebooks/15_statsforecast`.
