Modelos e bibliotecas
=====================

A AutoHybTS não implementa algoritmos de previsão próprios: cada modelo é um adaptador
fino sobre uma biblioteca conhecida. O adaptador cuida do que é específico de séries
temporais (janelas de lags, diferenciação, previsão recursiva, contexto temporal) e
delega o ajuste à biblioteca de origem. O nome curto é o que se passa para
``create_model``, ``create_hybrid`` e aparece no leaderboard.

.. code-block:: python

   exp.models()     # nome, papel (base/corretor), família, capacidades e extra necessário

Modelos-base
------------

.. list-table::
   :header-rows: 1
   :widths: 16 12 40 32

   * - Nome
     - Família
     - Implementação
     - Instalação
   * - ``naive``
     - statistical
     - próprio (numpy): repete o último valor
     - núcleo
   * - ``seasonal_naive``
     - statistical
     - próprio (numpy): repete a última temporada
     - núcleo
   * - ``arima``
     - statistical
     - ``statsmodels.tsa.arima.model.ARIMA``; ordem escolhida com o teste KPSS
       (``statsmodels.tsa.stattools.kpss``) e o AIC
     - extra ``stats`` (statsmodels)
   * - ``linear``
     - linear
     - ``sklearn.linear_model.LinearRegression`` sobre lags
     - núcleo (scikit-learn)
   * - ``knn``
     - kernel
     - ``sklearn.neighbors.KNeighborsRegressor`` sobre lags
     - núcleo (scikit-learn)
   * - ``svm``
     - kernel
     - ``sklearn.svm.SVR`` sobre lags
     - núcleo (scikit-learn)
   * - ``tree``
     - tree
     - ``sklearn.tree.DecisionTreeRegressor`` sobre lags
     - núcleo (scikit-learn)
   * - ``random_forest``
     - tree
     - ``sklearn.ensemble.RandomForestRegressor`` sobre lags
     - núcleo (scikit-learn)
   * - ``mlp``
     - neural
     - ``sklearn.neural_network.MLPRegressor`` sobre lags (``hidden_layer_sizes=(32,)``,
       ``max_iter=1000``)
     - núcleo (scikit-learn)

Os modelos do scikit-learn viram previsores de séries pelo mesmo adaptador
(``LagRegressionForecaster``): aprendem ``y[t]`` a partir de ``y[t-1], …, y[t-lags]``,
com diferenciação e padronização opcionais ajustadas só no treino, e preveem vários
passos de forma recursiva. Os hiperparâmetros do estimador vão em ``estimator_params``:

.. code-block:: python

   exp.create_model("knn", lags=14, estimator_params={"n_neighbors": 3})

Corretores de resíduos
----------------------

.. list-table::
   :header-rows: 1
   :widths: 16 52 32

   * - Nome
     - Implementação
     - Instalação
   * - ``linear``
     - ``sklearn.linear_model.LinearRegression`` sobre janelas de resíduos
     - núcleo (scikit-learn)
   * - ``knn``
     - ``sklearn.neighbors.KNeighborsRegressor`` sobre janelas de resíduos
     - núcleo (scikit-learn)
   * - ``mlp``
     - ``sklearn.neural_network.MLPRegressor`` sobre janelas de resíduos
     - núcleo (scikit-learn)

Outras dependências
-------------------

Partes fora dos modelos também usam bibliotecas externas, sempre importadas só na hora
do uso (``import autohybts`` carrega apenas numpy, pandas e o próprio pacote):

.. list-table::
   :header-rows: 1
   :widths: 34 40 26

   * - Recurso
     - Biblioteca
     - Instalação
   * - decomposição ``stl(...)``
     - ``statsmodels.tsa.seasonal.STL``
     - extra ``stats``
   * - testes estatísticos na comparação e na seleção
     - ``scipy.stats`` (vem com o scikit-learn)
     - núcleo
   * - ``tune_model(sampler="optuna")``
     - ``optuna`` (TPE); com ``sampler="auto"``, sem Optuna instalado, a busca é
       aleatória com semente
     - extra ``optuna``
   * - gráficos
     - ``matplotlib``
     - extra ``plot``
   * - leitura do M3
     - ``rdata``
     - extra ``datasets``

Sem o extra necessário, o modelo ou recurso falha com ``UnsupportedCapabilityError``
dizendo o que instalar; na busca automática, a candidata vira ``SKIPPED`` e as outras
seguem. A origem dos *dados* (Rdatasets, M4, M3) está em :doc:`datasets`.

Modelos próprios
----------------

Modelos de PyTorch, TensorFlow/Keras ou de qualquer outra biblioteca entram por registro
(``exp.register_model``) e passam a valer em todo o experimento: veja
:doc:`custom_models`.
