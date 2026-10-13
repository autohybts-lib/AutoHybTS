Modelos e bibliotecas
=====================

A maior parte dos modelos é um adaptador fino sobre uma biblioteca conhecida
(scikit-learn, statsmodels, StatsForecast, LightGBM ...): o adaptador cuida do que é
específico de séries temporais (janelas de lags, diferenciação, previsão recursiva,
contexto temporal) e delega o ajuste à biblioteca de origem. Métodos simples sem uma
implementação de referência adequada — baselines, regressão harmônica e os métodos de
demanda intermitente — são implementados em numpy na própria AutoHybTS e marcados como
"próprio (numpy)" nas tabelas. O nome curto é o que se passa para ``create_model`` e
``create_hybrid``, e é o que aparece no leaderboard.

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

Esses nove formam a busca padrão. Os modelos abaixo estão registrados, mas só entram
quando pedidos (``base_models=[...]``, ``create_model``): controles simples para
comparar estruturas compostas e boosting sobre lags.

.. list-table::
   :header-rows: 1
   :widths: 22 12 38 28

   * - Nome
     - Família
     - Implementação
     - Instalação
   * - ``historic_average``
     - statistical
     - próprio (numpy): média do treino, congelada
     - núcleo
   * - ``window_average``
     - statistical
     - próprio (numpy): média das últimas ``window`` observações
     - núcleo
   * - ``seasonal_window_average``
     - statistical
     - próprio (numpy): média de cada posição sazonal nas últimas ``window``
       temporadas
     - núcleo
   * - ``random_walk_drift``
     - statistical
     - próprio (numpy): último valor + ``h`` · variação média do treino
     - núcleo
   * - ``exponential_smoothing``
     - statistical
     - ``statsmodels.tsa.holtwinters.ExponentialSmoothing`` aditiva (Holt, Holt-Winters,
       amortecida: ``trend``, ``damped``, ``season_length``); parâmetros estimados no treino
       e congelados, estados atualizados pelo contexto
     - extra ``stats`` (statsmodels)
   * - ``sarima``
     - statistical
     - ``statsmodels.tsa.arima.model.ARIMA`` com parte sazonal ``(P, D, Q, m)``: ``D`` pela
       força sazonal, ``d`` por KPSS e ``(p, q, P, Q)`` por AICc em busca *stepwise*
       (:doc:`diagnostics`)
     - extra ``stats`` (statsmodels)
   * - ``harmonic_regression``
     - statistical
     - próprio (numpy): regressão harmônica dinâmica, com tendência, pares de Fourier
       para cada período de ``periods`` (aceita não inteiros, como 365,25) e
       ``ar_lags`` defasagens; coeficientes congelados. Representa séries com várias
       sazonalidades (semanal + anual em dados diários; diária + semanal em horários)
       sem um estado por posição sazonal. Padrão para dados diários:
       ``periods=(7, 365.25)``, ``harmonics=(3, 5)``, ``ar_lags=1``
     - núcleo
   * - ``ridge_regression``, ``lasso_regression``, ``elastic_net_regression``
     - linear
     - ``sklearn.linear_model.Ridge`` / ``Lasso`` / ``ElasticNet`` sobre lags
     - núcleo (scikit-learn)
   * - ``extra_trees_regressor``
     - tree
     - ``sklearn.ensemble.ExtraTreesRegressor`` sobre lags
     - núcleo (scikit-learn)
   * - ``hist_gradient_boosting``
     - tree
     - ``sklearn.ensemble.HistGradientBoostingRegressor`` sobre lags
     - núcleo (scikit-learn)
   * - ``adaboost_r2``
     - tree
     - ``sklearn.ensemble.AdaBoostRegressor`` (AdaBoost.R2) sobre lags
     - núcleo (scikit-learn)
   * - ``lightgbm_regressor``
     - tree
     - ``lightgbm.LGBMRegressor`` sobre lags (200 árvores, 15 folhas, ``n_jobs=1``)
     - extra ``boosting`` (lightgbm)
   * - ``xgboost_regressor``
     - tree
     - ``xgboost.XGBRegressor`` sobre lags (200 árvores, profundidade 4, ``n_jobs=1``)
     - extra ``boosting`` (xgboost)
   * - ``catboost_regressor``
     - tree
     - ``catboost.CatBoostRegressor`` sobre lags (200 iterações, profundidade 4,
       ``thread_count=1``, sem gravar ``catboost_info/``)
     - extra ``boosting`` (catboost)

Os modelos automáticos do StatsForecast (``sf_auto_arima``, ``sf_auto_ets``,
``sf_auto_theta``, ``sf_auto_ces``; extra ``nixtla-stats``, que traz pandas 2.x) têm página
própria: :doc:`nixtla`.

Os modelos de boosting rodam com uma thread por modelo: em séries de poucas centenas de
pontos, as threads do OpenMP custam mais do que economizam. O paralelismo do
experimento continua entre candidatas (``n_jobs``).

Os modelos do scikit-learn viram previsores de séries pelo mesmo adaptador
(``LagRegressionForecaster``): aprendem ``y[t]`` a partir de ``y[t-1], …, y[t-lags]``,
com diferenciação e padronização opcionais ajustadas só no treino, e preveem vários
passos de forma recursiva. Os hiperparâmetros do estimador vão em ``estimator_params``:

.. code-block:: python

   exp.create_model("knn", lags=14, estimator_params={"n_neighbors": 3})

A padronização (``scale``) é ajustada no treino: ``True`` (padrão) usa uma média e um
desvio da série; ``"columns"`` padroniza cada defasagem e o alvo separadamente, como o
``StandardScaler`` do scikit-learn em X e y; ``False`` não padroniza. Modelos lineares e
árvores não mudam com a escolha; KNN e SVR mudam, porque dependem das distâncias. Os
corretores aceitam as mesmas opções. Com ``scale="columns"``, a AutoHybTS reproduz
exatamente os quatro híbridos com corretor KNN do projeto de replicação.

Demanda intermitente
--------------------

Para séries com muitos zeros (vendas esparsas, peças de reposição), quatro métodos
formam a família ``intermittent``. São modelos-base, não combinações de candidatas:

.. list-table::
   :header-rows: 1
   :widths: 16 56 28

   * - Nome
     - Previsão (constante no horizonte)
     - Parâmetros
   * - ``croston_sba``
     - Croston com a correção de Syntetos-Boylan: ``(1 − α/2) · z / p``, com ``z`` o
       tamanho das demandas e ``p`` o intervalo entre elas, suavizados separadamente
     - ``alpha=0.1``
   * - ``tsb``
     - Teunter-Syntetos-Babai: probabilidade de demanda (suavizada todo período) ×
       tamanho (suavizado quando há demanda); decai quando a demanda para
     - ``alpha_d=0.1``, ``alpha_p=0.1``
   * - ``adida``
     - soma em blocos de ``k`` períodos (``k`` = intervalo médio entre demandas no
       treino, congelado), suavização dos blocos e divisão por ``k``
     - ``alpha=0.1``
   * - ``imapa``
     - média das previsões do ADIDA nos níveis ``1..k``
     - ``alpha=0.1``

Todos são implementados em numpy, rodam nas variantes A e B (o nível de agregação é
aprendido no treino; as recursões leem o contexto) e nunca preveem valores negativos.
Uma série com valores negativos não é demanda: o ajuste é recusado e a tentativa fica
``SKIPPED``. Os diagnósticos de cada tentativa trazem o perfil da série no treino:
fração de zeros, intervalo médio entre demandas (ADI) e CV² dos tamanhos.

A família entra na cadeia de seleção só quando um desses modelos está em
``base_models``, logo antes dos híbridos residuais; sem eles, o protocolo padrão não
muda. Em demanda intermitente, prefira ``metrics=["RMSSE", ...]``: métricas baseadas no
erro absoluto (MASE, MAE) favorecem prever zero.

.. code-block:: python

   y = get_data("intermittent_daily")
   exp = AutoHybridExperiment(
       fh=14, metrics=["RMSSE", "MASE"], base_models=["naive", "croston_sba", "tsb", "imapa"]
   ).fit(y)

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

Corretores opcionais (fora da grade padrão; use ``correctors=[...]`` ou
``create_hybrid``): ``svm``, ``tree``, ``random_forest``, ``hist_gradient_boosting``,
``adaboost_r2``, ``lightgbm_regressor``, ``xgboost_regressor`` e ``catboost_regressor``
(os três últimos no extra ``boosting``), com as mesmas implementações dos modelos-base,
agora sobre janelas de resíduos.

Corretores de sequência residual
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Um corretor de janelas regride o próximo resíduo nos ``window`` anteriores e é iterado
passo a passo. Um corretor de **sequência** é um modelo de série temporal da própria
série de resíduos: é ajustado uma vez nos resíduos fora da amostra contíguos do treino e
prevê o vetor residual do horizonte inteiro a partir do histórico residual disponível na
origem. Modelos de estado (ARIMA, suavização exponencial) precisam do histórico residual
completo para os seus estados, o que uma janela fixa não dá.

.. list-table::
   :header-rows: 1
   :widths: 30 46 24

   * - Nome
     - Modelo sobre os resíduos
     - Instalação
   * - ``arima_sequence``
     - ``arima`` (ordem automática), parâmetros congelados
     - extra ``stats``
   * - ``exponential_smoothing_sequence``
     - ``exponential_smoothing`` sem tendência (suavização simples)
     - extra ``stats``

.. code-block:: python

   exp.create_hybrid("seasonal_naive", "arima_sequence")
   # parâmetros do modelo sobre os resíduos:
   exp.create_hybrid("linear", "arima_sequence",
                     corrector_params={"model_params": {"order": (1, 0, 0)}})

Funcionam no híbrido aditivo (e em estágios), nas variantes A e B: na B, novas
observações estendem a série de resíduos sem reajuste. A janela não se aplica a eles:
na busca, cada um é proposto com a primeira janela de ``residual_windows`` só. Estratégias
que exigem um corretor de janelas (decomposição, combinação de corretores, seleção
dinâmica) não os propõem, e uma candidata montada à mão com eles fica ``SKIPPED``.

Capacidades
-----------

``exp.models()`` mostra o que cada modelo declara. Além de ``multi_step`` e
``online_state_update`` (variantes A e B), ``training_scope`` (``local``, ``global``,
``pretrained``), ``output_type`` (``point``, ``quantile``, ``samples``,
``distribution``), ``target_mode`` (``univariate``, ``multivariate``, ``exogenous``),
``horizon_strategy`` e ``max_horizon`` descrevem o modelo. A busca é univariada, local,
pontual e sem covariáveis: um modelo treinado globalmente num painel, com saída não
pontual, que prevê vários alvos juntos, que precisa de covariáveis ou com horizonte
máximo menor que o pedido é ``SKIPPED`` sem ser ajustado.

Métricas
--------

MASE (padrão de ordenação), MAE, RMSE, sMAPE, MAPE, MdAPE, WAPE, RRMSE, RMAE (``MAE / média do
observado``, como no protocolo de replicação) e RMSSE (raiz do erro quadrático médio
escalado pela diferença sazonal ao quadrado do treino, com o mesmo ``m`` da MASE; o M5
usa ``m = 1``). Escolha com
``metrics=[...]``; a primeira ordena e seleciona. Fórmulas e casos indefinidos em
:doc:`results`.

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

.. seealso::

   Tutoriais: :doc:`../tutorials/notebooks/12_baselines_and_boosting`, :doc:`../tutorials/notebooks/14_intermittent_demand`.
