Referência dos modelos
======================

Para cada modelo registrado: os parâmetros que ele recebe sem configuração, o que a
busca automática passa para ele, o espaço que ``tune_model`` explora e onde ele roda.
As tabelas são geradas do código a cada build da documentação, então mostram o que a
versão instalada faz. A implementação de cada modelo (biblioteca de origem) está em
:doc:`models`.

Como ler
--------

* **★** marca os modelos da busca padrão (``compare_models`` sem ``base_models``).
* **Parâmetros padrão:** o que ``exp.create_model(nome)`` usa sem argumentos. Nos
  modelos sobre defasagens, ``estimator_params`` mostra o que a AutoHybTS muda em
  relação ao padrão da biblioteca (por exemplo, uma thread nos modelos de boosting).
* **Na busca:** o que ``compare_models`` passa ao modelo quando você não dá uma grade;
  o período sazonal e as defasagens vêm do experimento (aqui, série diária: 7).
* **Espaço do** ``tune_model``: distribuições de ``exp.tune_model(nome)``. ``[a, b]``
  é um intervalo (inteiro ou contínuo, ``(log)`` em escala logarítmica) e ``{...}``,
  as opções. Nos modelos sobre defasagens, ``lags`` vai de 2 a ``4 × período``
  (no mínimo 14, no máximo 60). "—" significa que o modelo escolhe sozinho (``arima``,
  StatsForecast) ou não tem o que ajustar.
* **Variantes:** A (vários passos de uma origem) e B (um passo, parâmetros congelados).

Mudar os parâmetros
-------------------

.. code-block:: python

   # um modelo, uma configuração
   exp.create_model("knn", lags=14, estimator_params={"n_neighbors": 3})

   # uma grade na busca: cada dicionário é uma candidata
   exp = AutoHybridExperiment(
       fh=14,
       base_models=["knn", "lightgbm_regressor"],
       base_params={"knn": [{"lags": 7}, {"lags": 14, "estimator_params": {"n_neighbors": 3}}]},
   )

   # ajuste em V_search, com o espaço embutido ou com o seu
   exp.tune_model("lightgbm_regressor", n_trials=30)
   exp.tune_model("knn", search_space={"lags": (7, 28), "estimator_params.n_neighbors": [3, 5, 9]})

Modelos-base
------------

.. include:: _generated/base_models.rst

Corretores de resíduos
----------------------

★ marca os corretores da busca padrão. Os corretores de **janela** regridem o próximo
resíduo nos ``window`` anteriores; os de **sequência** modelam a série de resíduos e
preveem o horizonte inteiro (:doc:`models`). Na busca, os parâmetros do corretor variam
por ``corrector_params`` (uma grade por nome), e ``tune_model`` ajusta só a base.

.. include:: _generated/correctors.rst

Opções comuns aos modelos sobre defasagens e aos corretores de janela:

* ``scale``: ``True`` (média e desvio da série), ``"columns"`` (por coluna, como o
  ``StandardScaler``), ``"minmax"`` (por coluna, para [0, 1]) ou ``False``;
* ``differencing`` (só bases): diferenças ajustadas no treino de cada origem;
* ``random_state``: semente; na busca, cada origem recebe uma semente derivada da
  semente do experimento, salvo se você a fixar;
* ``feature_order`` (só corretores): ``"chronological"`` ou ``"lag"``.
