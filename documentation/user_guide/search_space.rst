Espaço de busca e orçamento
===========================

``compare_models`` propõe candidatas em etapas, avalia cada uma em todas as origens de
``V_search`` e leva a melhor de cada família para a seleção em ``V_select``
(:doc:`selection`). Esta página diz o que é proposto, quanto o orçamento deixa avaliar e
como abrir ou fechar o espaço. Os modelos e seus parâmetros estão em
:doc:`model_reference`.

Etapas
------

``S0`` — modelos-base
   cada base de ``base_models`` com os parâmetros da busca, ou cada item da grade
   ``base_params`` dela.
``S1`` — retenção
   as duas melhores configurações de cada família, o ``arima`` e o melhor modelo
   individual seguem como bases dos híbridos (sem duplicatas).
``S2`` — híbridos residuais
   cada base retida × corretor × janela × α × fonte de resíduos, em largura: todas as
   bases recebem a primeira combinação antes da segunda, então um corte de orçamento
   mantém as bases variadas.
``S3`` — refinamento
   em torno das duas melhores estruturas residuais, um eixo de cada vez: outra fonte de
   resíduos, as alternativas de ``corrector_params`` e, com ``"multistage"`` no
   ``include``, estágios extras.
``S4`` — famílias avançadas
   só com ``include``: decomposição, combinação, seleção dinâmica (variante B) e
   ensembles, no orçamento que sobrou, com uma vaga reservada por família.

Padrões
-------

.. include:: _generated/search_defaults.rst

Orçamento
---------

O orçamento conta **candidatas**: uma candidata é avaliada em todas as origens de
``V_search``. Os limites valem para a lista ordenada de propostas antes de cada etapa,
então a mesma série e a mesma semente dão a mesma busca em qualquer máquina, com
``n_jobs`` sequencial ou paralelo. As propostas cortadas aparecem no ledger como
``PRUNED_BUDGET``.

.. include:: _generated/budgets.rst

``max_trials`` substitui o total; ``max_wall_time_s`` acrescenta um prazo de relógio
opcional (:doc:`strategies`). O número de origens de cada camada é ``fold``, não o
orçamento.

Exemplo: ``smoke`` e ``small``
------------------------------

Série ``synthetic_daily`` (730 dias), ``fh=14``, ``session_id=42``, demais padrões
(``fold="auto"`` usou 7 origens por camada). Contagem do ledger por etapa e situação:

.. list-table::
   :header-rows: 1
   :widths: 14 12 16 18 20 20

   * - Orçamento
     - Etapa
     - Avaliadas
     - Cortadas pelo orçamento
     - Puladas por capacidade
     - Propostas
   * - ``smoke``
     - S0
     - 6
     - 3
     - 0
     - 9
   * -
     - S2
     - 2
     - 70
     - 0
     - 72
   * -
     - S3
     - 2
     - 0
     - 0
     - 2
   * - ``small``
     - S0
     - 9
     - 0
     - 0
     - 9
   * -
     - S2
     - 11
     - 85
     - 0
     - 96
   * -
     - S3
     - 2
     - 0
     - 2
     - 4

* ``smoke``: 10 candidatas, 70 unidades (candidata × origem), 140 ajustes, 22 s; só 6
  das 9 bases couberam no limite por etapa. Campeão: ``linear``.
* ``small``: 22 candidatas, 154 unidades, 350 ajustes, cerca de 5 min nesta máquina
  (com outra carga em paralelo). Todas as bases e 11 dos 96 híbridos possíveis foram
  avaliados; o limite **por família** (15) parou os híbridos antes do total (50). Campeão:
  ``random_forest``, aceito contra o ``linear`` com ganho mediano de 12 % e 6 vitórias
  em 7.
* As 2 propostas puladas em S3 eram a fonte ``in_sample`` com bases que não expõem
  valores ajustados.

O S2 propõe 96 híbridos: 8 bases retidas × 3 corretores × 2 janelas × 2 valores de α.
Para avaliar mais deles, aumente o orçamento ou reduza o espaço:

.. code-block:: python

   exp = AutoHybridExperiment(
       fh=14,
       budget="medium",                 # 200 candidatas, 40 por família
       base_models=["seasonal_naive", "arima", "linear", "knn"],
       base_params={"knn": [{"lags": 7}, {"lags": 14}, {"lags": 28}]},
       correctors=["linear", "knn"],
       corrector_params={"knn": [{"estimator_params": {"n_neighbors": 3}}]},
       residual_windows=[7, 14],
       alphas=[1.0],
   )
   result = exp.compare_models()
   result.search.ledger_frame()        # cada proposta, avaliada ou não, e por quê

Custo
-----

O custo de uma busca é proporcional a candidatas × origens × ajustes por candidata. Com
``fold="auto"`` (7 origens por camada quando a série comporta; veja :doc:`selection`), cada
candidata é avaliada em mais origens que com o antigo padrão de 3: no exemplo acima, o
``small`` levou cerca de 3,5 vezes o tempo. Um híbrido ajusta a base duas vezes por
origem (treino e prefixo dos resíduos) mais o corretor. Para uma exploração rápida,
``fold=3`` e ``budget="smoke"`` continuam disponíveis, com a regra de aceite menos
confiável que isso implica (:doc:`selection`).
