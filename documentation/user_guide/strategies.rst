Estratégias de sistemas híbridos
================================

Cada candidata tem um **nome curto** que descreve a estrutura (``arima-mlp``) e um
rótulo completo com a configuração (janela, alfa, fonte de resíduos).

==========================  ======================  ============================================
Estratégia                  Nome                    Ideia
==========================  ======================  ============================================
``standalone``              ``arima``               um modelo
``residual_additive``       ``arima-mlp``           ``ŷ = b + α·r̂``: o corretor prevê o erro
``residual_multistage``     ``arima-mlp-linear``    cada estágio corrige o erro do anterior
``decomposition``           ``ma(linear-mlp)``      base na parte suave, corretor na volátil
``fixed_combination``       ``comb(arima,knn)``     média ponderada de bases
==========================  ======================  ============================================

Correção aditiva
----------------

O modelo-base ``b`` é ajustado; os resíduos ``e = y - b`` são gerados dentro do treino
pela fonte configurada; o corretor aprende ``e[t]`` a partir das ``w`` últimas entradas.
Na variante A (multipasso), a janela do corretor é alimentada com os próprios resíduos
**previstos** após a origem — nunca com resíduos observados no futuro.

.. code-block:: python

   exp.create_hybrid("arima", "mlp", window=7, alpha=1.0,
                     residual_source="historical_oof_1step")

Vários estágios
---------------

O estágio *k* é um híbrido residual cuja base é o estágio *k−1*; o alvo do estágio *k*
é o erro do modelo inteiro anterior, gerado dentro do treino. Até 3 estágios.

.. code-block:: python

   exp.create_hybrid("arima", "mlp", extra_stages=["linear"])

Decomposição
------------

``y = s + v``. A base prevê a parte suave ``s`` e o corretor, a volátil ``v``.

``ma``
   média móvel **causal** (``s[t]`` usa ``y`` até ``t``), com janela escolhida pela
   curtose da parte volátil mais próxima de 3, calculada só no treino. Recalculada
   causalmente em qualquer contexto: vale nas variantes A e B.
``stl``
   STL robusta ajustada só no treino; só na variante A.

.. code-block:: python

   exp.create_decomposition("linear", "mlp", method="ma")       # janela pela curtose
   exp.create_decomposition("linear", "mlp", method="ma", ma_window=14)
   exp.create_decomposition("linear", "mlp", method="stl")

Combinação fixa
---------------

Bases combinadas com pesos fixados no ajuste: ``equal`` ou ``inverse_mse`` (inverso do
erro de um passo fora da amostra, medido no próprio treino).

.. code-block:: python

   exp.create_combination(["arima", "knn"], weights="inverse_mse")

Na busca
--------

.. code-block:: python

   result = exp.compare_models(
       include=("standalone", "residual", "multistage", "decomposition", "combination")
   )

========  ===================================================================
Etapa     O que faz
========  ===================================================================
S0        avalia todas as bases
S1        retém as 2 melhores por família, o ARIMA e a melhor geral
S2        grade residual (corretor × janela × alfa × fonte) nas bases retidas
S3        refina as 2 melhores estruturas residuais: fonte, parâmetros do
          corretor e estágios extras
S4        decomposições das bases retidas e combinação das 2 melhores de famílias
          diferentes, com o orçamento que sobrar (uma vaga garantida por família)
========  ===================================================================

Todas as propostas — avaliadas, podadas pelo orçamento, puladas por capacidade ou
duplicadas — ficam no ledger: ``result.search.ledger_frame()``.
