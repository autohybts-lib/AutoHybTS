Estratégias de sistemas híbridos
================================

Cada candidata tem um **nome curto** que descreve a estrutura (``arima-mlp``) e um
rótulo completo com a configuração (janela, alfa, fonte de resíduos).

==========================  ========================  ============================================
Estratégia                  Nome                      Ideia
==========================  ========================  ============================================
``standalone``              ``arima``                 um modelo
``residual_additive``       ``arima-mlp``             ``ŷ = b + α·r̂``: o corretor prevê o erro
``residual_multistage``     ``arima-mlp-linear``      cada estágio corrige o erro do anterior
``decomposition``           ``ma(linear-mlp)``        base na parte suave, corretor na volátil
``fixed_combination``       ``comb(arima,knn)``       média ponderada de bases
``residual_combination``    ``arima-comb(mlp,knn)``   base + média ponderada de corretores
``dynamic_selection``       ``dref(arima-mlp,knn)``   escolhe o corretor a cada passo (só B)
==========================  ========================  ============================================

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

Por padrão os estágios extras repetem janela, alfa e fonte de resíduo do primeiro.
``stage_params`` muda isso por estágio; na busca,
``AutoHybridExperiment(stage_params="per_stage")`` também varia a janela e o alfa do
estágio extra no refinamento.

.. code-block:: python

   exp.create_hybrid("arima", "mlp", extra_stages=["linear"],
                     stage_params=[{"window_size": 28, "alpha": 0.5}])

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

Combinação de corretores: vários corretores sobre o mesmo resíduo,
``ŷ = b + α·Σ wⱼ r̂ⱼ``. Com ``inverse_mse``, os pesos vêm do erro de cada corretor nas
últimas linhas do seu próprio treino, ajustado nas anteriores.

.. code-block:: python

   exp.create_hybrid("arima", "mlp", combine_correctors=["knn", "linear"],
                     combination_weights="inverse_mse")

Seleção dinâmica (só variante B)
--------------------------------

Um grupo de corretores (mais a opção de **não corrigir**). A cada passo, os resíduos
logo antes da origem são comparados com uma biblioteca de situações passadas (o DSEL,
formado por erros de um passo **fora da amostra** numa cauda do treino, nunca na janela
que pontua) e os ``k`` vizinhos mais parecidos decidem:

``dref``
   usa o corretor de menor erro nesses vizinhos — ou nenhum.
``local``
   ajusta uma Kernel Ridge pequena nos vizinhos, de (previsão da base, correção) para o
   valor observado, só para aquele passo.

Na busca, a grade é a do legado (Silva et al., 2020): ``k ∈ {10, 5, 20}``,
``lag ∈ {7, 14}`` e, no ``local``, ``alpha ∈ {1, 0.1}``; os valores padrão vêm primeiro.

Sem observação entre os passos não há o que selecionar: na variante A a estratégia é
recusada.

.. code-block:: python

   exp = AutoHybridExperiment(fh=7, variant="B").fit(y)
   result = exp.compare_models(include=("standalone", "residual", "dynamic"))

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
S4        decomposições das bases retidas, combinação das 2 melhores de famílias
          diferentes e, na variante B, seleção dinâmica em torno das melhores
          estruturas residuais, com o orçamento que sobrar (uma vaga por família)
========  ===================================================================

Todas as propostas — avaliadas, podadas pelo orçamento, puladas por capacidade ou
duplicadas — ficam no ledger: ``result.search.ledger_frame()``.
