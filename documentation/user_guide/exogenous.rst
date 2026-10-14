Variáveis exógenas
==================

Uma série pode depender de outras: a demanda elétrica da temperatura e do dia útil, as
vendas de um indicador antecedente, as mortes no trânsito de uma lei. Com ``fit(y, X)`` o
experimento tenta modelos com essas covariáveis — ARIMAX, SARIMAX, os modelos sobre lags e
os híbridos sobre eles — e a seleção decide, pela mesma regra de aceite, se elas entram.

.. code-block:: python

   from autohybts import AutoHybridExperiment
   from autohybts.datasets import load_exog_dataset

   y, X = load_exog_dataset("bjsales")             # vendas e indicador antecedente
   exp = AutoHybridExperiment(fh=3, exog_selection="auto").fit(y, X)
   cmp = exp.compare_models()                      # família "exogenous" incluída
   cmp.decisions                                   # aceita? com que ganho?
   exp.compare_exog().tests()                      # Wilcoxon e Diebold–Mariano
   final = exp.finalize_model(cmp.best)
   exp.evaluate_external(final)

``X`` é um ``DataFrame`` indexado por data, com todas as datas de ``y`` (inclusive as do
teste selado) e, para prever além dos dados, as datas seguintes das colunas ``future``.

O que cada coluna pode ver
--------------------------

O risco das exógenas é o vazamento: usar na data prevista um valor que só seria conhecido
depois. Por isso cada coluna tem um tipo.

.. list-table::
   :header-rows: 1
   :widths: 14 40 46

   * - Tipo
     - Quando usar
     - Como entra
   * - ``past`` (padrão)
     - valores observados com a série: preços, indicadores, temperatura medida
     - defasada em pelo menos os passos previstos de uma origem: ``max(fh, test_size)``
       na variante A, 1 na B. O valor usado na data ``t`` é ``x[t - lag]``, já conhecido
       na origem.
   * - ``future``
     - valores conhecidos antes: calendário, feriados, preço ou promoção planejados, uma
       lei já aprovada
     - na própria data prevista (lag 0)

Declare as colunas ``future`` em ``exog_future=[...]``; lags extras vão em
``exog_lags={"coluna": [7, 14]}``. Uma coluna ``past`` com lag menor que o mínimo é
recusada. O contexto temporal e o modelo finalizado recusam qualquer linha além do que era
conhecido na origem (``TemporalLeakageError``).

.. warning::

   Declarar como ``future`` um valor que só é observado depois — a temperatura do dia, o
   preço de amanhã — transforma a avaliação num **oráculo**: o erro medido é o de quem
   conhecesse o futuro. Use ``future`` assim só para medir o potencial de uma boa
   previsão da covariável; em produção, a coluna precisa vir dessa previsão.

Escolha automática das variáveis
--------------------------------

``exog_selection`` define os conjuntos de features que a busca tenta:

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Valor
     - Conjuntos
   * - ``None`` (padrão)
     - todas as colunas no menor lag causal
   * - ``"ccf"``
     - correlação cruzada pré-branqueada (Box–Jenkins), com Holm
   * - ``"granger"``
     - teste F de Granger por coluna, com Holm; o lag de maior ``|t|``
   * - ``"lasso"``
     - Lasso com a penalidade da regra de um erro-padrão numa validação temporal
   * - ``"stepwise_bic"``
     - seleção progressiva por BIC
   * - ``"mutual_info"``
     - informação mútua contra 30 permutações (nulo do máximo entre todas as features)
   * - ``"vote"``
     - colunas escolhidas pela maioria dos cinco métodos
   * - ``"auto"``
     - um conjunto por método e o de todas as colunas; a validação decide
   * - lista de métodos
     - um conjunto por método da lista

Os métodos usam só os dados antes da primeira origem de ``V_select``, como o ajuste de
hiperparâmetros usa ``V_search``; ``V_select`` e o teste selado nunca chegam a eles.
``exp.exog_report()`` mostra, por método, cada feature candidata, o escore, o p-valor
(quando existe) e se foi escolhida.

Medido em séries sintéticas com uma exógena relevante (lag 7), um feriado e três colunas de
ruído (10 séries com relação e 10 sem, 300 pontos):

.. list-table::
   :header-rows: 1
   :widths: 20 26 26 28

   * - Método
     - ruído escolhido (séries sem relação, ruído AR / passeio aleatório)
     - relevantes encontradas (de 20)
     - extras nas séries com relação
   * - ``vote``
     - 0 / 0
     - 17 / 19
     - 0 / 0
   * - ``granger``
     - 0 / 1
     - 17 / 17
     - 0 / 0
   * - ``ccf``
     - 0 / 0
     - 10 / 16
     - 1 / 0
   * - ``mutual_info``
     - 1 / 0
     - 10 / 10
     - 0 / 0
   * - ``lasso``
     - 0 / 0
     - 20 / 20
     - 12 / 16
   * - ``stepwise_bic``
     - 3 / 10
     - 20 / 20
     - 17 / 14

``lasso`` e ``stepwise_bic`` encontram tudo, mas levam lags vizinhos junto; ``vote`` é o
melhor equilíbrio. Em ``"auto"`` isso pesa pouco: um conjunto ruim perde na validação.

Com × sem exógenas
------------------

A decisão é da seleção incremental: a família ``exogenous`` desafia o campeão univariado
depois das estruturas residuais, e só entra se passar na regra de aceite calibrada (ganho
mediano ≥ 5 %, vitórias ≥ 60 %). Um modelo com exógenas nunca é o campeão inicial.

``exp.compare_exog()`` compara o melhor finalista com e o melhor sem exógenas nas mesmas
unidades de ``V_select``: ``.summary``, ``.tests()`` (Wilcoxon e Diebold–Mariano, com Holm)
e ``.plot()``. Os testes são relatório; com 7 unidades, o menor p-valor possível do
Wilcoxon é 0,016. Para o teste selado, finalize os dois e use ``compare_external``.

Resultados nos benchmarks (horizonte e seleção do notebook 20):

.. list-table::
   :header-rows: 1
   :widths: 30 14 14 14 28

   * - Conjunto
     - aceito
     - ganho mediano
     - p (Wilcoxon)
     - MASE no teste (sem → com)
   * - ``bjsales`` (fh = 3, indicador antecedente)
     - sim
     - 69 %
     - 0,016
     - 0,110 → 0,029
   * - ``elecdaily`` com temperatura ``past``
     - não
     - −4 %
     - 0,58
     - 1,954 → 1,954
   * - ``elecdaily`` com temperatura ``future`` (oráculo)
     - sim
     - 28 %
     - 0,031
     - 1,954 → 0,689
   * - ``seatbelts``, ``uschange`` (fh = 12 e 8)
     - não
     - < 0
     - > 0,1
     - igual

Covariáveis ``past`` defasadas pelo horizonte inteiro raramente ajudam em horizontes
longos; um indicador que antecede a série pelo horizonte, ou covariáveis de fato conhecidas
antes, sim.

Prever, atualizar e servir
--------------------------

.. code-block:: python

   final = exp.finalize_model(cmp.best)
   exp.predict_model(final)                    # colunas future vêm do X do fit...
   final.predict(3, X_future=novas_linhas)     # ...ou de X_future

   novo = final.update_state(y_novo, X_new=X_novo)   # sem reajuste
   stream = novo.stream()                            # variante B
   ticket = stream.predict_next(exog={"feriado": 1.0})
   stream.observe(valor, exog={"indicador": 12.3})

O modelo finalizado guarda as covariáveis até a última observação; ``save_model`` e
``load_model`` as preservam. Um horizonte maior que a defasagem das colunas ``past`` é
recusado.

Conjuntos com exógenas
----------------------

``list_exog_datasets()`` lista e ``load_exog_dataset(nome)`` devolve ``(y, X)``, baixados do
Rdatasets na primeira vez:

.. list-table::
   :header-rows: 1
   :widths: 18 58 24

   * - Nome
     - Série
     - Colunas ``future``
   * - ``seatbelts``
     - motoristas mortos ou feridos no Reino Unido ~ km rodados, preço da gasolina, lei do
       cinto (mensal, 1969–1984)
     - ``lei_cinto``
   * - ``bjsales``
     - vendas ~ indicador antecedente (Box–Jenkins)
     - —
   * - ``uschange``
     - consumo nos EUA ~ renda, produção, poupança, desemprego (trimestral)
     - —
   * - ``insurance``
     - cotações de seguro ~ publicidade na TV (mensal, 40 pontos)
     - ``publicidade_tv``
   * - ``elecdaily``
     - demanda elétrica diária de Victoria ~ temperatura (e o quadrado) e dia útil, 2014
     - ``dia_util``
   * - ``vic_elec_daily``
     - demanda diária de Victoria em GWh ~ temperatura máxima (e o quadrado), feriado e fim
       de semana, 2012–2014
     - ``feriado``, ``fim_de_semana``

Painéis de séries
-----------------

Num painel, ``X`` é um dicionário ``série -> DataFrame`` ou um ``DataFrame`` longo com as
mesmas ``series_col`` e ``date_col`` de ``y``. Todas as séries precisam das mesmas colunas;
cada uma usa só as suas covariáveis.

.. code-block:: python

   exp = AutoHybridExperiment(fh=7, exog_future=["feriado"], exog_selection="auto")
   exp.fit({"loja_a": y_a, "loja_b": y_b}, {"loja_a": X_a, "loja_b": X_b})
   # ou: exp.fit(y_longo, X_longo, series_col="loja", date_col="data", target="vendas")
   final = exp.finalize_model(exp.compare_models().best)     # PanelForecaster
   exp.predict_model(final, X_future={"loja_a": Xf_a, "loja_b": Xf_b})

O conjunto de features é o mesmo para todas as séries; a configuração vencedora pode variar
por série, como no painel sem exógenas. Na seleção automática, cada método roda em cada
série e fica com as features que a maioria das séries escolheu; ``exog_report()`` traz a
coluna ``series_id``.

Limites
-------

* Decomposição, combinação, ensemble e seleção dinâmica não recebem exógenas.
* Nos híbridos, com exógenas só a fonte residual ``historical_oof_1step`` é usada (a
  multipasso leria colunas ``past`` além da defasagem).
* Nos modelos sobre lags, as covariáveis entram como dadas (as diferenças valem só para a
  série).

.. seealso::

   Tutoriais: :doc:`../tutorials/notebooks/20_exogenous`,
   :doc:`../tutorials/notebooks/21_mlops_exogenous`.
