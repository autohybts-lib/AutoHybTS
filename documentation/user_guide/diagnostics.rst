Diagnóstico da série
====================

Antes de escolher modelos, vale saber se a série tem um ciclo regular. A sazonalidade
decide quais modelos fazem sentido: ``seasonal_naive``, ``exponential_smoothing``
sazonal, a decomposição ``stl`` e o ``sarima`` só ajudam quando o ciclo existe. Esta
página mostra o diagnóstico exploratório e como a busca pode usá-lo sem vazamento.
O notebook :doc:`../tutorials/notebooks/16_series_diagnostics` percorre tudo com
figuras.

Sazonalidade
------------

.. code-block:: python

   exp = AutoHybridExperiment(fh=14).fit(y)
   exp.seasonality_report()              # período do experimento
   exp.seasonality_report(periods=[7, 30])

Uma linha por série e período candidato, calculada no histórico de desenvolvimento
(nunca no teste selado):

``n_cycles``
   ciclos completos disponíveis. Com menos de 3, não há teste.
``strength``
   força da sazonalidade, ``F_s = max(0, 1 − Var(R)/Var(S + R))``, de uma decomposição
   clássica (média móvel centrada de um ciclo para a tendência, médias por posição para
   a sazonalidade; Wang, Smith e Hyndman, 2006). Perto de 1: ciclo dominante; perto de
   0: nenhum.
``autocorrelation``
   autocorrelação da série sem tendência no lag do período.
``p_value``
   teste de Friedman da série sem tendência: cada ciclo é um bloco e cada posição no
   ciclo, um tratamento, de modo que mudanças lentas entre ciclos não contam como
   sazonalidade.
``seasonal``
   decisão da regra: ``p_value < 0,05`` com ao menos 3 ciclos.
``seasonal_differences``
   diferenças sazonais sugeridas: 1 quando há sazonalidade e ``strength ≥ 0,64`` (a regra
   do ``nsdiffs`` do pacote R ``forecast``).

Nas séries sintéticas da biblioteca (730 dias, período 7):

.. list-table::
   :header-rows: 1
   :widths: 28 14 14 18 14 12

   * - Série
     - ``n_cycles``
     - ``strength``
     - ``p_value``
     - ``seasonal``
     - ``D``
   * - ``synthetic_daily``
     - 102,3
     - 0,87
     - 1,6 × 10⁻¹¹⁴
     - sim
     - 1
   * - ``white_noise_daily``
     - 102,3
     - 0,01
     - 0,81
     - não
     - 0

Em 200 séries simuladas por cenário, sem sazonalidade, a regra declarou um ciclo em
2–9 % das séries com ruído independente (com ou sem tendência linear) e em até 11 % com
ruído persistente (AR(0,8) ou passeio aleatório), contra os 5 % nominais. Com um ciclo
de amplitude igual ao desvio do ruído, detectou 68–100 % das séries a partir de 6 ciclos
e quase nenhuma com 3. O poder depende do número de ciclos: com poucos, não rejeitar não
é evidência de que o ciclo não existe.

Sazonalidade na busca: ``seasonal_period="auto"``
-------------------------------------------------

Por padrão, o período vem da frequência (diária: 7, mensal: 12, anual: 1) e os modelos
sazonais o recebem sempre. Com ``seasonal_period="auto"``, a busca testa antes:

.. code-block:: python

   exp = AutoHybridExperiment(fh=14, seasonal_period="auto").fit(y)
   exp.seasonal_period_    # 7: período das métricas (escala da MASE), sempre o da frequência
   exp.search_period_      # 7 com ciclo; 1 sem evidência de ciclo
   exp.seasonality_        # o teste que decidiu

O teste usa só o treino da **primeira** origem de validação: dados que todas as origens
já viram, então a decisão não depende de nenhum alvo avaliado. Se esse treino tem ao
menos 6 ciclos e o teste não mostra ciclo, ``search_period_`` vale 1: ``seasonal_naive``,
``sarima`` e a STL ficam ``SKIPPED`` com o motivo no ledger, e os demais modelos recebem
a versão não sazonal. Com menos de 6 ciclos o teste quase não tem poder, e o período da
frequência fica (coluna ``keep_period`` de ``seasonality_``). A escala da MASE
continua a da frequência, para os erros serem comparáveis com e sem ``"auto"``. Num
painel, o ciclo vale quando ao menos metade das séries o mostra.

Com os dados acima, ``synthetic_daily`` manteve o período 7 (p ≈ 3 × 10⁻⁸² no treino da
primeira origem) e ``white_noise_daily`` passou a 1 (p = 0,59).

.. warning::

   O teste é decidido no treino da primeira origem, que pode ser curto. No
   ``air_passengers`` (mensal, ``fh=12``), esse treino tem só 3 ciclos: o teste deu
   p = 0,039 e o período 12 ficaria de qualquer forma, pela regra dos 6 ciclos. Em séries
   curtas, ``"auto"`` quase sempre mantém o período da frequência; prefira um
   ``seasonal_period`` explícito, escolhido pelo conhecimento do domínio.

Caracterização (regras do projeto de replicação)
------------------------------------------------

``exp.characterize()`` aplica ao histórico de desenvolvimento as duas regras do projeto de
replicação, com o mesmo resultado (teste de paridade com o código legado):

1. **Estacionariedade (ADF)**, com constante e lags pelo AIC: a série é estacionária
   quando ``p ≤ 0,05`` e a estatística fica abaixo do valor crítico de 5 %. Se não for,
   ``diff_order`` é a menor ordem até 2 que passa. O KPSS de cada ordem também é
   informado, sem entrar na decisão.
2. **Sazonalidade**, só para as séries não estacionárias: força ``F_S`` de uma STL robusta
   sobre a série transformada por Box-Cox (λ por máxima verossimilhança entre −1 e 2),
   para cada período candidato com ao menos dois ciclos. O de maior força é o período
   dominante, e a série é sazonal quando ``F_S > 0,5``.

.. code-block:: python

   exp.characterize()                  # período do experimento
   exp.characterize(periods=[7, 365])  # diária: semana e ano, como na replicação

A categoria resume as duas regras: ``stationary``, ``nonseasonal`` ou ``seasonal``. Nas
séries sintéticas (histórico de 716 dias):

.. list-table::
   :header-rows: 1
   :widths: 24 14 14 14 12 12 14

   * - Série
     - ADF (nível)
     - p ADF d=0
     - p ADF d=1
     - ``diff_order``
     - ``fs_7``
     - categoria
   * - ``synthetic_daily``
     - −0,75
     - 0,83
     - < 0,001
     - 1
     - 0,84
     - ``seasonal``
   * - ``white_noise_daily``
     - −25,0
     - < 0,001
     - —
     - 0
     - —
     - ``stationary``

Com ``seasonal_period="auto"``, ``seasonality_test="strength"`` troca o teste de Friedman
pela regra 2: uma série que o ADF considera estacionária não tem sazonalidade, e as
demais precisam de ``F_S > 0,5``. Sem dois ciclos a força é indefinida e o período da
frequência fica.

Gráficos
--------

Todos usam o histórico de desenvolvimento, nunca o teste selado (extras ``plot`` e
``stats``):

``exp.plot_data("stationarity")``
   a série em nível, com uma e com duas diferenças, com os p-valores do ADF e do KPSS em
   cada painel e a ordem escolhida pelo ADF destacada.
``exp.plot_data("acf_pacf")``
   ACF e PACF na ordem escolhida, com a faixa ±1,96/√n. A PACF é a de Yule-Walker com
   autocovariâncias enviesadas (a ``method="ywm"`` do statsmodels).
``exp.plot_data("decomposition")``
   tendência, sazonalidade e resíduo da STL robusta da série Box-Cox, no período do
   experimento.
``exp.plot_characterization()``
   estatística ADF contra ``F_S``, um ponto por série, colorido pela categoria; útil num
   painel.

Estacionariedade: ADF e KPSS
----------------------------

``exp.stationarity_report()`` testa o nível e as diferenças até ``max_d`` (padrão 2) com
os dois testes, que têm hipóteses nulas opostas: o ADF rejeita a raiz unitária quando
``p ≤ 0,05`` e a estatística fica abaixo do crítico de 5 %; o KPSS diz estacionária
quando não rejeita (``p ≥ 0,05``). ``consensus`` marca as ordens em que os dois
concordam, e ``attrs`` traz a menor ordem de cada regra (``d_adf``, ``d_kpss``,
``d_consensus``). Na ``synthetic_daily``:

.. code-block:: text

   d   adf_stat  adf_pvalue  adf_stationary  kpss_pvalue  kpss_stationary  consensus
   0     -0.749       0.834           False         0.01            False      False
   1    -11.128     < 0.001            True         0.10             True       True
   2    -12.048     < 0.001            True         0.10             True       True

   {'d_adf': 1, 'd_kpss': 1, 'd_consensus': 1}

O KPSS informa p-valores só entre 0,01 e 0,10 (fora disso, o limite da tabela); com
poucas observações, os dois testes têm pouco poder.

Decomposição em tabela
----------------------

``exp.decompose("stl")`` (STL robusta) ou ``exp.decompose("classical")`` (média móvel
centrada e índices sazonais médios) devolvem ``observed``, ``trend``, ``seasonal`` e
``resid`` indexados pela data, no período do experimento (ou ``period=``). Na clássica, a
tendência é ``NaN`` nas pontas. É exploratório: a estratégia ``decomposition`` refaz a
decomposição no treino de cada origem.

Resíduos de uma base: Ljung-Box
-------------------------------

``exp.residual_diagnostics("arima")`` ajusta a base na primeira metade do histórico de
desenvolvimento e prevê a outra metade um passo por vez, com parâmetros congelados: são
os resíduos fora da amostra de que os corretores aprendem. O teste de Ljung-Box diz se
sobrou autocorrelação neles; ``p`` pequeno indica espaço para um corretor residual. Lags
padrão: ``min(10, n/5)`` e duas vezes o período sazonal. Na ``synthetic_daily`` (358
resíduos):

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Base
     - p (lag 10)
     - p (lag 14)
   * - ``naive``
     - < 0,001
     - < 0,001
   * - ``seasonal_naive``
     - < 0,001
     - < 0,001
   * - ``arima``
     - 0,070
     - 0,110

Os resíduos do ``naive`` e do ``seasonal_naive`` carregam a estrutura que a base não
modelou; os do ``arima``, quase nenhuma. Um teste sem rejeição não garante que um
corretor seja inútil: a regra de aceite decide nas origens de validação.

Diferenças automáticas
----------------------

Modelos sobre defasagens (linear, KNN, SVR, árvores, MLP, boosting e as redes
registradas) aprendem melhor numa série sem tendência nem raiz unitária. Com
``differencing="auto"``, cada origem escolhe, no seu próprio vetor de treino:

1. ``D``: uma diferença sazonal (``y[t] − y[t − m]``) quando o teste acima encontra um
   ciclo significativo com ``strength ≥ 0,64``;
2. ``d``: a menor ordem de 0 a 2 em que a série (já com a diferença sazonal) passa no
   teste de ``differencing_test``:

   * ``"kpss"`` (padrão): o KPSS não rejeita a estacionariedade a 5 %, como o ``ndiffs``
     do ``auto.arima`` e o ``arima`` desta biblioteca;
   * ``"adf"``: o ADF rejeita a raiz unitária (``p ≤ 0,05`` e estatística abaixo do valor
     crítico de 5 %), a regra do projeto de replicação.

As ordens escolhidas ficam congeladas: a previsão integra de volta a partir dos valores
observados no contexto, sem refazer os testes (variante B incluída). Os testes vêm do
statsmodels (extra ``stats``); sem ele, a candidata fica ``SKIPPED`` com o motivo.

.. code-block:: python

   # um modelo
   exp.create_model("knn", differencing="auto", season_length=7)
   exp.create_model("linear", differencing=1, seasonal_differencing=1, season_length=12)

   # todos os modelos sobre defasagens embutidos da busca
   exp = AutoHybridExperiment(fh=14, differencing="auto", differencing_test="adf")

A opção do experimento vale para os modelos embutidos da busca (``compare_models``) e usa
o período da busca (``search_period_``); ``create_model`` usa só os parâmetros que você
passa, e uma rede registrada recebe ``differencing="auto"`` no registro
(``torch_model(LSTMNet, differencing="auto")``). O padrão
(``differencing=None``) não diferencia, e os IDs das candidatas não mudam.

Pré-processamento opcional: Box-Cox e outliers
----------------------------------------------

Desligados por padrão; o usuário liga por modelo ou para todos os modelos sobre
defasagens da busca. Tudo é ajustado no treino de cada origem, e os alvos das métricas
continuam os valores observados.

``transform="boxcox"`` ou ``"log"``
   λ por máxima verossimilhança no treino (com deslocamento quando há valores ≤ 0),
   congelado; o contexto da previsão passa pela mesma transformação e a previsão volta à
   escala original. Aplicado antes das diferenças.
``outliers="winsorize"`` ou ``"interpolate"``
   pontos com z robusto acima de 4 (em módulo) contra uma mediana móvel são cortados no limiar ou
   substituídos por interpolação, só nos valores com que o modelo aprende. O contexto da
   previsão não é alterado.

.. code-block:: python

   exp.create_model("linear", lags=12, transform="boxcox", differencing="auto", season_length=12)
   AutoHybridExperiment(fh=12, transform="boxcox", outliers="winsorize")   # toda a busca

No ``air_passengers`` (mensal, 4 origens), o linear sobre 12 lags teve MASE 0,822; com
``transform="log"``, 0,881; com ``"boxcox"``, 0,927; com Box-Cox e ``differencing="auto"``,
0,773. A transformação sozinha não melhorou: combinada com as diferenças, sim. Use-a como
mais uma opção a validar, não como padrão.

ARIMA sazonal
-------------

O ``sarima`` (extra ``stats``) é o ARIMA sazonal do statsmodels com escolha automática no
treino de cada origem, como o ``auto.arima`` (Hyndman e Khandakar, 2008):

1. ``D`` (0 ou 1) pela regra acima: ciclo significativo e ``strength ≥ 0,64``;
2. ``d`` (até 2) por testes KPSS sucessivos na série com a diferença sazonal;
3. ``(p, q, P, Q)`` por AICc numa busca *stepwise* a partir de quatro modelos iniciais,
   com ``p, q ≤ 2`` e ``P, Q ≤ 1``.

.. code-block:: python

   exp = AutoHybridExperiment(fh=12, base_models=["seasonal_naive", "arima", "sarima"])
   exp.create_model("sarima", season_length=12)
   exp.create_model("sarima", season_length=12, order=(1, 1, 0), seasonal_order=(0, 1, 1))

A ordem e os parâmetros ficam congelados depois do ajuste: um contexto mais longo
(variante B, resíduos fora da amostra) só atualiza o estado do filtro de Kalman. O
``sarima`` está fora da busca padrão (entra por ``base_models``) e recusa períodos acima
de 24, lentos demais para uma busca; para ciclos longos use ``harmonic_regression``. O
``arima`` continua não sazonal.

Medido em 13 de outubro de 2026 com ``create_model`` e os padrões do experimento (MASE
médio nas origens de validação e tempo total):

.. list-table::
   :header-rows: 1
   :widths: 28 18 18 18 18

   * - Série
     - ``seasonal_naive``
     - ``arima``
     - ``exponential_smoothing``
     - ``sarima``
   * - ``air_passengers`` (8 origens)
     - 1,27 · 0 s
     - 1,30 · 6 s
     - 0,68 · 0,5 s
     - 0,60 · 12 s
   * - ``synthetic_daily`` (14 origens)
     - 1,13 · 0 s
     - 1,13 · 26 s
     - 1,06 · 1,5 s
     - 0,76 · 310 s

O ganho veio com custo: numa série diária de 730 pontos, cada origem levou cerca de 22 s.
Por isso o ``sarima`` fica fora da busca padrão até ser medido em mais séries.
