Escolher a configuração
=======================

Os parâmetros de ``AutoHybridExperiment`` que mais mudam o resultado, com o padrão,
quando mudar e o que isso custa. A lista completa está na :doc:`../api/index`.

.. list-table::
   :header-rows: 1
   :widths: 20 16 40 24

   * - Parâmetro
     - Padrão
     - Quando mudar
     - Efeito e custo
   * - ``fh``
     - 14
     - sempre: é o horizonte que interessa à decisão (7 para uma semana de dados diários,
       12 para um ano de dados mensais)
     - horizontes longos acumulam erro na variante A
   * - ``test_size``
     - ``fh``
     - ``0`` para prever o futuro sem teste selado (operação); um bloco maior para uma
       avaliação final mais estável
     - pontos selados saem do histórico de desenvolvimento
   * - ``fold``
     - ``"auto"`` (7)
     - um inteiro menor só para explorar rápido
     - a regra de aceite foi calibrada para 7 origens; com menos, aceita estruturas
       espúrias com mais frequência; o custo cresce com o número de origens
   * - ``seasonal_period``
     - pela frequência
     - um inteiro quando o ciclo é conhecido; ``"auto"`` para testar o ciclo antes da
       busca (:doc:`diagnostics`)
     - define os modelos sazonais, a STL e a escala da MASE
   * - ``variant``
     - ``"A"``
     - ``"B"`` quando cada valor chega antes da próxima previsão (:doc:`variant_b`)
     - os erros das duas variantes não são comparáveis
   * - ``budget``
     - ``"small"``
     - ``"smoke"`` para testar; ``"medium"``/``"large"`` para explorar mais híbridos
     - custo proporcional a candidatas × origens; ``max_trials`` fixa o total
   * - ``base_models``, ``correctors``
     - 9 bases, 3 corretores
     - para incluir modelos opcionais (``sarima``, boosting …) ou reduzir a busca
     - cada modelo a mais multiplica as combinações de híbridos
   * - ``profile``
     - ``"standard"``
     - ``"research_strict"`` para exigir cobertura total; ``"quickstart"`` desliga o
       critério de variabilidade
     - muda o que a regra de aceite exige
   * - ``significance_alpha``
     - ``None``
     - para exigir também um teste de Wilcoxon pareado
     - quase não aceita híbridos espúrios, mas recusa a maioria dos ganhos úteis
       (:doc:`selection`)
   * - ``metrics``
     - MASE, MAE, RMSE, sMAPE
     - ``RMSSE`` primeiro para demanda intermitente (:doc:`results`)
     - a primeira métrica ordena e decide
   * - ``differencing``
     - ``None``
     - ``"auto"`` para os modelos sobre defasagens escolherem as diferenças
     - escolha no treino de cada origem; precisa do extra ``stats``
   * - ``max_wall_time_s``
     - ``None``
     - para limitar o tempo da busca
     - resultados podem mudar entre máquinas; para reproduzir, limite por quantidade
   * - ``n_jobs``
     - 1
     - para avaliar candidatas em paralelo
     - não muda o resultado (os limites valem para a lista ordenada de propostas)
   * - ``session_id``
     - 42
     - para repetir um experimento com outra semente
     - a mesma semente e os mesmos dados dão a mesma busca

Um ponto de partida por objetivo:

.. code-block:: python

   # explorar rápido (resultados indicativos)
   AutoHybridExperiment(fh=14, budget="smoke", fold=3)

   # decisão para um relatório
   AutoHybridExperiment(fh=14)                         # padrões calibrados

   # operação diária, um passo por vez
   AutoHybridExperiment(fh=1, variant="B", test_size=0)

   # série com ciclo incerto e tendência
   AutoHybridExperiment(fh=14, seasonal_period="auto", differencing="auto")

Acompanhar o progresso
----------------------

``progress=True`` mostra uma barra (tqdm) por etapa: cada estágio da busca (S0 bases, S2
resíduos, S3 refino, SX exógenas, S4 avançadas), os finalistas avaliados em ``V_select``, a
escolha de exógenas, as dobras da validação cruzada e da CV aninhada e o teste externo de
``compare_external``. Em notebooks a barra é um widget; no terminal, texto. Precisa do
extra ``progress`` (o ``notebook`` já o traz). As barras só informam: nada do que é avaliado
muda.

.. code-block:: python

   exp = AutoHybridExperiment(fh=14, progress=True).fit(y)
   exp.compare_models()

Com ou sem barras, cada evento do experimento (ajuste, fim da busca, fim da seleção,
finalização, avaliação externa) vai para o logger ``"autohybts"`` do ``logging``, com a
duração. Para gravar num arquivo:

.. code-block:: python

   import logging

   logging.basicConfig(filename="experimento.log", level=logging.INFO)
   # ou só a biblioteca:
   logging.getLogger("autohybts").setLevel(logging.INFO)
