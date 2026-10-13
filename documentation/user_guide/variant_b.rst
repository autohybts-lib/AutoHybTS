Variantes A e B: como a previsão encontra o futuro
==================================================

A variante diz **quando o valor real fica disponível** enquanto o modelo prevê. Nas duas,
o modelo é ajustado uma vez na origem; o que muda é como as ``fh`` previsões seguintes são
produzidas. A variante é escolhida no experimento e vale para a busca, a seleção, a
finalização e a previsão:

.. code-block:: python

   AutoHybridExperiment(fh=14)                 # variante A (padrão)
   AutoHybridExperiment(fh=7, variant="B")     # variante B

Variante A: vários passos de uma vez (padrão)
---------------------------------------------

Na origem *t*, o modelo prevê *t+1, …, t+fh* **sem ver nenhum valor real depois de t**.
Para prever *t+3*, os valores de *t+1* e *t+2* ainda não existem: o modelo usa as próprias
previsões deles.

Num híbrido (``ŷ = b + α·r̂``), o corretor precisa dos últimos ``w`` resíduos. Depois da
origem não há resíduo observado, então a janela recebe os **resíduos previstos** ``r̂``.
O erro tende a se acumular ao longo do horizonte, e o ganho do corretor costuma ser mais
forte no primeiro passo. Quando o padrão dominante é regular e bem modelado, a diferença
para a variante B pode ser pequena: no notebook 04, num trecho de 28 dias, a variante A
teve até erro menor.

**Quando usar:** quando o plano inteiro é necessário de antemão, por exemplo a demanda dos
próximos 14 dias calculada hoje.

Variante B: um passo por vez, parâmetros congelados
---------------------------------------------------

O modelo prevê **só o próximo passo**. O valor real chega, entra no contexto, e o modelo
prevê o passo seguinte, até completar ``fh`` passos. Os parâmetros **não mudam**: nada é
reajustado; só o estado (o contexto observado) avança.

Num híbrido, o resíduo observado ``y − b`` entra na janela do corretor **somente depois**
da observação: o corretor trabalha com erros verdadeiros, não com erros que ele mesmo
previu.

**Quando usar:** em operação contínua, quando cada valor chega antes da próxima previsão,
por exemplo monitoramento ou a previsão do dia seguinte atualizada todo dia. A escolha
segue a operação, não uma vantagem garantida de precisão.

Exemplo: ``fh = 3``, origem no dia 10
-------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 16 42 42

   * -
     - Variante A
     - Variante B
   * - Dia 11
     - prevê com dados até o dia 10
     - prevê com dados até o dia 10
   * - Dia 12
     - prevê com dados até o dia 10 (e a previsão do dia 11)
     - prevê com dados até o dia **11 observado**
   * - Dia 13
     - prevê com dados até o dia 10 (e as previsões dos dias 11 e 12)
     - prevê com dados até o dia **12 observado**
   * - Ajustes
     - 1, na origem
     - 1, na origem; nenhum depois

Efeito na validação
-------------------

A validação temporal simula a variante escolhida. Em A, cada origem gera uma previsão de
``fh`` passos. Em B, cada origem gera ``fh`` previsões de um passo, e cada alvo só é
revelado depois da sua previsão. Os erros das duas variantes **não são comparáveis**: em
B todo erro é de um passo, em A eles cobrem os horizontes 1 a ``fh``. Compare modelos
sempre dentro da mesma variante.

Compatibilidades
----------------

.. list-table::
   :header-rows: 1
   :widths: 46 18 18

   * - Recurso
     - A
     - B
   * - Modelos-base que incorporam observações novas sem reajuste
       (``online_state_update``)
     - sim
     - sim
   * - Modelos-base sem essa capacidade
     - sim
     - ``SKIPPED``
   * - Decomposição ``ma`` (recalculada causalmente)
     - sim
     - sim
   * - Decomposição ``stl`` (ajustada só no treino)
     - sim
     - não
   * - Seleção dinâmica do corretor (``dref``, ``local``)
     - não
     - sim
   * - Corretores de sequência residual (``arima_sequence`` …)
     - sim
     - sim
   * - ``predict_model`` de vários passos
     - sim
     - só ``fh=1``; use ``stream()``

Em produção (variante B)
------------------------

.. code-block:: python

   exp = AutoHybridExperiment(fh=7, variant="B").fit(y)
   result = exp.compare_models()
   final = exp.finalize_model(result.best)

   stream = final.stream()
   ticket = stream.predict_next()      # previsão, base e correção do próximo passo
   stream.observe(valor_observado)     # revela o valor; nada é reajustado

* ``predict_next`` e ``observe`` alternam estritamente: prever duas vezes sem observar é
  recusado.
* ``ForecastTicket`` traz a previsão, a parte da base e a correção daquele passo.
* ``FinalizedForecaster.update_state`` devolve um **novo** objeto com as observações novas
  e recusa ``refit=True``. ``training_cutoff`` (fim do treino) difere de
  ``last_observation`` (último valor recebido).

O notebook :doc:`../tutorials/notebooks/04_rolling_one_step` mostra o streaming, a base e
a correção a cada passo e as duas variantes no mesmo trecho, com figuras.
