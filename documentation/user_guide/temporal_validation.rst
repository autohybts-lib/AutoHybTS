Validação temporal sem vazamento
================================

Tudo em AutoHybTS parte de uma divisão temporal congelada em ``fit``:

.. code-block:: text

   |----------- histórico de desenvolvimento -----------|-- teste selado --|
   |  treino  ... | V_search (busca) | V_select (seleção)|  fh pontos       |

* **Teste selado**: os últimos ``test_size`` pontos (padrão ``fh``). É um objeto opaco
  (``SealedTest``): não pode ser iterado, indexado nem convertido. Só
  ``evaluate_external`` o lê, uma vez por modelo finalizado.
* **V_search**: origens usadas pela busca (ajuste de estruturas, retenção, refinamento,
  ``tune_model``).
* **V_select**: origens mais recentes, usadas só para decidir o campeão. Separar as duas
  camadas reduz o otimismo de escolher o vencedor no mesmo lugar onde ele foi procurado.

Em cada origem ``o``, o modelo é ajustado em ``y[:o+1]`` e prevê ``fh`` passos. Nada
depois da origem entra em ajuste, transformações, escala da MASE, janelas do corretor,
janela da média móvel ou pesos de combinação.

Parâmetros
----------

.. code-block:: python

   AutoHybridExperiment(
       fh=14,                     # horizonte por origem
       fold=3,                    # origens em cada camada
       fold_strategy="expanding", # ou "rolling" com window_size
       test_size=None,            # None = fh; 0 desliga o teste selado
       seasonal_period=None,      # inferido pela frequência (diária: 7, mensal: 12 ...)
   )

Resíduos do corretor
--------------------

O corretor aprende com resíduos gerados **dentro** do vetor de treino de cada origem:

``historical_oof_1step`` (padrão)
   uma base ajustada no prefixo prevê cada ponto da cauda um passo à frente, com
   parâmetros congelados.
``historical_oof_multistep``
   a cauda é cortada em blocos; cada bloco é previsto por uma base reajustada antes dele,
   com erros de horizontes 1..bloco, como os da variante A.
``in_sample``
   valores ajustados da base no próprio treino. Experimental: pode ser otimista.

Todo resíduo guarda origem, alvo, horizonte e ``available_at``; uma janela que use um
resíduo posterior à origem levanta ``TemporalLeakageError``.

Validação cruzada temporal
--------------------------

``exp.cross_validate(folds=8, step=7)`` cria dobras próprias (rolling origin, ancoradas
no fim do histórico de desenvolvimento) e avalia vários modelos nas mesmas origens. A
CV aninhada, ``exp.nested_cross_validate``, repete o procedimento inteiro (busca +
seleção) em cada dobra externa e mede quanto a escolha automática generaliza.
