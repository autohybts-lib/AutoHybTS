Conceitos
=========

O caminho dos dados
-------------------

.. code-block:: text

   série ──► fit: valida, sela o teste e congela as origens
               │
               ├─ histórico de desenvolvimento
               │     ├─ V_search: busca em etapas S0–S4 (bases, retenção, híbridos,
               │     │            refinamento, famílias avançadas)
               │     └─ V_select: seleção incremental (o desafiante só substitui o
               │                  campeão se passar na regra de aceite)
               │
               └─ teste selado: lido uma única vez por evaluate_external
                                     │
   finalize_model: reajusta o campeão no histórico anterior ao teste
                                     │
   predict_model / stream: previsão na variante A (vários passos) ou B (um passo)

O princípio da biblioteca: **complexidade extra só é aceita quando passa na validação**.
Um híbrido que não melhora o campeão nas mesmas origens, com ganho e frequência de vitórias
suficientes, é recusado e o modelo mais simples continua. Cada decisão, aceita ou não, fica
registrada com o motivo (``result.decisions``).

Glossário
---------

.. glossary::

   base
      O modelo que prevê a série (``arima``, ``knn``, ``linear`` …). Sozinho, é uma
      candidata ``standalone``.

   corretor
      O modelo que aprende o erro que a base deixa, a partir de uma janela dos resíduos
      anteriores. No híbrido aditivo, ``ŷ = b + α·r̂``.

   resíduo
      O erro da base, ``e = y − b``. É gerado **dentro do treino** de cada origem pela
      fonte de resíduos configurada (``historical_oof_1step`` por padrão: previsões de um
      passo fora da amostra), nunca na série inteira.

   candidata
      Uma configuração completa a avaliar: estratégia, base, corretor, janela, α, fonte de
      resíduos, variante. Tem um ``candidate_id`` estável (hash da configuração).

   estrutura
      A forma de uma candidata, sem os detalhes de configuração: ``arima-mlp`` é a
      estrutura de todas as candidatas com base ARIMA e corretor MLP, qualquer que seja a
      janela ou o α.

   família
      O grupo de modelos usado na seleção incremental: estatística, linear, kernel,
      árvore, neural, residual, decomposição, combinação, seleção dinâmica (e as que
      entram quando usadas). Cada família leva uma finalista ao desafio.

   campeão
      A candidata vencedora até o momento. A seleção começa no melhor modelo individual.

   desafiante
      A finalista de uma família que tenta substituir o campeão em ``V_select``.

   origem
      O ponto da série onde um modelo é ajustado e de onde ele prevê. Nada depois da
      origem entra no ajuste daquela previsão.

   unidade
      Uma origem de uma série (``série:dobra`` num painel). As comparações são pareadas
      por unidade: os dois modelos são avaliados exatamente nas mesmas.

   V_search
      As origens usadas pela busca: avaliar, reter e refinar candidatas, e ajustar
      hiperparâmetros (``tune_model``).

   V_select
      As origens mais recentes do histórico de desenvolvimento, usadas só para decidir o
      campeão. Separar busca e decisão reduz o otimismo de escolher o vencedor onde ele
      foi procurado.

   teste selado
      Os últimos ``test_size`` pontos (padrão ``fh``). Não podem ser lidos pela busca nem
      pela seleção; ``evaluate_external`` os lê uma vez por modelo finalizado.

   cobertura
      A fração das unidades esperadas em que a candidata produziu uma métrica válida. A
      regra de aceite exige cobertura completa.

   ledger
      O registro de todas as propostas da busca, avaliadas ou não (cortadas pelo
      orçamento, puladas por capacidade, duplicadas, interrompidas), com o motivo:
      ``result.search.ledger_frame()``.

   variante
      Como as previsões encontram o futuro: **A**, ``fh`` passos de uma vez a partir da
      origem (padrão); **B**, um passo por vez, com o valor real revelado antes do
      próximo e parâmetros congelados (:doc:`user_guide/variant_b`).

   perfil
      O conjunto de regras da seleção: ``standard`` (padrão), ``quickstart`` (sem o
      critério de variabilidade), ``research_strict`` e ``exploratory``.

   orçamento
      Quantas candidatas a busca pode avaliar: ``smoke``, ``small`` (padrão),
      ``medium`` ou ``large``, no total, por família e por etapa.

Onde cada conceito é detalhado
------------------------------

* Divisão temporal, origens e resíduos: :doc:`user_guide/temporal_validation`.
* Regra de aceite, famílias e perfis: :doc:`user_guide/selection`.
* Etapas, orçamento e ledger: :doc:`user_guide/search_space`.
* Estratégias de montar o sistema híbrido: :doc:`user_guide/strategies`.
* Variantes A e B: :doc:`user_guide/variant_b`.
