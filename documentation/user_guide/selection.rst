Seleção incremental
===================

A seleção começa no melhor modelo individual e visita as famílias na ordem
``family_order`` (estatística, linear, kernel, árvore, neural, residual, decomposição,
combinação). A finalista de cada família — sua melhor estrutura na busca — desafia o
**campeão do momento** em ``V_select``.

Regra de aceite
---------------

O desafiante substitui o campeão só se, nas mesmas unidades (origens):

* tiver cobertura completa e métricas finitas;
* **ganho mediano** pareado ``(E_campeão − E_desafiante) / |E_campeão|`` ≥ 1 %;
* **taxa de vitórias** ≥ 60 % (empates não contam);
* **variabilidade** ``correction_std_ratio`` ≥ 0,05 quando houver corretor — um
  corretor que quase não mexe na previsão é inerte.

Opcionalmente, ``AutoHybridExperiment(significance_alpha=0.05)`` acrescenta um critério
estatístico: o desafiante também precisa vencer num teste de Wilcoxon pareado unilateral
nas mesmas unidades, ``p < alfa`` (motivo ``not_significant``; o p-valor aparece na
coluna ``p_value`` das decisões). Com ``n`` unidades o menor p-valor possível é
``1/2ⁿ``: são precisas ao menos 5 unidades de seleção para ``alfa = 0,05``.

Rejeição mantém o campeão. Cada decisão fica registrada, aceita ou não:

.. code-block:: python

   result.decisions
   #   family    champion_before  challenger   accepted  median_gain  win_rate ...
   #   residual  arima            arima-mlp    False     -0.07        0.33

Perfis
------

``quickstart``
   critério de variabilidade desligado (e dito explicitamente).
``standard``
   regra completa (padrão).
``research_strict``
   regra completa e cobertura total obrigatória.
``exploratory``
   resultados marcados como exploratórios na auditoria.

Ranking
-------

Na busca e entre séries, cada estrutura é representada pela sua melhor configuração;
estruturas são ordenadas por posto (empates com posto médio), depois erro agregado,
cobertura, custo e ``candidate_id``. Num painel, os postos são calculados por série e
promediados.

.. warning::

   Sem ``significance_alpha``, a regra é um protocolo, não um teste de hipótese: em ruído
   branco ela aceitou híbridos espúrios em parte das sementes testadas. Use o critério
   estatístico, a comparação e a CV aninhada para controlar e relatar a incerteza.
