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
  corretor que quase não mexe na previsão é inerte. O valor é a mediana por série
  (sobre as unidades comparadas) e depois a mediana entre as séries; uma série sem
  diagnóstico finito aparece como ``NaN`` em ``variability_by_series`` da decisão e
  fica fora da mediana (versão 2 do critério).

Opcionalmente, ``AutoHybridExperiment(significance_alpha=0.05)`` acrescenta um critério
estatístico: o desafiante também precisa vencer num teste de Wilcoxon pareado unilateral
nas mesmas unidades, ``p < alfa`` (motivo ``not_significant``; o p-valor aparece na
coluna ``p_value`` das decisões). Unidades com erros iguais saem do teste; com ``n``
unidades restantes (todas as séries juntas) o menor p-valor possível é ``1/2ⁿ``: são
precisas ao menos 5 para ``alfa = 0,05``. Com o ``fold=3`` padrão numa série, o
critério **nunca** passa — aumente ``fold``. O teste controla cada comparação, não a
cadeia inteira: várias famílias desafiam um campeão que muda com as decisões
anteriores, e origens vizinhas não são independentes.

Rejeição mantém o campeão. Cada decisão fica registrada, aceita ou não
(``result.selection.decision_units_frame()`` traz, por decisão e unidade, os dois
erros, o ganho e a vitória, para recalcular ou reamostrar fora da biblioteca):

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
   branco ela aceitou híbridos espúrios em parte das sementes testadas. O número de
   origens em ``V_select`` pesa muito: com 2 unidades, 60 % de vitórias exige vencer as
   duas, o que acontece por acaso em cerca de 1 de cada 4 comparações. Meça o
   comportamento da cadeia com ``autohybts.benchmarks.calibration`` e use a comparação
   e a CV aninhada para relatar a incerteza.
