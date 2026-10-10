Comparação de modelos e estratégias
===================================

Toda comparação é **relatório**: nada aqui alimenta a busca ou a regra de seleção.

Entre estratégias: Pareto
-------------------------

Cada candidata da busca é um ponto: erro em ``V_search`` contra custo (segundos de
ajuste e previsão) ou complexidade (número de modelos ajustados na estrutura). A
**fronteira de Pareto** liga as candidatas que nenhuma outra supera nos dois eixos.

.. code-block:: python

   result.strategies                     # por estratégia: melhor, mediana, custo
   result.strategy_table()               # cada candidata, com flags de Pareto
   result.plot("pareto")                 # erro × custo (escala log)
   result.plot("pareto", x="complexity") # erro × complexidade
   result.plot("strategies")             # dispersão por estratégia

Entre finalistas
----------------

.. code-block:: python

   cmp = result.comparison               # finalistas em V_select
   cmp.summary                           # média, desvio, posto médio
   cmp.win_rates                         # matriz de vitórias pareadas
   cmp.tests()                           # Wilcoxon e Diebold-Mariano, Holm
   cmp.friedman()                        # Friedman + diferença crítica de Nemenyi
   result.plot("critical_difference")    # também: leaderboard, boxplot, win_rate …

Validação cruzada
-----------------

.. code-block:: python

   cv = exp.cross_validate(["naive", "arima", result.best], folds=8, step=7)
   cv.summary
   cv.plot("boxplot")

No teste selado
---------------

.. code-block:: python

   exp.compare_external()                # finaliza e avalia cada modelo uma vez
