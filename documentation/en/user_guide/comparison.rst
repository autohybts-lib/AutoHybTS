Comparing models and strategies
===============================

Every comparison is **reporting**: nothing here feeds the search or the selection rule.

Across strategies: Pareto
-------------------------

Each candidate of the search is a point: error on ``V_search`` against cost (seconds
of fitting and forecasting) or complexity (number of models fitted in the structure).
The **Pareto front** joins the candidates that no other one beats on both axes.

.. code-block:: python

   result.strategies                     # per strategy: best, median, cost
   result.strategy_table()               # every candidate, with Pareto flags
   result.plot("pareto")                 # error × cost (log scale)
   result.plot("pareto", x="complexity") # error × complexity
   result.plot("strategies")             # scatter per strategy

Across finalists
----------------

.. code-block:: python

   cmp = result.comparison               # finalists on V_select
   cmp.summary                           # mean, deviation, mean rank
   cmp.win_rates                         # matrix of paired wins
   cmp.tests()                           # Wilcoxon and Diebold-Mariano, Holm
   cmp.friedman()                        # Friedman + Nemenyi critical difference
   cmp.kruskal()                         # Kruskal-Wallis + Dunn (Holm); ignores the pairing
   result.plot("critical_difference")    # also: leaderboard, boxplot, win_rate …

Cross-validation
----------------

``exp.cross_validate`` builds its own folds and returns the same kind of comparison as
this page (``cv.summary``, ``cv.plot("boxplot")`` …); the folds, nested CV and what each
one estimates are in :doc:`temporal_validation`.

On the sealed test
------------------

.. code-block:: python

   exp.compare_external()                # finalizes and evaluates each model once

.. seealso::

   Tutorials: :doc:`../tutorials/notebooks/06_compare_and_cross_validate`.
