Ajuste de hiperparâmetros
=========================

``tune_model`` ajusta os parâmetros da base de um modelo **só em V_search**. O resultado
é avaliado nas duas camadas como ``create_model``; o ``V_select`` dele não foi visto pelo
ajuste, então a comparação com o original continua honesta.

.. code-block:: python

   tuned = exp.tune_model("knn", n_trials=20)
   tuned, tuner = exp.tune_model("knn", n_trials=20, return_tuner=True)
   tuner.trials            # cada configuração e seu escore
   tuner.improvement       # ganho relativo em V_search

* ``sampler="auto"`` usa Optuna (TPE, com semente) se o extra ``optuna`` estiver
  instalado; senão, busca aleatória com semente. ``"optuna"`` e ``"random"`` forçam um.
* A configuração original é a tentativa 0: o ajuste nunca fica pior que ela em
  ``V_search``.
* Espaços embutidos para linear, knn, svm, tree, random_forest e mlp. Espaço próprio:

.. code-block:: python

   from autohybts.search.tuning import Distribution

   exp.tune_model(
       "svm",
       search_space={
           "lags": (7, 28),                              # inteiros: intervalo inteiro
           "estimator_params.C": Distribution("log_float", 0.01, 100),
           "estimator_params.epsilon": [0.01, 0.1, 0.5], # lista: opções
       },
   )

.. seealso::

   Tutoriais: :doc:`../tutorials/notebooks/08_advanced_families_and_tuning`.
