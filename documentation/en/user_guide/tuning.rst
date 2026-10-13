Hyperparameter tuning
=====================

``tune_model`` tunes the parameters of a model's base **on V_search only**. The result
is evaluated on both layers like ``create_model``; its ``V_select`` was not seen by the
tuning, so the comparison with the original stays honest.

.. code-block:: python

   tuned = exp.tune_model("knn", n_trials=20)
   tuned, tuner = exp.tune_model("knn", n_trials=20, return_tuner=True)
   tuner.trials            # every configuration and its score
   tuner.improvement       # relative gain on V_search

* ``sampler="auto"`` uses Optuna (TPE, seeded) if the ``optuna`` extra is installed;
  otherwise, a seeded random search. ``"optuna"`` and ``"random"`` force one of them.
* The original configuration is trial 0: tuning never ends up worse than it on
  ``V_search``.
* Built-in spaces for linear, knn, svm, tree, random_forest and mlp. Custom space:

.. code-block:: python

   from autohybts.search.tuning import Distribution

   exp.tune_model(
       "svm",
       search_space={
           "lags": (7, 28),                              # integers: integer range
           "estimator_params.C": Distribution("log_float", 0.01, 100),
           "estimator_params.epsilon": [0.01, 0.1, 0.5], # list: choices
       },
   )
