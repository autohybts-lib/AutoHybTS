Início rápido
=============

Um experimento AutoHybTS segue o fluxo do PyCaret: criar, ajustar, comparar, finalizar,
prever. Cada método devolve um resultado imutável.

1. Criar o experimento e selar o teste
--------------------------------------

.. code-block:: python

   from autohybts import AutoHybridExperiment
   from autohybts.datasets import get_data

   y = get_data("synthetic_daily")            # pd.Series com DatetimeIndex regular
   exp = AutoHybridExperiment(fh=14, session_id=42).fit(y)
   exp.splits                                  # origens de V_search e V_select

``fit`` valida a série, separa os últimos ``test_size`` pontos (padrão: ``fh``) como
**teste selado** e congela as origens de validação. O teste selado só é lido por
``evaluate_external``.

2. Explorar os dados
--------------------

.. code-block:: python

   exp.data_profile()
   exp.plot_data("seasonality")                # também: series, acf, distribution

3. Comparar modelos e híbridos
------------------------------

.. code-block:: python

   result = exp.compare_models()
   result.leaderboard        # cada candidata da busca (nome curto, ex.: arima-mlp)
   result.decisions          # cadeia incremental: desafiante, ganho, vitórias, motivo
   result.best               # o campeão

A busca roda em ``V_search``; a escolha do campeão, em ``V_select``. Para incluir as
outras estratégias de montar o sistema:

.. code-block:: python

   result = exp.compare_models(
       include=("standalone", "residual", "multistage", "decomposition", "combination")
   )
   result.strategies
   result.plot("pareto")

4. Inspecionar um modelo
------------------------

.. code-block:: python

   hybrid = exp.create_hybrid("arima", "mlp")
   hybrid.cv_results
   exp.plot_model(hybrid, "cv_forecasts")      # também: residuals, metric_by_horizon …

5. Finalizar, prever e avaliar uma vez
--------------------------------------

.. code-block:: python

   final = exp.finalize_model(result.best)     # reajuste no histórico anterior ao teste
   exp.predict_model(final).predictions
   exp.evaluate_external(final).metrics        # uma única avaliação por modelo

6. Salvar
---------

.. code-block:: python

   from autohybts import load_model, save_model

   path = save_model(final, "artifacts/champion")
   model = load_model(path, trusted=True)

Previsão operacional
--------------------

Sem teste selado (``test_size=0``), ou com o estimador de uma chamada:

.. code-block:: python

   from autohybts import AutoHybridForecaster

   forecaster = AutoHybridForecaster(fh=14).fit(y)
   forecaster.predict()
