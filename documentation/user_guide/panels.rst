Várias séries (painel)
======================

``fit`` aceita um dicionário de séries, um :class:`~autohybts.PanelFrame` ou uma tabela
longa:

.. code-block:: python

   exp = AutoHybridExperiment(fh=14).fit({"loja_a": y_a, "loja_b": y_b})
   exp = AutoHybridExperiment(fh=14).fit(
       long, series_col="series_id", date_col="timestamp", target="value"
   )

Cada série tem teste selado, origens e escala da MASE próprios. As unidades de
comparação são ``série:dobra``.

* **Ranking**: melhor configuração de cada estrutura em cada série, postos por série,
  média dos postos.
* **Seleção**: ganho mediano e vitórias sobre as unidades de todas as séries juntas;
  a variabilidade é a mediana por série e depois entre as séries;
  ``result.selection.champion_per_series`` guarda a configuração de cada série.
* **Finalização**: ``finalize_model`` devolve um :class:`~autohybts.PanelForecaster`,
  um modelo por série, cada um com a sua configuração.
* **Avaliação externa**: uma linha por série mais a média.

.. code-block:: python

   model = exp.finalize_model(result.best)
   exp.predict_model(model).predictions
   exp.evaluate_external(model).metrics
   exp.plot_model(result.best, "cv_forecasts", series_id="loja_b")

A CV aninhada também funciona no painel: cada série é cortada na sua própria origem
externa, o experimento interno roda no painel truncado e a tabela tem uma linha por
dobra e série.

.. code-block:: python

   nested = exp.nested_cross_validate(outer_folds=3)
   nested.folds        # índice (fold_id, series_id)
   nested.summary
