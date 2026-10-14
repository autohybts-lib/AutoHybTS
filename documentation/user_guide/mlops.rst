Modelos em produção (MLOps)
===========================

O mesmo ``ExperimentStore`` que registra experimentos guarda o ciclo de vida de um modelo em
operação: registro, promoção, previsões feitas em produção, observações que chegaram depois
e o erro real ao longo do tempo. A biblioteca fornece essas peças; quando treinar, servir
e retreinar (agendador, cron, Airflow …) fica a cargo do seu código.

Registrar e promover
--------------------

.. code-block:: python

   from autohybts import AutoHybridExperiment, ExperimentStore, save_model

   store = ExperimentStore("artifacts/mlops.db")

   exp = AutoHybridExperiment(fh=1, variant="B", test_size=60).fit(history)
   final = exp.finalize_model(exp.compare_models().best)
   mae = float(exp.evaluate_external(final).metrics["MAE"].iloc[0])

   run_id = store.log(exp, name="treino", models=[final], values=True)
   path = save_model(final, "artifacts/modelos/v1")
   store.register_model(final, path, name="preco", run_id=run_id, reference_mae=mae)
   store.promote(final.model_id, reason="primeiro modelo")

``register_model`` guarda o caminho do artefato e o SHA-256 do manifesto, nunca o modelo,
e entra como ``candidate``. ``promote`` move o modelo para ``production`` (o anterior de
mesmo nome vai para ``archived``) ou para outro estágio; toda troca fica em
``stage_history`` com o motivo. ``reference_mae`` é o erro esperado em produção, em geral
o MAE do teste selado.

Servir
------

.. code-block:: python

   model = store.load("preco", trusted=True)            # confere o SHA-256 registrado
   model = model.update_state(new_observations)          # alcança o presente, sem reajuste
   stream = model.stream()

   ticket = stream.predict_next()
   store.log_forecasts(model.model_id, pd.DataFrame(
       {"timestamp": [ticket.timestamp], "origin": [ticket.origin], "y_pred": [ticket.prediction]}
   ))
   stream.observe(value)                                  # quando o valor chega
   store.log_actuals(model.history.series_id, observed)   # pd.Series indexada por data

``load`` exige ``trusted=True``, como ``load_model``. O hash registrado fica no mesmo banco
que aponta para o artefato, então detecta corrupção, não adulteração; para isso, salve com
``signing_key`` e passe a mesma chave a ``load``. ``update_state`` acrescenta observações
sem reajustar os parâmetros. ``log_forecasts`` aceita também as tabelas de
``predict_model`` e de ``ForecastStream.log()``; uma observação registrada de novo para a
mesma data substitui a anterior.

Monitorar
---------

.. code-block:: python

   status = store.monitor(model.model_id, window=30, threshold=1.5)
   status.alert.iloc[-1]

Uma linha por previsão já observada, com o erro absoluto, o MAE das últimas ``window``
previsões, a razão para ``reference_mae`` e ``alert``: janela completa com MAE acima de
``threshold`` vezes a referência. Sem referência, não há alerta.

Retreinar com portão
--------------------

Treine um desafiante com o histórico atual, deixando os últimos dias selados, e compare o
erro dele nesses dias com o erro real da produção nas mesmas datas, que já está no banco:

.. code-block:: python

   live = store.query(
       """
       SELECT AVG(ABS(a.y_true - f.y_pred)) AS mae
       FROM forecasts f JOIN actuals a USING (series_id, timestamp)
       WHERE f.model_id = ?
         AND f.timestamp IN (SELECT timestamp FROM predictions WHERE run_id = ?)
       """,
       (production_id, challenger_run_id),
   ).mae[0]
   if challenger_mae <= live:
       store.promote(challenger.model_id, reason=f"MAE {challenger_mae:.3f} <= {live:.3f}")

As duas medidas são fora da amostra e nas mesmas datas. A regra "não pior" troca, num
empate, o modelo por um treinado com dados mais recentes, o que também atualiza a
referência do monitoramento depois de uma mudança de regime.

Tabelas de operação
-------------------

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Tabela
     - Conteúdo
   * - ``registry``
     - Um modelo por linha: nome, execução de origem, candidata, rótulo, variante, série,
       caminho do artefato, SHA-256, estágio, ``reference_mae`` e data de registro.
   * - ``stage_history``
     - Cada mudança de estágio, com data e motivo.
   * - ``forecasts``
     - Previsões feitas em produção: modelo, série, origem, data prevista, horizonte e valor.
   * - ``actuals``
     - Observações de cada série por data.

Essas tabelas não pertencem a uma execução: ``log`` e ``delete_run`` não as alteram.

Limites
-------

* O agendamento e a decisão de quando retreinar ficam no seu código; a biblioteca registra,
  carrega e mede.
* ``monitor`` mede o MAE. Outras métricas podem ser calculadas com SQL sobre ``forecasts``
  e ``actuals``.
* O banco é um arquivo local, sem objetivo de acesso concorrente pesado.

.. seealso::

   Tutorial: :doc:`../tutorials/notebooks/19_mlops_flow`.
