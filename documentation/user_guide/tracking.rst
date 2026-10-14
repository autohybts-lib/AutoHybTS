Registro de experimentos em SQLite
==================================

``ExperimentStore`` grava experimentos num arquivo SQLite para comparar execuções, séries
e configurações com consultas SQL. Usa só o ``sqlite3`` da biblioteca padrão: não há
dependência nova e nada é gravado sem pedido.

.. code-block:: python

   from autohybts import AutoHybridExperiment, ExperimentStore

   store = ExperimentStore("artifacts/experimentos.db")

   exp = AutoHybridExperiment(fh=14, session_id=42).fit(y)
   cmp = exp.compare_models()
   final = exp.finalize_model(cmp.best)
   exp.evaluate_external(final)

   run_id = store.log(exp, name="baseline", tags={"autor": "ana"}, models=[final])
   store.runs()                                   # uma linha por execução

``log`` grava o que o experimento já calculou; nunca avalia candidatas de novo e nunca lê o
teste selado. Funciona também sem ``compare_models``: um experimento só com
``create_model`` grava as avaliações, sem campeão nem decisões.

O que é gravado
---------------

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Tabela
     - Conteúdo
   * - ``runs``
     - Uma linha por execução: nome, versão, variante, horizonte, perfil, orçamento,
       semente, número de séries e de origens, métrica de ordenação
       (``ranking_metric``), hashes do protocolo e dos dados, campeão, parâmetros,
       etiquetas e snapshot (JSON).
   * - ``series``
     - Identidade de cada série: frequência, início, fim, tamanho e impressão digital
       (sem valores).
   * - ``splits``
     - Origens de cada camada (``search`` e ``select``) por série.
   * - ``candidates``
     - Cada candidata avaliada: rótulo, família, estratégia, base, corretor, janela,
       ``alpha`` e a especificação completa (JSON).
   * - ``evaluations`` e ``metrics``
     - Resumo por candidata e camada (posto médio, cobertura, custo, campeão) e as
       métricas médias em formato longo (``metric``, ``value``).
   * - ``trials`` e ``trial_metrics``
     - Cada origem de cada candidata: estado, tempos, motivo de falha e métricas.
   * - ``ledger``
     - Toda proposta da busca, com estágio e estado (``EVALUATED``, ``PRUNED_BUDGET`` …).
   * - ``decisions``
     - Cada comparação da seleção incremental: desafiante, aceite, motivos, ganho
       mediano, taxa de vitórias e variabilidade.
   * - ``events``
     - Os eventos do experimento, com duração.
   * - ``models``, ``external`` e ``predictions``
     - Modelos finalizados passados em ``models``, métricas da avaliação externa (por
       série; num painel, a média tem ``series_id`` nulo) e, só com ``values=True``,
       previsões com as observações.

Métricas indefinidas são ``NULL``, nunca 0. Os valores da série não são gravados por
padrão, como em ``export_evidence``; ``values=True`` grava as previsões e as observações
do teste externo.

Execuções e novo registro
-------------------------

O ``run_id`` depende dos parâmetros do experimento, dos dados, do plano temporal e do
``name``. Registrar de novo o mesmo experimento com o mesmo nome substitui as linhas da
execução: dá para chamar ``log`` depois de ``compare_models`` e outra vez depois de
``evaluate_external``. Um nome diferente cria outra execução; ``store.delete_run(run_id)``
apaga uma.

Consultas
---------

``store.query`` roda SQL numa conexão somente leitura e devolve um ``DataFrame``. Passe
valores por ``params`` (marcadores ``?``), nunca formatando o texto da consulta.

.. code-block:: python

   # erro de cada candidata na camada de seleção de uma execução
   store.query(
       """
       SELECT c.label, c.family, m.value AS mase
       FROM metrics m JOIN candidates c USING (run_id, candidate_id)
       WHERE m.run_id = ? AND m.layer = 'select' AND m.metric = 'MASE'
       ORDER BY mase
       """,
       (run_id,),
   )

   # campeão e erro externo de cada execução
   store.query(
       """
       SELECT r.name, r.champion_label, x.value AS mase_externo
       FROM runs r JOIN external x USING (run_id)
       WHERE x.metric = 'MASE' AND x.series_id IS NOT NULL
       """
   )

   # por que os desafiantes foram rejeitados
   store.query(
       "SELECT family, reason_codes, COUNT(*) AS n FROM decisions "
       "WHERE accepted = 0 GROUP BY family, reason_codes"
   )

O arquivo é um banco SQLite comum: abre também no cliente ``sqlite3``, no DB Browser for
SQLite ou com ``pandas.read_sql_query``.

Limites
-------

* O banco é um relatório: nunca alimenta a busca, a seleção nem os limiares.
* Modelos não vão para o banco. Salve-os com ``save_model`` e guarde o caminho.
* O esquema é versionado (``autohybts-store/1``, tabela ``meta``); um arquivo de outro
  esquema ou que não seja um banco é recusado.
* É um arquivo local, sem objetivo de acesso concorrente pesado.

.. seealso::

   Tutorial: :doc:`../tutorials/notebooks/18_sql_analysis`.
