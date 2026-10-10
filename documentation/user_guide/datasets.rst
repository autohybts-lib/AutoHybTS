Datasets e benchmarks
=====================

Séries de exemplo offline e benchmarks baixáveis sob demanda.

.. code-block:: python

   from autohybts.datasets import get_data, list_datasets, load_benchmark, load_dataset

   list_datasets()                                  # provedores (sem download)
   get_data("synthetic_daily")                      # offline
   get_data("m4:D1")                                # série M4 completa (teste oficial no fim)

Cache e segurança
-----------------

* Os arquivos vão para ``~/.cache/autohybts/datasets`` (ou ``AUTOHYBTS_DATA_DIR``), uma
  pasta por provedor. Nada é baixado na instalação nem no ``import``.
* Só HTTPS, só domínios aprovados por provedor, com tempo limite, poucas tentativas e
  tamanho máximo.
* Cada arquivo tem um manifesto (URL, data de acesso, SHA-256, tamanho, licença,
  versão). Um arquivo alterado depois do download é recusado
  (``DatasetIntegrityError``); os arquivos do M4 diário e o pacote ``Mcomp`` têm o
  checksum fixado no código.
* ``offline=True`` usa só o cache e diz o que falta (``DatasetUnavailableError``).

M4
--

.. code-block:: python

   list_datasets("m4", subset="daily")
   series = load_benchmark("m4", subset="daily", series_id="D1")
   series.train, series.test, series.descriptor.horizon

   exp = AutoHybridExperiment(fh=series.descriptor.horizon,
                              test_size=len(series.test)).fit(series.full())

``full()`` junta treino e teste oficial; ``test_size=len(series.test)`` sela o teste
oficial. As datas do M4 vêm de ``M4-info.csv`` (anos com dois dígitos, sabidamente pouco
confiáveis); quando não formam um índice válido, um calendário sintético é usado e
``descriptor.extra["index_source"] == "synthetic"``.

M3
--

Do pacote R ``Mcomp`` (CRAN, GPL-3). Ler o arquivo R requer o extra ``datasets``.

.. code-block:: python

   m3 = load_benchmark("m3", subset="monthly")      # BenchmarkDataset
   panel = m3.panel(m3.series_ids[:5])              # PanelFrame, testes oficiais no fim

Rdatasets
---------

Catálogo heterogêneo: não há inferência de tempo nem de alvo.

.. code-block:: python

   list_datasets("rdatasets", search="passenger")
   raw_dataset("rdatasets", package="datasets", name="AirPassengers")
   air = load_dataset("rdatasets", package="datasets", name="AirPassengers",
                      target="value", time_col="time", frequency="MS")

Tempo em anos decimais (objetos ``ts`` do R) é convertido para ``YS``, ``QS`` ou ``MS``
e conferido; datas em texto passam por ``pd.to_datetime``.

.. warning::

   A licença de cada dataset do Rdatasets é a do pacote R de origem. A biblioteca não
   redistribui dados: consulte a origem antes de republicar.
