Dados e séries de referência
============================

Copie o nome e passe para ``get_data``; o resultado é uma ``pd.Series`` com
``DatetimeIndex`` regular, pronta para o ``fit``.

.. code-block:: python

   from autohybts.datasets import get_data

   y = get_data("ozone")

.. list-table::
   :header-rows: 1
   :widths: 18 40 10 8 24

   * - Nome
     - O que é
     - Frequência
     - Pontos
     - Origem
   * - ``synthetic_daily``
     - sintética: tendência, sazonalidade semanal e ruído AR(1)
     - diária
     - 730
     - gerada pela AutoHybTS
   * - ``white_noise_daily``
     - sintética: nível + ruído branco (controle negativo)
     - diária
     - 730
     - gerada pela AutoHybTS
   * - ``intermittent_daily``
     - sintética: demanda esparsa (venda em ~25% dos dias, tamanho 1 + Poisson)
     - diária
     - 730
     - gerada pela AutoHybTS
   * - ``commodity_price_daily``
     - sintética: preço diário de uma commodity (safra anual, choques persistentes e choque de oferta)
     - diária
     - 1095
     - gerada pela AutoHybTS
   * - ``air_passengers``
     - passageiros aéreos, 1949–1960
     - mensal
     - 144
     - ``datasets::AirPassengers``
   * - ``nottem``
     - temperatura em Nottingham, 1920–1939
     - mensal
     - 240
     - ``datasets::nottem``
   * - ``us_acc_deaths``
     - mortes acidentais nos EUA, 1973–1978
     - mensal
     - 72
     - ``datasets::USAccDeaths``
   * - ``sunspots_monthly``
     - manchas solares, 1749–1983
     - mensal
     - 2820
     - ``datasets::sunspots``
   * - ``uk_gas``
     - consumo de gás no Reino Unido, 1960–1986
     - trimestral
     - 108
     - ``datasets::UKgas``
   * - ``johnson_johnson``
     - lucro por ação da Johnson & Johnson, 1960–1980
     - trimestral
     - 84
     - ``datasets::JohnsonJohnson``
   * - ``ozone``
     - ozônio total em Halley Bay (Antártida), 1956–2000
     - anual
     - 45
     - ``DAAG::ozone``
   * - ``nile``
     - vazão do rio Nilo, 1871–1970
     - anual
     - 100
     - ``datasets::Nile``
   * - ``lynx``
     - linces capturados no Canadá, 1821–1934
     - anual
     - 114
     - ``datasets::lynx``
   * - ``lake_huron``
     - nível do lago Huron, 1875–1972
     - anual
     - 98
     - ``datasets::LakeHuron``
   * - ``sunspots``
     - manchas solares, 1700–1988
     - anual
     - 289
     - ``datasets::sunspot.year``
   * - ``m4:D1``
     - M4; troque ``D1`` por ``Y…``, ``Q…``, ``M…``, ``W…``, ``D…`` ou ``H…``
     - várias
     - —
     - competição M4 (GitHub)
   * - ``m3:N1402``
     - M3; exige o extra ``datasets``
     - várias
     - —
     - pacote R ``Mcomp`` (CRAN)

Maiúsculas e minúsculas não importam (``get_data("Ozone")`` também funciona). A lista
está em ``autohybts.datasets.NAMED_SERIES``.

De onde vêm os dados
--------------------

A biblioteca não embute nenhuma série real: cada uma é baixada na primeira vez que é
pedida, guardada em cache e lida dali nas vezes seguintes. São quatro origens:

.. list-table::
   :header-rows: 1
   :widths: 16 48 36

   * - Provedor
     - Fonte
     - Licença
   * - sintéticas
     - geradas na hora por ``autohybts.datasets.synthetic`` (numpy, ``seed=42``);
       nada é baixado
     - parte da AutoHybTS
   * - ``rdatasets``
     - `Rdatasets <https://vincentarelbundock.github.io/Rdatasets/>`_: um site que
       publica em CSV os conjuntos de dados de centenas de pacotes R. O endereço de cada
       CSV vem do índice ``Rdatasets/datasets.csv``
     - a do pacote R de origem
   * - ``m4``
     - CSVs oficiais da competição M4 no repositório
       `Mcompetitions/M4-methods <https://github.com/Mcompetitions/M4-methods>`_
       (pasta ``Dataset``): ``M4-info.csv``, ``<Freq>-train.csv`` e ``<Freq>-test.csv``
     - dados da competição M4 (Makridakis et al.)
   * - ``m3``
     - arquivo do pacote R `Mcomp <https://cran.r-project.org/package=Mcomp>`_ 2.8 no
       CRAN, lido em Python pela biblioteca ``rdata`` (extra ``datasets``); o R não é
       necessário
     - GPL-3 (pacote ``Mcomp``); dados da competição M3 (IIF)

Na coluna *Origem*, ``pacote::nome`` é o pacote R e o nome do objeto, como se escreve no
R. ``datasets`` é o pacote que já vem com o R; ``DAAG`` é o pacote do livro *Data
Analysis and Graphics Using R*. Os dois chegam pelo Rdatasets, e o R não precisa estar
instalado. ``get_data("ozone")`` equivale a
``load_dataset("rdatasets", package="DAAG", name="ozone", target="Annual",
time_col="Year", frequency="YS")``. O manifesto de cada arquivo no cache guarda a URL, a
licença e a versão do provedor.

Para ver o catálogo completo de cada provedor:

.. code-block:: python

   from autohybts.datasets import list_datasets

   list_datasets()                                  # provedores (sem download)
   list_datasets("m4", subset="daily")
   list_datasets("rdatasets", search="ozone")

Cache e segurança
-----------------

* Os arquivos vão para ``~/.cache/autohybts/datasets`` (ou ``AUTOHYBTS_DATA_DIR``), uma
  pasta por provedor. Nada é baixado na instalação nem no ``import``.
* Só HTTPS, só domínios aprovados por provedor, com tempo limite, poucas tentativas e
  tamanho máximo. Redirecionamentos também precisam ficar em HTTPS e nos domínios
  aprovados.
* Nomes de pacote e de dataset viram nome de arquivo no cache e só aceitam letras,
  dígitos e ``._+-`` (nada de ``/``, ``\`` ou ``..``).
* Cada arquivo tem um manifesto (URL, data de acesso, SHA-256, tamanho, licença,
  versão). Um arquivo alterado depois do download é recusado
  (``DatasetIntegrityError``); os arquivos do M4 diário e o pacote ``Mcomp`` têm o
  checksum fixado no código.
* ``offline=True`` usa só o cache e diz o que falta (``DatasetUnavailableError``).
* Arquivos sem hash fixado são confiados no primeiro download; faça esse download numa
  rede confiável.

M4
--

.. code-block:: python

   from autohybts import AutoHybridExperiment
   from autohybts.datasets import list_datasets, load_benchmark

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

   from autohybts.datasets import load_benchmark

   m3 = load_benchmark("m3", subset="monthly")      # BenchmarkDataset
   panel = m3.panel(m3.series_ids[:5])              # PanelFrame, testes oficiais no fim

Outras séries do Rdatasets
--------------------------

Fora da tabela acima, o Rdatasets é um catálogo heterogêneo: a biblioteca não adivinha
qual coluna é o tempo nem qual é o alvo. Olhe a tabela bruta e diga as duas, com a
frequência:

.. code-block:: python

   from autohybts.datasets import load_dataset, raw_dataset

   raw_dataset("rdatasets", package="datasets", name="AirPassengers")   # colunas: time, value
   air = load_dataset("rdatasets", package="datasets", name="AirPassengers",
                      target="value", time_col="time", frequency="MS")
   y = air.full()                                   # load_dataset devolve um BenchmarkSeries

Tempo em anos decimais (objetos ``ts`` do R) é convertido para ``YS``, ``QS`` ou ``MS``
e conferido; datas em texto passam por ``pd.to_datetime``.

Tabelas sem coluna de data, como ``airquality`` (só ``Month`` e ``Day``), são montadas
à mão a partir da tabela bruta:

.. code-block:: python

   import pandas as pd

   d = raw_dataset("rdatasets", package="datasets", name="airquality")
   idx = pd.to_datetime(dict(year=1973, month=d["Month"], day=d["Day"]))
   oz = pd.Series(d["Ozone"].to_numpy(), index=idx, name="Ozone").asfreq("D")
   oz = oz.interpolate(limit_direction="both")      # 37 dias faltando: escolha como tratar

.. note::

   Por padrão, ``missing="reject"`` recusa valores ausentes no ``fit``
   (``DataValidationError``). Para preenchimento causal explícito,
   ``missing="ffill"`` usa somente o último valor observado; lacunas iniciais sem
   histórico continuam inválidas. A interpolação acima é uma escolha de preparação,
   que pode usar valores posteriores e não deve atravessar uma origem de validação.
   Veja :doc:`diagnostics` para o contrato de tratamento de ausentes.

.. warning::

   A licença de cada dataset do Rdatasets é a do pacote R de origem. A biblioteca não
   redistribui dados: consulte a origem antes de republicar.

.. seealso::

   Séries com covariáveis (``load_exog_dataset``): veja :doc:`exogenous`.

   Tutoriais: :doc:`../tutorials/notebooks/05_benchmark_m4`, :doc:`../tutorials/notebooks/17_benchmark_study`.

.. seealso::

   Tutorial executável do piloto M4: auditoria, custo e retomada.
   :doc:`../tutorials/notebooks/22_m4_development_pilot`.
