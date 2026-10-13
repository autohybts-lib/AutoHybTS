Contribuindo
============

Como acrescentar um modelo, uma estratégia ou um tutorial sem perder as garantias da
biblioteca: causalidade, teste selado intocado e decisões auditáveis.

Tutoriais (notebooks)
---------------------

Os tutoriais em ``notebooks/`` (português) e ``notebooks/en/`` (inglês) são o contrato
executável da API pública e ficam no repositório **com as saídas**, para serem lidos sem
executar.

Modelo de aula
~~~~~~~~~~~~~~

Um tutorial novo segue esta ordem:

1. **Objetivo e requisitos**: tema, referência científica, issue ou PR relacionado, variantes A/B
   suportadas, dependências.
2. **Dados e protocolo**: origem, frequência, ``fh``, ``test_size``, ``fold``, sementes e
   orçamento. A busca nunca toca o teste selado.
3. **Exploração causal**: série, sazonalidade e escalas, só com o que existe na origem.
4. **Controles e modelos propostos**: mesma métrica, mesmo horizonte, mesmas origens; o
   melhor modelo isolado e uma combinação simples são os controles mínimos.
5. **Busca e seleção**: API pública, ``V_search`` para a busca, ``V_select`` para a
   decisão, custo em candidatas, unidades e ajustes.
6. **Resultados e gráficos**: previsões contra observações, erros, pesos, ganhos, aceites
   e recusas.
7. **Diagnóstico e limites**: riscos de vazamento, cobertura, falhas, custo.
8. **Avaliação externa**: uma única leitura do teste selado, depois de congelar a escolha.
9. **Leitura**: escrita depois da execução, com os números dela; o que venceu, contra
   quais controles, com quantas unidades e com quais limites.

O esqueleto sai pronto do gerador (``--lang en`` grava a versão em inglês em
``notebooks/en/``):

.. code-block:: bash

   python notebooks/new_notebook.py 16_temporal_bagging "Bagging temporal" \
       --guide user_guide/ensembles --extras plot \
       --requires autohybts.AutoHybridExperiment.compare_models

Metadados
~~~~~~~~~

Cada notebook declara ``metadata.autohybts`` com estes campos, conferidos por
``tests/contracts/test_notebook_metadata.py``:

.. list-table::
   :header-rows: 1
   :widths: 14 86

   * - Campo
     - Significado
   * - ``requires``
     - nomes públicos que o notebook chama (``autohybts.AutoHybridExperiment.create_model``
       ...); enquanto algum não existe, o teste de execução pula o notebook
   * - ``network``
     - baixa dados na primeira execução (os testes só o executam com ``--run-network``)
   * - ``heavy``
     - leva muitos minutos; os testes só o executam com ``AUTOHYBTS_RUN_HEAVY=1``
   * - ``extras``
     - extras do ``pyproject.toml`` necessários (``plot``, ``stats``, ``boosting`` ...);
       sem eles o teste pula e diz qual falta; um pulo nunca conta como execução
   * - ``seeds``
     - sementes que tornam a execução reprodutível
   * - ``guide``
     - páginas desta documentação que o notebook ilustra (devem existir)
   * - ``template``
     - segue o modelo de aula: abre com um título e termina com a seção de leitura
       ("Leitura" em português, "Reading" em inglês)

Execução e saídas
~~~~~~~~~~~~~~~~~

Execute o notebook a partir da sua pasta (``nbclient`` com
``resources={"metadata": {"path": "notebooks"}}``, ou ``"notebooks/en"`` para os em
inglês) e salve as saídas. Antes de salvar,
confira figuras e tabelas; as saídas não podem conter caminhos locais
(``tests/contracts/test_notebook_outputs.py``), chaves, dados privados ou artefatos
grandes. O build da documentação não executa notebooks: ele mostra as saídas salvas.

Traduções
---------

A documentação existe em português (``documentation/``) e em inglês
(``documentation/en/``), com os mesmos nomes de arquivo página a página; os tutoriais, em
``notebooks/`` e ``notebooks/en/``. Uma mudança numa página ou num notebook entra nos dois
idiomas no mesmo PR. ``tests/contracts/test_docs_translation.py`` confere que as duas
árvores têm as mesmas páginas e os mesmos notebooks, com os mesmos metadados. Compile as
duas antes de abrir o PR:

.. code-block:: bash

   sphinx-build -W documentation documentation/_build/html
   sphinx-build -W documentation/en documentation/_build/html-en

Publicação no Read the Docs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

O idioma configurado no painel define o segmento da URL (``pt-br`` ou ``en``),
mas não escolhe a pasta que o Sphinx compila. Configure o idioma e o arquivo de
build de acordo com a mesma linha da tabela:

.. list-table::
   :header-rows: 1

   * - Idioma no painel (Language)
     - Arquivo de build (Build configuration file)
     - Configuração Sphinx no YAML
   * - Portuguese (Brazil)
     - ``documentation/.readthedocs.yaml``
     - ``documentation/conf.py``
   * - English
     - ``.readthedocs.yaml``
     - ``documentation/en/conf.py``

Para publicar a documentação em
``https://autohybts.readthedocs.io/pt-br/latest/``:

1. Abra o projeto que publica o português no painel do Read the Docs e entre em
   **Admin → Settings** (ou **Settings**).
2. Confira o repositório: ``https://github.com/jvataidee/AutoHybTS``.
3. Defina **Language** como **Portuguese (Brazil)**.
4. Em **Build configuration file**, informe ``documentation/.readthedocs.yaml``.
   Esse campo recebe o YAML; o caminho do ``conf.py`` já está dentro dele.
5. Defina **Default branch** como ``main`` e confira que essa branch contém os
   arquivos de documentação. Uma alteração apenas em outra branch não entra em
   ``latest``; integre-a à ``main`` para a publicação definitiva. Para uma prévia,
   ative a branch desejada em **Versions** e compile essa versão.
6. Salve e, em **Versions**, confira que ``latest`` está ativa. Em **Builds**,
   execute **Build version** para ``latest``.
7. No log, confira o commit compilado e que o comando Sphinx usa
   ``documentation/conf.py``. Após o build terminar com sucesso, abra a URL acima.

Os dois YAMLs já instalam o pacote com o extra ``docs`` em Python 3.12 e Ubuntu
24.04. Todos os caminhos dentro deles são relativos à raiz do repositório,
inclusive no YAML que fica em ``documentation/``. Os notebooks são copiados durante
o build e renderizados com as saídas salvas, sem execução.

Para oferecer os dois idiomas no mesmo domínio, crie um projeto Read the Docs por
idioma, usando o mesmo repositório e os valores da tabela. Escolha um como principal
e adicione o outro em **Translations**, no painel do principal. Se ``autohybts``
continuar como principal em português, o projeto em inglês pode ser cadastrado como
tradução dele. Se o inglês for o principal, o projeto em português será a tradução.
Mantenha as mesmas versões ativas nos dois projetos e compile ambas.

Se aparecer inglês em ``/pt-br/``, confira o **Build configuration file**: o YAML
da raiz compila inglês mesmo que o painel indique português. Se o conteúdo estiver
antigo, confira a branch e o commit no último build bem-sucedido.

Referências: `arquivo de configuração em subpasta
<https://docs.readthedocs.com/platform/stable/guides/setup/monorepo.html>`_,
`idiomas e traduções
<https://docs.readthedocs.com/platform/stable/localization.html>`_ e
`versões e branches
<https://docs.readthedocs.com/platform/stable/versions.html>`_.

Documentação de uma estratégia
------------------------------

A página do guia de uma estratégia nova traz:

* definição e fórmula ou pseudocódigo;
* referência científica e diferença para abordagens próximas;
* parâmetros e o espaço que a busca explora;
* modelos e dados compatíveis, variantes A e B, horizonte;
* como ajusta, de onde vêm os resíduos ou previsões fora da amostra e quando os pesos
  mudam;
* custo, comportamento em falhas e limites conhecidos;
* exemplo executável com resultado real, nunca inventado;
* links para o notebook, o teste de causalidade e a seção de benchmarks.

Entrega
-------

Uma estratégia nova está pronta quando:

* o algoritmo está registrado com capacidades e incompatibilidades explícitas;
* passa nos testes unitários, de contrato temporal, de integração e de vazamento;
* foi comparada com o melhor modelo isolado, o híbrido correspondente e uma combinação
  simples, nas mesmas origens;
* tem página do guia, notebook com saídas revisadas, referência de API e navegação
  atualizadas, nos dois idiomas, e a documentação compila sem avisos
  (``sphinx-build -W documentation documentation/_build/html`` e o mesmo para
  ``documentation/en``);
* custos, versões, sementes e resultados estão registrados, e as decisões (aceites e
  recusas, com motivos) aparecem nos relatórios;
* famílias compostas novas ficam fora do ``include`` padrão até passarem pelo portão de
  calibração (``autohybts.benchmarks.calibration``).
