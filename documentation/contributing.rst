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

Idioma do código
----------------

Código, nomes e docstrings em inglês; mensagens de erro, avisos e textos mostrados ao
usuário em português. A referência da API mostra os docstrings em inglês nos dois
idiomas da documentação.

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

Ambiente e testes
-----------------

.. code-block:: bash

   python -m venv .venv
   .venv/bin/python -m pip install -e ".[dev,stats,notebook]"   # Windows: .venv\Scripts\python
   ruff check . && ruff format --check .
   mypy                                  # estrito, todo o pacote
   pytest -m "not slow" --cov            # cobertura mínima: 85 %
   pytest -m leakage                     # uma suíte

* As suítes (``unit``, ``integration``, ``contracts``, ``leakage``, ``regression``) vêm da
  pasta em ``tests/``. ``slow`` fica fora do CI de cada PR; ``network`` só roda com
  ``--run-network``; os notebooks ``heavy``, só com ``AUTOHYBTS_RUN_HEAVY=1``.
* ``import autohybts`` não pode carregar statsmodels, torch, optuna, matplotlib, plotly,
  yaml nem scikit-learn (``tests/contracts/test_import_minimal.py``): importe bibliotecas
  pesadas dentro das funções.
* Os modelos do StatsForecast exigem pandas < 3: teste-os num segundo ambiente com o
  extra ``nixtla-stats``.

Acrescentar um modelo embutido
------------------------------

1. Crie uma subclasse de ``ForecasterBase`` (``autohybts.adapters.base``) e guarde cada
   argumento do ``__init__`` num atributo de mesmo nome: é assim que ``get_params`` e
   ``clone`` funcionam.
2. Implemente ``_fit(y)`` e ``_predict(horizon, history)``. A previsão parte do contexto
   observado com os parâmetros congelados: nunca reajuste dentro da previsão (é o que a
   variante B e os resíduos fora da amostra exigem).
3. Importe a biblioteca dentro dos métodos e declare o extra em
   ``Capabilities(requires_extra=...)``.
4. Registre em ``registry.default_registry()`` com um nome descritivo; nomes curtos
   colidem com modelos registrados por usuários (confira notebooks e testes).
5. Os testes de contrato (``tests/contracts/test_base_forecaster.py``) e de causalidade
   (``tests/leakage/test_base_causality.py``) cobrem automaticamente todo modelo
   registrado; se o padrão for lento, acrescente parâmetros rápidos em ``FAST_PARAMS``.
6. Mantenha o modelo fora da busca padrão até medi-lo e descreva-o em
   :doc:`user_guide/models` (as tabelas de :doc:`user_guide/model_reference` são geradas).

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
