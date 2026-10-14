Manutenção
==========

Tarefas de quem publica a biblioteca e a documentação: versões no PyPI, a tag e os
workflows de publicação, o checklist antes da tag e a configuração do Read the Docs.
Quem contribui com código ou documentação encontra o necessário em :doc:`contributing`.

Publicação e instalação da candidata
------------------------------------

Instale a versão publicada diretamente pelo PyPI:

.. code-block:: bash

   python -m pip install "autohybts[stats,plot]==1.0.0rc4"

O guia :doc:`installation` traz instalação do núcleo, extras, notebooks e a opção
de desenvolvimento pelo GitHub.

O workflow ``publish.yml`` é acionado ao enviar uma tag ``v*``. Ele gera o wheel e
o pacote-fonte, confere-os com Twine e aguarda a aprovação do ambiente ``pypi``
antes do envio. O Trusted Publisher cadastrado no PyPI deve corresponder a:

* projeto: ``autohybts``;
* proprietário e repositório: ``jvataidee/AutoHybTS``;
* arquivo de workflow: ``publish.yml``;
* ambiente: ``pypi``.

No GitHub, abra a execução, use **Review deployments**, selecione **pypi** e clique
em **Approve and deploy**. O cadastro de um publisher não inicia um envio; a tag
e a aprovação do ambiente são etapas do fluxo.

O ``release.yml`` mantém um fluxo separado de validação e TestPyPI, que depende de
seu próprio Trusted Publisher. Para próximas versões, execute as verificações
completas antes da tag e escolha um único workflow de publicação por índice.

Cada versão enviada ao PyPI é imutável, incluindo o README incorporado aos
metadados. Mudanças no README do GitHub e no Read the Docs aparecem no PyPI apenas
na próxima versão publicada. A ``1.0.0rc4`` incorpora essa atualização; alterações
posteriores exigem outra versão. Não reenvie uma versão já publicada.

Checklist local
---------------

.. code-block:: bash

   ruff check .
   ruff format --check .
   mypy
   pytest -m "not slow" --cov
   pytest -m slow --run-network  # AUTOHYBTS_RUN_HEAVY=1; todos os extras dos tutoriais
   pytest -m network --run-network
   sphinx-build -W documentation documentation/_build/html
   sphinx-build -W documentation/en documentation/_build/html-en
   python -m build --outdir dist/1.0.0rc4
   python -m twine check dist/1.0.0rc4/*
   python tools/check_release.py --tag v1.0.0rc4 --dist dist/1.0.0rc4

Depois de instalar o wheel num ambiente limpo sem extras, execute
``python tools/smoke_wheel.py`` com o Python desse ambiente. O CI também valida os
pisos das dependências, Python 3.11–3.13, Windows, frameworks opcionais e a auditoria
das dependências selecionadas pelo job ``audit``. Os notebooks pesados e de rede
rodam no fluxo da tag; o CI de cada PR não executa esse trabalho longo.
Também é possível executar ``ci`` manualmente com ``release-checks=true`` antes de
criar a tag, sem publicar. Nesse modo, requisitos ausentes nos notebooks causam
falha, em vez de um teste pulado.

Referências: `workflows reutilizáveis do GitHub
<https://docs.github.com/en/actions/how-tos/reuse-automations/reuse-workflows>`_ e
`publicação com Trusted Publisher
<https://docs.pypi.org/trusted-publishers/using-a-publisher/>`_.

Publicação no Read the Docs
---------------------------

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
