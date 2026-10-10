Versões e release
=================

Versionamento
-------------

AutoHybTS segue `SemVer <https://semver.org/lang/pt-BR/>`_. Enquanto a versão for
``0.x``, mudanças incompatíveis podem acontecer em versões menores e ficam no
CHANGELOG. Contam como API pública:

* tudo que ``autohybts`` e ``autohybts.datasets`` exportam em ``__all__``;
* os nomes e significados dos parâmetros de ``AutoHybridExperiment`` e dos seus métodos;
* a hierarquia de exceções;
* o formato do artefato de ``save_model`` (manifesto e checksum);
* a identidade das candidatas (``candidate_id``): mudar a serialização canônica
  invalida identidades guardadas e exige versão maior (ou menor, em ``0.x``) e nota de
  migração.

Checklist de release candidate
------------------------------

1. ``ruff check . && ruff format --check . && mypy`` sem erros.
2. ``pytest -m "not slow" --cov`` (cobertura ≥ 85 %) e ``pytest -m slow`` (notebooks);
   ``pytest --run-network -k network`` uma vez com rede.
3. Teste no piso de versões (pandas 2.1, NumPy 1.26, scikit-learn 1.4, statsmodels 0.14).
4. ``python -m build`` e ``twine check dist/*``; instalar o wheel num ambiente limpo sem
   extras e rodar um ``compare_models`` pequeno; ``import autohybts`` não pode carregar
   statsmodels, matplotlib, Optuna nem rdata.
5. Segurança de carga: ``load_model`` só com ``trusted=True`` e checksum conferido;
   downloads só HTTPS em domínios aprovados, com manifesto e verificação a cada leitura.
6. ``sphinx-build -W documentation documentation/_build/html`` sem avisos.
7. Versão em ``src/autohybts/__init__.py``, data no CHANGELOG, tag ``vX.Y.Z``.
8. README: trocar ``<organizacao>`` pelo endereço do repositório e usar URL absoluta
   para o logo (o PyPI não resolve caminhos relativos de imagem).
9. Publicar primeiro no TestPyPI; o nome ``autohybts`` estava livre no PyPI e no TestPyPI
   em 2026-10-10.

Publicação
----------

A pasta ``documentation/`` vai para o git; só ficam fora a saída do build
(``documentation/_build/``) e as cópias dos notebooks feitas na hora do build
(``documentation/tutorials/notebooks/``). O site ``.io`` é gerado a partir do repositório
no GitHub, e o pacote do pip é enviado ao PyPI: os dois dependem de o repositório estar
publicado.

1. Repositório no GitHub
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git add .
   git commit -m "AutoHybTS 0.1.0rc1"
   git remote add origin https://github.com/<organizacao>/AutoHybTS.git
   git push -u origin main

Antes do commit, ajuste no README o endereço do repositório (``<organizacao>``) e a URL
do logo para uma absoluta, por exemplo
``https://raw.githubusercontent.com/<organizacao>/AutoHybTS/main/figures/AutoHybTS.png``,
e a versão em ``src/autohybts/__init__.py``.

2. Documentação no Read the Docs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

O arquivo ``.readthedocs.yaml`` na raiz já diz ao Read the Docs como gerar o site.

1. Entre em https://readthedocs.org com a conta do GitHub.
2. Use **Import a Project** e escolha o repositório AutoHybTS.
3. O Read the Docs lê o ``.readthedocs.yaml``, instala o pacote com o extra ``docs`` e
   roda o Sphinx sobre ``documentation/``.
4. O site fica em ``https://autohybts.readthedocs.io`` e é refeito a cada ``git push``;
   as tags (``v0.1.0``) podem virar versões no seletor do site.

Com ``fail_on_warning: true``, qualquer aviso do Sphinx quebra o build. Confira antes do
push:

.. code-block:: bash

   python -m sphinx -W documentation documentation/_build/html

Alternativa: GitHub Pages (``<organizacao>.github.io/AutoHybTS``) com uma GitHub Action
que roda o mesmo comando e publica ``documentation/_build/html``.

3. Pacote no PyPI
~~~~~~~~~~~~~~~~~

**Primeira vez, manual.** Crie contas em https://test.pypi.org e https://pypi.org e gere
um *API token* em cada uma.

.. code-block:: bash

   python -m pip install build twine
   python -m build                      # dist/*.whl e dist/*.tar.gz
   python -m twine check dist/*

   # TestPyPI primeiro (usuário __token__, senha = token)
   python -m twine upload --repository testpypi dist/*
   pip install --index-url https://test.pypi.org/simple/ \
               --extra-index-url https://pypi.org/simple/ autohybts

   # tudo certo: PyPI
   python -m twine upload dist/*

Depois disso, ``pip install autohybts`` funciona para qualquer pessoa.

**Depois, automático (recomendado).** Com *Trusted Publishing* do PyPI não há token
guardado: cadastre no PyPI (*Publishing*) o repositório e o workflow
``.github/workflows/release.yml``, que gera o pacote e publica a cada tag:

.. code-block:: bash

   git tag v0.1.0rc1
   git push origin v0.1.0rc1

.. warning::

   Cada versão só pode ser enviada ao PyPI uma vez. Se algo sair errado, suba o número
   (``0.1.0rc1`` → ``0.1.0rc2``). Por isso comece pelo TestPyPI e por versões ``rc``.
