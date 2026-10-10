Segurança
=========

Esta página resume o que a AutoHybTS protege, o que fica por conta de quem a usa e como
relatar uma vulnerabilidade.

Modelos salvos (pickle)
-----------------------

``save_model`` grava o modelo com ``pickle``. **Carregar um pickle executa código**: um
artefato adulterado pode rodar qualquer comando na sua máquina. Por isso:

* ``load_model`` só desserializa com ``trusted=True``. Para inspecionar um artefato sem
  executar nada, use ``read_manifest(path)``.
* O checksum do manifesto fica na mesma pasta do modelo: ele detecta arquivo corrompido,
  **não** adulteração — quem edita ``model.pkl`` também edita o manifesto.
* Para resistir a adulteração, use uma das verificações abaixo. As duas acontecem antes de
  qualquer desserialização.

.. code-block:: python

   import secrets
   from autohybts import load_model, read_manifest, save_model

   # 1) Assinatura HMAC: a chave fica fora do artefato (cofre, variável de ambiente).
   key = secrets.token_bytes(32)
   path = save_model(final, "artifacts/champion", signing_key=key)
   model = load_model(path, trusted=True, signing_key=key)

   # 2) Hash fixado: quem recebe o artefato confere o SHA-256 publicado por um canal
   #    confiável (notas da versão, mensagem assinada).
   digest = read_manifest(path)["payload"]["sha256"]
   model = load_model(path, trusted=True, sha256=digest)

Outras regras do carregamento: o manifesto só pode apontar para ``model.pkl`` na própria
pasta, links simbólicos são recusados e a especificação desserializada tem de bater com a
do manifesto.

.. warning::

   Nunca use ``trusted=True`` em artefatos baixados da internet, recebidos por e-mail ou
   gerados por terceiros sem uma das verificações acima. Um modelo não substitui um
   formato de troca: para compartilhar resultados, prefira ``export_evidence``.

Downloads de datasets
---------------------

* Nada é baixado na instalação nem no ``import``; só chamadas explícitas acessam a rede.
* Só HTTPS, só domínios aprovados por provedor. Redirecionamentos para fora do HTTPS são
  recusados e a URL final precisa estar nos mesmos domínios.
* Tempo limite, número de tentativas e tamanho máximo por arquivo.
* Cada arquivo tem manifesto com SHA-256, conferido a cada leitura; os arquivos do M4
  diário e o pacote ``Mcomp`` têm o hash fixado no código. Os demais são fixados no
  primeiro download (*trust on first use*): baixe-os numa rede confiável.
* Nomes de provedor, pacote e arquivo são um único componente de caminho (letras, dígitos
  e ``._+-``): nenhum nome vindo de um índice remoto escreve fora da pasta do cache.
* Arquivos compactados são extraídos membro a membro, recusando caminhos absolutos,
  ``..``, links e membros acima de um tamanho máximo.
* Os CSVs que a biblioteca deriva do M3 têm seu próprio registro de hashes; uma cópia
  alterada é refeita a partir do arquivo original verificado, nunca lida.

A pasta do cache (``AUTOHYBTS_DATA_DIR`` ou ``~/.cache/autohybts/datasets``) deve ser
gravável só pelo seu usuário.

Evidências e artefatos
----------------------

* ``export_evidence`` e os gráficos nunca usam o teste selado nem gravam valores brutos da
  série.
* Os manifestos registram versões de pacotes, a plataforma (sistema operacional) e o
  commit do repositório git do diretório atual. O commit é lido direto da pasta ``.git``:
  a biblioteca não executa o programa ``git`` (no Windows, um ``git.exe`` plantado no
  diretório atual seria executado no lugar do verdadeiro). Revise o manifesto antes de
  publicar um artefato se a plataforma ou o commit forem sensíveis.
* Tabelas CSV podem conter identificadores vindos dos seus dados (``series_id``); ao abrir
  CSVs de terceiros numa planilha, desconfie de células que começam com ``=``, ``+``,
  ``-`` ou ``@``.

Notebooks
---------

Os tutoriais são guardados com as saídas e publicados na documentação. Um teste de
contrato recusa saídas com caminhos locais (pasta do usuário, unidade do Windows,
``site-packages``). Antes de publicar um notebook seu, limpe as saídas que mostrem
caminhos, nomes de usuário ou dados privados.

Dependências e integração contínua
----------------------------------

* O núcleo depende só de NumPy, pandas e scikit-learn; bibliotecas pesadas ficam em extras
  opcionais e são importadas só quando usadas.
* O CI roda com token só de leitura, sem credenciais persistidas no checkout, e audita as
  dependências resolvidas com ``pip-audit``. O Dependabot propõe atualizações semanais das
  Actions e dos pacotes.
* A publicação no PyPI usa *Trusted Publishing* (sem token guardado), num job separado,
  único com permissão ``id-token: write`` e protegido pelo ambiente ``pypi``.

Relatar uma vulnerabilidade
---------------------------

Não abra uma *issue* pública. Use o *private vulnerability reporting* do GitHub
(aba **Security → Report a vulnerability** do repositório) com a versão, os passos para
reproduzir e o impacto esperado. As correções saem numa versão nova e no CHANGELOG.
