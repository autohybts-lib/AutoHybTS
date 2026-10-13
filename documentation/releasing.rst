Versão 1 e publicação
=====================

1.0.0rc1 — candidata, 13 de outubro de 2026
-------------------------------------------

Esta pré-versão prepara a primeira versão estável da AutoHybTS. O escopo inclui
previsores individuais, híbridos residuais, estágios múltiplos, decomposição,
combinações e ensembles opcionais, painéis de modelos locais, busca e ajuste de
hiperparâmetros, validação temporal, relatórios e persistência. A variante A prevê
vários passos; a B prevê um passo por atualização com parâmetros congelados.

A regra padrão usa ganho mediano mínimo de 5%, vitórias em pelo menos 60% das
unidades pareadas e o critério de variabilidade. ``fold="auto"`` usa 7 origens por
camada quando o histórico comporta; séries menores recebem menos origens e um
aviso. A calibração e a confirmação em controles sintéticos mediram 2–4% de falso
aceite e 54–56% de ganho útil aceito. Essas taxas valem para as condições medidas,
não são uma garantia para séries reais ou para todas as famílias opcionais.

Mudanças de entrega nesta candidata:

* versão ``1.0.0rc1`` e classificação beta nos metadados do pacote;
* CI reutilizado no commit da tag antes de qualquer publicação;
* conferência da correspondência exata entre tag, versão e metadados do wheel/sdist;
* teste do wheel instalado sem extras, incluindo previsão A/B e persistência;
* builds Sphinx dos dois idiomas e contratos StatsForecast no CI;
* notebooks lentos, inclusive os pesados, e downloads verificados antes da publicação;
* candidatas ``rc`` destinadas ao TestPyPI e versões finais ao PyPI.

Compatibilidade
---------------

A API pública candidata à v1 é a documentada em :doc:`api/index`, incluindo os
parâmetros dos experimentos, resultados retornados, exceções, datasets e interfaces
de registro de modelos. Módulos e atributos privados, detalhes de cache, tempos de
execução e mensagens textuais de diagnóstico não são contratos de compatibilidade.

Antes de ``1.0.0``, uma correção necessária pode mudar a API entre candidatas; essa
mudança será registrada nas notas de release. A partir da versão estável:

* correções compatíveis usam versões de patch;
* adições compatíveis usam versões menores;
* remoções e alterações incompatíveis exigem versão maior; APIs destinadas à
  remoção serão documentadas e avisadas com ``DeprecationWarning`` numa versão menor
  antes da remoção na próxima versão maior.

IDs de candidatas e o esquema ``autohybts-model/1`` são contratos versionados.
Mudanças que invalidem IDs ou artefatos exigem nota de migração. O manifesto registra
as versões do ambiente, mas pickle também depende das bibliotecas e dos modelos
externos: não há garantia de carregar um modelo em versões arbitrárias dessas
dependências. Preserve o ambiente de treino e confira ``read_manifest`` antes de
carregar. Veja :doc:`user_guide/persistence`.

A migração de ``0.0.1`` não altera assinaturas ou IDs nesta candidata. A regra
calibrada já faz parte do código anterior à candidata. Para reproduzir experimentos,
fixe versões, sementes, modelos, grades, ``fold`` e limiares; uma API compatível não
implica previsões idênticas após trocar dependências.

Critérios para 1.0.0 estável
----------------------------------------

O estudo M3/M4 ainda não tem resultados. Antes da versão final, congele o protocolo
com a regra calibrada, amostras, sementes e orçamento; execute-o sem usar o teste
para reajustar a regra e relate ganhos, pioras, cobertura e custo contra baselines.
Revise também a experiência de instalação pelo TestPyPI, os contratos públicos e
as limitações encontradas durante o uso da candidata.

Famílias compostas continuam opcionais até cumprir a calibração por cadeia. Novas
famílias de modelos, variáveis exógenas, previsões probabilísticas e modelos globais
não são requisitos para este release. A evidência sintética não substitui o estudo
em séries reais. Os notebooks de demonstração não constituem esse estudo.

Publicação e instalação da candidata
------------------------------------

Integrar o PR prepara o código; não publica pacotes. Depois da revisão, a tag
``v1.0.0rc1`` dispara ``release.yml`` no mesmo commit. O workflow valida a tag,
executa o CI completo e os testes de release, gera e confere os dois artefatos e
publica no TestPyPI. Uma tag final, como ``v1.0.0``, segue para o PyPI.

Cadastre um Trusted Publisher em cada índice para o repositório
``jvataidee/AutoHybTS``, arquivo ``release.yml`` e ambiente correspondente:
``testpypi`` ou ``pypi``. Crie esses ambientes no GitHub com revisores obrigatórios.
A configuração externa precisa estar pronta antes da tag; o PR não cria contas,
publishers ou revisores.

Depois de a candidata estar publicada, teste num ambiente virtual limpo:

.. code-block:: bash

   python -m pip install numpy pandas scikit-learn
   python -m pip install --index-url https://test.pypi.org/simple/ --no-deps autohybts==1.0.0rc1

Instale as dependências pelo PyPI e só o pacote pelo TestPyPI; evite misturar os
índices com ``--extra-index-url``. O teste acima instala o núcleo. Instale
``statsmodels`` e ``matplotlib`` pelo PyPI para usar os exemplos estatísticos e os
gráficos. Cada versão enviada a um índice é imutável; se for preciso corrigir a
candidata, use ``1.0.0rc2``.

Checklist local
----------------

.. code-block:: bash

   ruff check .
   ruff format --check .
   mypy
   pytest -m "not slow" --cov
   pytest -m slow --run-network  # AUTOHYBTS_RUN_HEAVY=1; todos os extras dos tutoriais
   pytest -m network --run-network
   sphinx-build -W documentation documentation/_build/html
   sphinx-build -W documentation/en documentation/_build/html-en
   python -m build --outdir dist/1.0.0rc1
   python -m twine check dist/1.0.0rc1/*
   python tools/check_release.py --tag v1.0.0rc1 --dist dist/1.0.0rc1

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
