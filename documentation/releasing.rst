Versão 1 e publicação
=====================

1.0.0rc4 — experimentos em SQLite, MLOps e variáveis exógenas, 14 de outubro de 2026
------------------------------------------------------------------------------------

Adições compatíveis: sem ``X`` e sem banco, nada muda nos IDs das candidatas, no hash do
protocolo padrão nem no esquema de persistência dos modelos.

* **Registro em SQLite:** ``ExperimentStore`` grava experimentos (candidatas, métricas por
  camada e por origem, decisões, campeão, avaliação externa) e responde a consultas SQL
  somente leitura; valores da série só com ``values=True``.
* **MLOps no mesmo banco:** registro de modelos com SHA-256 do artefato, promoção com
  histórico de estágios, ``load`` com ``trusted=True``, previsões, observações e
  covariáveis registradas, ``monitor`` (MAE móvel contra a referência, com alerta);
  série sintética ``commodity_price_daily``.
* **Variáveis exógenas:** ``fit(y, X)`` com colunas ``past`` (defasadas pelo horizonte) e
  ``future`` (declaradas), ARIMAX, SARIMAX, modelos sobre lags e híbridos com covariáveis,
  família ``exogenous`` decidida pela regra de aceite, previsão com ``X_future``,
  streaming, persistência e painéis de séries.
* **Seleção automática de exógenas:** ``exog_selection`` (``ccf``, ``granger``, ``lasso``,
  ``stepwise_bic``, ``mutual_info``, ``vote``, ``auto``), ``exp.exog_report()`` e
  ``exp.compare_exog()`` (Wilcoxon e Diebold–Mariano).
* **Conjuntos com exógenas:** ``list_exog_datasets()`` e ``load_exog_dataset()``
  (seatbelts, bjsales, uschange, insurance, elecdaily, vic_elec_daily).
* **Documentação:** guias de registro em SQLite, MLOps e exógenas; notebooks 18 (análise
  com SQL), 19 (fluxo de MLOps), 20 (exógenas em benchmarks) e 21 (MLOps com exógenas na
  demanda elétrica real); coautoria com IA (Claude e Codex) na página *Sobre*.

.. code-block:: bash

   python -m pip install "autohybts[stats,plot]==1.0.0rc4"

1.0.0rc3 — diagnóstico da série e pré-processamento, 14 de outubro de 2026
--------------------------------------------------------------------------

Adições compatíveis: nada muda nos IDs das candidatas, no hash do protocolo padrão nem no
esquema de persistência, e todo recurso novo só atua quando pedido.

* **Sazonalidade:** ``exp.seasonality_report()`` (força sazonal e teste de Friedman com os
  ciclos como blocos) e ``seasonal_period="auto"``, que testa o ciclo no treino da
  primeira origem antes de passar o período aos modelos.
* **SARIMA:** modelo ``sarima`` (extra ``stats``), fora da busca padrão.
* **Diferenças automáticas:** ``differencing="auto"`` nos modelos sobre defasagens escolhe
  a diferença sazonal e até duas diferenças no treino de cada origem, por KPSS ou pela
  regra ADF do projeto de replicação.
* **Caracterização:** ``exp.characterize()`` com as duas regras do projeto de replicação
  e os gráficos ``stationarity``, ``acf_pacf``, ``decomposition`` e
  ``plot_characterization``.
* **Relatórios:** ``exp.stationarity_report()`` (ADF, KPSS e consenso),
  ``exp.decompose()``, ``exp.residual_diagnostics()`` (Ljung-Box nos resíduos de uma
  base), ``exp.outlier_report()`` (candidatos AO/LS/TC, só exploratório), métricas
  ``MAPE`` e ``MdAPE`` e ``cmp.kruskal()`` (Kruskal-Wallis e Dunn).
* **Pré-processamento opcional:** ``transform="boxcox"``/``"log"`` e
  ``outliers="winsorize"``/``"interpolate"`` nos modelos sobre defasagens, ajustados no
  treino de cada origem; ``missing="ffill"`` preenche lacunas pelo último valor observado.
  Tudo desligado por padrão.
* **Documentação:** variantes A e B, conceitos, leitura de resultados, escolha da
  configuração, solução de problemas, manutenção, referência da API completa, figuras em
  todos os tutoriais e os notebooks 16 (diagnóstico) e 17 (17 séries clássicas em três
  classes, com cinco modelos nos protocolos A e B, 3, 6 e 12 lags e horizontes até 12).

.. code-block:: bash

   python -m pip install "autohybts[stats,plot]==1.0.0rc3"

1.0.0rc2 — documentação do PyPI, 13 de outubro de 2026
------------------------------------------------------------

Esta candidata incorpora ao pacote o README atualizado após a primeira publicação
no PyPI. A instalação principal usa o PyPI, os comandos fixam ``1.0.0rc2`` e a
documentação nos dois idiomas inclui extras, Jupyter e o fluxo de publicação.
Os algoritmos, assinaturas públicas, IDs de candidatas e esquema de persistência
mantêm o comportamento da ``1.0.0rc1``. A versão continua sendo uma candidata;
os critérios para ``1.0.0`` estável permanecem descritos abaixo.

.. code-block:: bash

   python -m pip install "autohybts[stats,plot]==1.0.0rc2"

Versão: `1.0.0rc2 no PyPI <https://pypi.org/project/autohybts/1.0.0rc2/>`_.

1.0.0rc1 — candidata, 13 de outubro de 2026
-------------------------------------------

Esta pré-versão prepara a primeira versão estável da AutoHybTS. O escopo inclui
previsores individuais, híbridos residuais, estágios múltiplos, decomposição,
combinações e ensembles opcionais, painéis de modelos locais, busca e ajuste de
hiperparâmetros, validação temporal, relatórios e persistência. A variante A prevê
vários passos; a B prevê um passo por atualização com parâmetros congelados.

A candidata foi `publicada no PyPI em 13 de outubro de 2026
<https://pypi.org/project/autohybts/1.0.0rc1/>`_, com wheel e pacote-fonte. A
publicação pelo ``publish.yml`` foi aprovada no ambiente ``pypi`` do GitHub.

A regra padrão usa ganho mediano mínimo de 5%, vitórias em pelo menos 60% das
unidades pareadas e o critério de variabilidade. ``fold="auto"`` usa 7 origens por
camada quando o histórico comporta; séries menores recebem menos origens e um
aviso. A calibração e a confirmação em controles sintéticos mediram 2–4% de falso
aceite e 54–56% de ganho útil aceito. Essas taxas valem para as condições medidas,
não são uma garantia para séries reais ou para todas as famílias opcionais.

Mudanças de entrega nesta candidata:

* versão ``1.0.0rc1`` e classificação beta nos metadados do pacote;
* workflow de validação com CI reutilizável no commit da tag;
* conferência da correspondência exata entre tag, versão e metadados do wheel/sdist;
* teste do wheel instalado sem extras, incluindo previsão A/B e persistência;
* builds Sphinx dos dois idiomas e contratos StatsForecast no CI;
* verificações completas de notebooks lentos/pesados e downloads no fluxo de validação;
* candidata disponível no PyPI; TestPyPI reservado ao fluxo separado de testes.

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
Revise também a experiência de instalação pelo PyPI, os contratos públicos e
as limitações encontradas durante o uso da candidata.

Famílias compostas continuam opcionais até cumprir a calibração por cadeia. Novas
famílias de modelos, variáveis exógenas, previsões probabilísticas e modelos globais
não são requisitos para este release. A evidência sintética não substitui o estudo
em séries reais. Os notebooks de demonstração não constituem esse estudo.

O fluxo de publicação e o checklist antes da tag estão em :doc:`maintainers`.
