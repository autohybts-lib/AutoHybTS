# Plano de ampliação do AutoHybTS: modelos, ensembles, boosting e validação temporal

> **Estado:** proposta técnica e agenda de pesquisa; não descreve funcionalidades já implementadas.
>
> **Referência:** 12 de outubro de 2026 · **Biblioteca:** AutoHybTS (pré-alfa 0.0.1) · **Fontes de código:** [AutoHybTS](https://github.com/jvataidee/AutoHybTS) e [projeto_st_replicacao](https://github.com/jvataidee/projeto_st_replicacao).

## 1. Objetivo e delimitação

Ampliar o AutoHybTS de um mecanismo de avaliação de previsores individuais e de híbridos residuais para uma plataforma de **busca e comparação de sistemas compostos de previsão**. O mesmo experimento deve avaliar:

1. Modelos *standalone*: estatísticos, lineares, baseados em árvores, aprendizado de máquina e redes neurais.
2. Híbridos: correção residual aditiva, múltiplos estágios, decomposição e múltiplos corretores.
3. Ensembles de previsão: médias simples/robustas, combinações ponderadas, bagging, stacking, seleção gulosa e dinâmica.
4. Boosting: (i) modelos de gradient boosting sobre lags, (ii) AdaBoost.R2 e (iii) **boosting sequencial de previsores completos**, este último como hipótese metodológica a ser desenvolvida e validada.
5. Composições recursivas: *ensemble de bases + corretor*, *base + ensemble de corretores* e *ensemble de híbridos*.

**Pergunta orientadora:** em quais categorias e horizontes de séries temporais a combinação, a hibridização residual e o boosting apresentam melhor generalização **por unidade de complexidade e custo**, em relação à melhor base individual e à média simples?

**Regra de produto:** a busca sempre deve poder manter o modelo individual se os modelos compostos não superarem os critérios de seleção. A biblioteca não deve se tornar um gerador indiscriminado de combinações.

### 1.1. Escopo incremental

A primeira versão concentra-se em previsão pontual univariada e painéis locais, com os protocolos existentes:

- **A:** previsão aberta de `h` passos, sem observações futuras intermediárias.
- **B:** avaliação rolling de um passo, com parâmetros congelados e observações reveladas apenas depois da previsão.

Modelos globais sobre painéis, covariáveis exógenas, séries hierárquicas e previsões probabilísticas são extensões posteriores: exigem contratos próprios e não devem ser apresentados como prontos.

### 1.2. Princípios metodológicos

- Preservar `V_search`, `V_select`, avaliação externa selada, testes de causalidade, registro de candidatos rejeitados e identidade estável de tentativas.
- Garantir que features, resíduos, pesos e meta-modelos sejam obtidos só com informações disponíveis na origem temporal.
- Comparar candidatos nas mesmas séries, origens, horizontes, escalas e métricas.
- Avaliar **custo total de desenvolvimento** (busca, OOF, ajuste de pesos, refits e inferência), não apenas o tempo do modelo final.
- Não interpretar a regra de ganho mínimo e vitórias como garantia estatística de ausência de sobreajuste; calibrar seu falso aceite.
- Documentar quando a previsão de um membro é direta, recursiva, probabilística, local ou global.

## 2. Diagnóstico dos dois repositórios

### 2.1. AutoHybTS atual: reutilizar em vez de reimplementar

| Área | Implementação existente | Lacuna da ampliação |
|---|---|---|
| Modelos individuais | Naive, SeasonalNaive, ARIMA via statsmodels, linear, KNN, SVR, árvore, RandomForest, MLP | StatsForecast, MLForecast, gradient boosting, mais redes |
| Corretores | Linear, KNN, MLP por janelas de resíduos | SVR, árvores, boosting e corretores de sequência residual |
| Híbridos | Residual aditivo, multistage, decomposição | Modelos compostos como bases e corretores; novos objetivos de boosting |
| Combinações | Pesos iguais e inverso do MSE; combinação de corretores | Mediana, pesos restritos, stacking, ensembles de híbridos |
| Dinâmico | DSEL de corretores, somente B | DSEL entre previsores completos, pesos por horizonte e experts |
| Execução | Registry, `CandidateSpec`, S0–S4, ledger, seleção incremental, avaliação temporal, CV | Representação de composições aninhadas e busca por famílias de ensemble |
| Painel | Série local por ID, agregação de postos | Treino *global* e transferência entre séries requerem novo contrato |

Arquivos relevantes: [registry](https://github.com/jvataidee/AutoHybTS/blob/main/src/autohybts/registry/registry.py), [candidate specs](https://github.com/jvataidee/AutoHybTS/blob/main/src/autohybts/domain/specs.py), [combination](https://github.com/jvataidee/AutoHybTS/blob/main/src/autohybts/strategies/combination.py), [search](https://github.com/jvataidee/AutoHybTS/blob/main/src/autohybts/search/spaces.py) e [validação temporal](https://github.com/jvataidee/AutoHybTS/blob/main/documentation/user_guide/temporal_validation.rst).

### 2.2. Conhecimento aproveitável de projeto_st_replicacao

O repositório experimental já implementa AutoARIMA, AutoSARIMA e AutoETS com **Nixtla StatsForecast**; LightGBM; GARCH; Prophet/NeuralProphet; LSTM; e VanillaTransformer pelo **Nixtla NeuralForecast**, além de quatro híbridos com KNN e três híbridos estatístico-ANN. Existem classes legadas de composição e seleção dinâmica.

| Tratamento da replicação | Composição nativa-alvo no AutoHybTS |
|---|---|
| `linear_knn_residual` | Linear + KNN sobre resíduos |
| `tree_knn_residual` | DecisionTree + KNN sobre resíduos |
| `knn_knn_residual` | KNN + KNN sobre resíduos |
| `svm_knn_residual` | SVR + KNN sobre resíduos |
| `ets_ann_residual` | AutoETS + MLP residual |
| `arima_ann_residual` | AutoARIMA + MLP residual |
| `sarima_ann_residual` | AutoSARIMA + MLP residual |

Na configuração atual, apenas os três híbridos estatístico-ANN estão ativos; os quatro KNN e as composições auto-híbridas estão implementados mas desativados. Ao portar, distinguir **capacidade disponível** de **resultado experimental atual**.

**Diretriz:** extrair contratos, parâmetros, transformações, geração OOF e testes de paridade, não importar o orquestrador, banco experimental nem modelos históricos como dependência interna. Preservar os dois repositórios independentes para comparação metodológica.

Ver: [models.yaml](https://github.com/jvataidee/projeto_st_replicacao/blob/main/configs/models.yaml), [experiment.yaml](https://github.com/jvataidee/projeto_st_replicacao/blob/main/configs/experiment.yaml), [statistical.py](https://github.com/jvataidee/projeto_st_replicacao/blob/main/src/models/statistical.py), [statistical_residual.py](https://github.com/jvataidee/projeto_st_replicacao/blob/main/src/models/statistical_residual.py).

## 3. Taxonomia de ensembles e respaldo bibliográfico

Cada estratégia deve ter um identificador, descrição algorítmica, requisitos, controles e comparações próprias. Um ensemble é uma composição de **previsões**; isso não torna *todas* as estratégias equivalentes.

| Código | Família / estratégia | Ideia e configuração mínima | Referências | Fase |
|---|---|---|---|---|
| E01 | Equal-weight average | Média de previsões alinhadas; pesos `1/K` | [R01], [R13] | F2 |
| E02 | Median / trimmed mean | Agregação robusta; fração de corte explícita | [R13] | F2 |
| E03 | Inverse-loss weighted | Pesos estáticos de erro OOF (já parcialmente existente) | [R01] | F2 |
| E04 | NNLS / simplex / shrinkage | Pesos não negativos, somando 1 quando definido; regularização para a média | [R05], [R14] | F2 |
| E05 | Ensemble selection | Inserção gulosa de especialistas; limite de membros e repetições | [R06] | F3 |
| E06 | Temporal bagging | Bootstrap por blocos de dependência (não amostras i.i.d.) | [R02], [R03] | F3 |
| E07 | Bagged ETS / decomposição | Box–Cox + STL + bootstrap de blocos no restante + remontagem | [R03] | F3 |
| E08 | Subspace / multi-seed | Subconjuntos de lags, features e modelos estocásticos | [R02] | F3 |
| E09 | GBDT como previsor | LightGBM, XGBoost, CatBoost, HistGradientBoosting em lags | [R08], [R09], [R10] | F1/F3 |
| E10 | AdaBoost.R2 | Reponderação de linhas de treino de regressão sobre atributos causais | [R11] | F3 |
| E11 | Forecast boosting | Pseudo-resíduos OOF de sistemas completos, shrinkage, parada temporal | Fundamento [R07]; **proposta experimental** | F4 |
| E12 | Stacking | Meta-regressor treinado somente com previsão OOF de bases/híbridos | [R04], [R05] | F4 |
| E13 | Stacking/horizon-weighted | Pesos ou metamodelos por faixas de horizonte | [R04], [R05] | F4 |
| E14 | Feature-based weighting / FFORMA | Meta-ponderação aprendida entre séries usando características | [R12] | F5 |
| E15 | Mixture of experts | Gating condicionado ao contexto disponível na origem | [R15] | F5 |
| E16 | Dynamic ensemble selection | DSEL de bases/híbridos, extrapolando a estratégia atual de corretores | Código AutoHybTS; comparar [R06] | F5 |
| E17 | Exponential weights / Hedge | Atualizar pesos após erros observados, principalmente B | [R16] | F5 |
| E18 | Ensemble of hybrids | Agregação de híbridos independentes e diversificados | [R13], [R17] | F2/F4 |
| E19 | Ensemble-base + residual | Construir a base combinada e prever **seus** resíduos OOF | Estruturas AutoHybTS | F2 |
| E20 | Composições probabilísticas/hierárquicas | Combinações de distribuições e coerência hierárquica | Extensão futura; fora do escopo inicial | F6+ |

As médias simples/medianas e o melhor modelo individual são **controles obrigatórios**, não somente mais dois candidatos entre muitos. Métodos avançados precisam demonstrar ganho líquido sobre esses controles.

### 3.1. Distinção rigorosa entre boosting e hibridização residual

A palavra *boosting* abriga três objetos **diferentes** no planejamento:

**B1. Gradient boosting como modelo individual.** LightGBM/XGBoost/CatBoost e similares treinam sequencialmente árvores para otimizar uma função de perda sobre atributos temporais causais. Podem prever a série diretamente **ou** atuar como regressor corretor sobre janelas residuais. O boosting pertence ao *backend* do estimador.

**B2. AdaBoost.R2.** Algoritmo de regressão com reponderação de amostras difíceis; treinar sobre lags, respeitando dependência temporal na avaliação. Não é simplesmente somar correções ARIMA + ETS.

**B3. Boosting de previsores temporais completos.** Estratégia nova que gera pseudo-resíduos de previsões temporais OOF, ajusta novos estágios com `learning_rate` e decide quando interromper. **Não** é equivalente automaticamente ao `residual_multistage` atual; exige definição formal, testes adicionais e ablação comparativa.

Para B3, com `F_0` uma previsão-base e `L` uma função de perda:

$$
g_m \approx -\left.\frac{\partial L(y,F)}{\partial F}\right|_{F=F_{m-1}},\qquad
F_m(t,h)=F_{m-1}(t,h)+\nu\,g_m(t,h).
$$

`g_m` recebe apenas features e pseudo-resíduos que seriam conhecidos na origem de treino correspondente; para perda quadrática, o pseudo-resíduo é proporcional a `y-F_{m-1}`. Para MAE, Huber ou perdas de quantis, explicitar convenções de derivadas/subgradientes. `0<\nu\le1` controla a contribuição do estágio.

**Critérios mínimos de B3:** gerar os alvos de cada estágio OOF com a configuração do estágio anterior; manter histórico distinto para treinar e validar o estágio; usar parada temporal e limite explícito de estágios; comparar com residual multistage de profundidade e orçamento equivalentes; conduzir teste com ruído branco para medir falsos ganhos.

### 3.2. Famílias de combinação

Para `K` componentes na mesma série, origem e horizonte:

$$
\widehat{y}_{t+h|t}^{E}=\sum_{k=1}^K w_{k,t,h}\,\widehat{y}_{t+h|t}^{(k)}.
$$

- **Pesos iguais**: sem parâmetros aprendidos.
- **Pesos de erro**: calcular perdas OOF em treinamento cronologicamente válido; nunca na janela de seleção do próprio ensemble.
- **Pesos otimizados**: minimização com restrições (não negatividade, simplex) e regularização/shrinkage. Estimar de forma robusta para não criar instabilidade com poucas origens [R14].
- **Stacking**: `\widehat y=g(\widehat y_1,...,\widehat y_K,z_t)` com `g` treinado em previsão OOF anterior à seleção. Começar por NNLS e Ridge, só depois árvore/MLP como metamodelos.
- **Horizonte**: `w_{k,h}` por `h` ou **faixa de horizontes**, com histórico mínimo. A versão A precisa de previsões OOF multipasso para treinar o meta-modelo.
- **Dinâmico**: pesos em função de erros **passados** ou características observadas na origem; em A, sem revelar `y` nos passos intermediários.
- **Mixture of experts**: treinar gating em dados causalmente disponíveis; para painéis globais, evitar usar séries do conjunto de avaliação.
- **Ensemble de híbridos**: combinar `\widehat y_j^{H}=\widehat b_j+\alpha_j\widehat e_j`, estimando novos pesos somente a partir da previsão OOF de cada híbrido.
- **Ensemble-base corrigido**: `\widehat y=\sum_kw_k\widehat b_k+\alpha\widehat e^{E}`, em que `e^{E}` é o erro OOF **do ensemble**, não a soma informal de resíduos in-sample.

## 4. Integração com Nixtla e modelos complementares

### 4.1. Backends opcionais

| Integração | Primeiros modelos | Uso | Pontos de contrato |
|---|---|---|---|
| **StatsForecast** | AutoARIMA, AutoSARIMA (AutoARIMA sazonal), AutoETS, AutoTheta, AutoCES | Bases e depois corretores de sequência | Horizonte variável, atualização de estado com parâmetros fixos, período sazonal |
| **MLForecast** | Pipeline de lags + Linear/Ridge/LightGBM/XGBoost/CatBoost/RandomForest | Backend de regressores locais e globais | Não confundir backend com algoritmo; features futuras e treino de painéis |
| **NeuralForecast** | NHITS, NBEATS/NBEATSx, PatchTST, depois LSTM/TCN/TFT | Bases e posteriormente corretores diretos | `h` de treinamento, dispositivo, histórico, validade das janelas e custo |
| **scikit-learn** | SVR, árvore, RandomForest, HistGradientBoosting, AdaBoost.R2 | Bases e corretores sobre lags | Encapsular escalonamento/diferenciação no treino |
| **LightGBM/XGBoost/CatBoost** | Regressão em lags | Bases e corretores | Extras opcionais e parâmetros específicos |
| **Prophet/NeuralProphet** | Tendência/sazonalidade | Inicialmente bases | Calendário real e capacidade de uso em B |
| **GARCH** | Média condicional e volatilidade | Especializado | Não comparar diretamente forecast de volatilidade com previsão pontual de nível |

Criar *extras* opcionais (nomes sugeridos, **não publicados**): `nixtla-stats`, `nixtla-ml`, `nixtla-neural`, `boosting`. Manter `import autohybts` independente desses backends e registrar conflitos de versões de Pandas/NumPy/PyTorch em matriz de CI.

**Importante:** o `arima` atual do AutoHybTS usa `statsmodels`; um novo `sf_autoarima` deve ter identificador diferente, para não reescrever comportamento de um modelo existente e invalidar suas provas de paridade.

### 4.2. Papéis possíveis dos modelos

| Modelo | Base | Corretor de janelas residuais | Corretor de sequência residual |
|---|---|---|---|
| Linear/KNN/MLP | Já existe | Já existe | Via novo adaptador, não automaticamente |
| SVR/árvore/RandomForest | Já existe | Adicionar | Via novo adaptador |
| LightGBM/XGBoost/CatBoost | Adicionar | Adicionar | Possível com previsão recursiva |
| AutoARIMA/AutoETS/AutoTheta | Adicionar | Não diretamente | Adicionar adaptador |
| NHITS/NBEATSx/PatchTST | Adicionar | Não diretamente | Pesquisa posterior |
| LSTM/Transformer/Mamba | Via wrappers externos | Via wrappers externos | Pesquisar versão direta multipasso |

São contratos distintos:
1. **ResidualWindowRegressor** aprende função de lags residuais e produz a próxima correção (com recursão explícita em A).
2. **ResidualSequenceForecaster** aprende uma série temporal de erros e produz todo o vetor residual de horizonte `h` (direto ou recursivo).

Não registrar automaticamente qualquer `BaseForecaster` como corretor: a responsabilidade sobre a origem dos resíduos, a disponibilidade do erro, o horizonte e a transformação de volta à escala original é da composição.

### 4.3. Capacidade A/B e painéis globais

Expandir `Capabilities` com sinalizações como `supported_horizons`, `trained_horizon`, `global_panel_fit`, `requires_future_covariates`, `probabilistic` e semântica de `online_state_update`. Nem todo modelo que prevê A pode prever B com parâmetros congelados, e vice-versa.

Hoje `PanelForecaster` ajusta um modelo por série. Treino global via Nixtla em várias séries requer contrato novo de `fit_panel` e `predict_panel`, com checagem de alvos de outras séries e separação entre séries vistas/não vistas. A previsão probabilística/hierárquica não deve ser forçada dentro de `predict` pontual.

Documentações: [StatsForecast](https://nixtlaverse.nixtla.io/statsforecast/), [MLForecast](https://nixtla.github.io/mlforecast/), [NeuralForecast](https://nixtlaverse.nixtla.io/neuralforecast/).

## 5. Arquitetura proposta

### 5.1. Representação recursiva de candidatos

O `CandidateSpec` atual representa estruturas limitadas (base, corretor, `members`), mas não grafo arbitrário `ensemble(hybrid(A,B),hybrid(C,D))` ou `hybrid(ensemble(A,B),C)`.

Adicionar `CandidateGraphSpec`, composto de nós imutáveis:

- `model`: nome, parâmetros, capacidades e versão do backend.
- `strategy`: residual aditivo, multistage, decomposição ou forecast boosting.
- `ensemble`: operação de combinação e submodelos (com pesos/metamodelo definidos como objetos e não componentes soltos).

Invariantes:
- grafo acíclico, profundidade e cardinalidade máximas configuráveis;
- ordem dos membros canônica em operações comutativas, ordem preservada em estágios;
- detecção de ciclos, duplicatas e requisitos incompatíveis **antes** de ajustar;
- `candidate_id` estável, dependente de subárvores/versões/protocolo e não de estado ajustado;
- salvar/carregar de modo a preservar topologia, pesos, seeds e hashes;
- manter desserialização e identidade dos `CandidateSpec` legados; não quebrar APIs vigentes.

Exemplo de composição:

```text
ensemble(method=stacking, meta=ridge)
├── hybrid(base=sf_autoarima, corrector=mlp)
├── hybrid(base=sf_autoets, corrector=knn)
└── hybrid(
      base=ensemble(method=equal, members=[lightgbm, linear]),
      corrector=svr
    )
```

### 5.2. Módulos sugeridos

```text
src/autohybts/
  integrations/
    nixtla/
      statsforecast.py
      mlforecast.py
      neuralforecast.py
  domain/
    candidate_graph.py
  residuals/
    sequence_corrector.py
  strategies/
    ensemble/
      fixed.py
      optimized_weights.py
      greedy.py
      bagging.py
      stacking.py
      dynamic.py
    boosting/
      lag_regressor.py
      forecast_boosting.py
  search/
    ensemble_spaces.py
  evaluation/
    oof_store.py
    ensemble_diagnostics.py
```

### 5.3. Serviço OOF com proveniência

Implementar armazenamento versionado para previsões fora da amostra, reutilizável por residual, stacking, pesos e boosting, evitando refit de membros equivalentes:

```text
series_id, origin, target_timestamp, horizon, variant,
candidate_id, component_id, fold_id, seed,
train_cutoff, available_at, y_pred, y_true,
data_fingerprint, split_id, protocol_fingerprint
```

- `target_timestamp` não significa que o valor já estava disponível; `available_at` registra quando o respectivo real se tornou conhecido.
- `y_true` só pode entrar em treino de peso/corretor quando `available_at <= cutoff` de quem o usa.
- Não usar a mesma cauda OOF para aprender pesos e reportar desempenho imparcial; separar subcamadas dentro do histórico ou usar procedimentos aninhados.
- Cada previsor de uma composição deve compartilhar `origin`, `horizon`, `series_id`, transformação e escala.
- Chaves de cache incluem configuração OOF, parâmetros de backend e versão de código; nunca reaproveitar resíduos incompatíveis.
- Registrar composição e falhas de membro; se houver `fallback`, sua regra precisa ser pré-definida e não ser selecionada pelo resultado do teste.

### 5.4. API-alvo — ilustração, **não implementada**

```python
from autohybts import AutoHybridExperiment

exp = AutoHybridExperiment(
    fh=30, fold=8, session_id=42
).fit(y)

base = exp.create_ensemble(
    ["sf_autoarima", "sf_autoets", "lightgbm"],
    method="nnls",
)

hybrid = exp.create_hybrid(base, "mlp")
others = exp.create_hybrid("sf_autoets", "knn")

combined = exp.create_ensemble(
    [hybrid, others],
    method="stacking",
    meta_model="ridge",
)

result = exp.compare_models(
    include=("standalone", "residual", "bagging",
             "boosting", "stacking", "ensemble_of_hybrids"),
    budget="medium",
)
final = exp.finalize_model(result.best)
```

Os métodos e strings acima são **proposta de ergonomia**. Definir os objetos de especificação e contratos antes de estabilizar a API pública.

## 6. Busca automática e prevenção do sobreajuste

Não ampliar simplesmente o produto cartesiano de bases, corretores, pesos, janelas, topologias e meta-modelos. Preservar a sequência por etapas e limitar a seleção por orçamento explícito.

| Etapa | Estratégias | Estratégia de contenção |
|---|---|---|
| S0 | Bases e baselines fixos | Avaliar com o mesmo protocolo e histórico |
| S1 | Retenção e diversidade de candidatos | Melhor por família + referência simples + diversidade de erros |
| S2 | Híbridos atuais, equal/median | Combinações pequenas, até 2–3 membros inicialmente |
| S3 | Pesos restritos e seleção gulosa | Espaço compacto, regularização, OOF separado |
| S4 | Stacking, bagging e boosting | Subconjunto selecionado; limites de ajustes/estágios |
| S5 | Contextual/global/mixture of experts | Pesquisa posterior, em painel próprio |

Incorporar orçamentos em:
- máximo de candidatos e de membros;
- profundidade máxima do grafo;
- número máximo de ajustes (incluindo folds dos membros);
- origens OOF e *inner folds*;
- limite de etapas de boosting;
- GPU/memória e duração de busca.

Os perfis atuais `smoke`, `small`, `medium` e `large` devem continuar funcionando sem mudar semântica por padrão; opções novas somente entram com `include` explícito inicialmente. Um perfil experimental `research` pode ser definido após avaliar custo real.

A cadeia de seleção incremental deve continuar começando no melhor standalone de `V_select` e permitir rejeitar ensembles mais complexos. Os limiares vigentes (1% de ganho, 60% de vitórias e 0,05 de variabilidade de correção, quando aplicável) são **critérios operacionais**, não inferência confirmatória. A documentação atual mostra falsos aceites em ruído branco; a ampliação demanda calibração, CV aninhada, diagnóstico de dependência entre origens e controle de multiplicidade nas análises.

## 7. Protocolo experimental e comparadores

### 7.1. Benchmarks

1. **Sintéticos/controles negativos**: ruído branco, processos AR corretamente especificados, tendência, sazonalidade forte e fraca, mudança de regime, heterocedasticidade, ruído residual previsível. Exigir que a seleção não aceite sistematicamente um ensemble quando o processo é puramente ruído.
2. **M3 e séries clássicas**: avaliação de heterogeneidade, histórico curto e inviabilidade de estratégias.
3. **M4 diário da replicação**: 30 séries selecionadas; treino/validação/teste do protocolo experimental do projeto; `h=365` em A, 365 atualizações de um passo em B; métricas RRMSE, RMAE, sMAPE e MASE. Não confundir esse protocolo com horizonte **oficial da competição M4**.
4. **Painéis variados**: para comparações de aprendizado global, somente após existir `fit_panel` apropriado.
5. **Novo benchmark confirmatório externo**: obrigatório para conclusões independentes sobre algoritmos cujo desenvolvimento já utilizou inspeção histórica do teste M4.

### 7.2. Comparações que devem ser sempre calculadas

| Bloco | Comparação | Pergunta |
|---|---|---|
| C1 | Melhor standalone vs híbrido correspondente | Correção residual é útil? |
| C2 | Melhor standalone vs média/mediana | Uma combinação simples basta? |
| C3 | Equal vs inverse loss vs NNLS/stacking | Pesos aprendidos compensam complexidade? |
| C4 | Bagging ETS vs ETS | Reamostragem de componentes residuais agrega robustez? |
| C5 | Bagging temporal vs random subspace vs multi-seed | Qual fonte de diversidade ajuda mais? |
| C6 | RandomForest vs AdaBoost.R2 vs GradientBoosting/LightGBM | Qual algoritmo de boosting em lags é melhor por custo? |
| C7 | LightGBM como base vs LightGBM residual | Deve prever o nível ou o erro? |
| C8 | Multistage atual vs forecast boosting | Pseudo-gradiente + shrinkage melhoram fora da amostra? |
| C9 | Stacking Ridge vs pesos restritos vs média | Meta-modelo traz ganho confiável? |
| C10 | Melhor híbrido vs ensemble de híbridos | Há complementaridade entre correções? |
| C11 | Ensemble-base vs ensemble-base corrigido | Ainda há erro estruturado residual? |
| C12 | Peso global vs específico por horizonte | A especialização em `h` é útil? |
| C13 | Estático vs DSEL vs exponential weights (B) | Adaptação pós-observação ajuda? |
| C14 | Local vs global (painel, futuro) | Há ganho com transferência entre séries? |
| C15 | Champion automático vs melhor standalone da seleção interna | A automação melhora o procedimento, sem seleção otimista? |

### 7.3. Métricas e relatórios

Manter as métricas existentes; no benchmark da replicação reportar:
- RRMSE, RMAE, sMAPE, MASE; tempos e variabilidade entre sementes;
- desempenho e ganho pareado por série, categoria e horizontes `1,7,14,30,90,180,365`;
- cobertura de execução (inclusive SKIPPED/FAILED), taxa de vitórias, distribuição de diferenças;
- treinamento total, OOF, busca, memória, número de ajustes, previsão e complexidade;
- estabilidade dos pesos ao longo das origens/horizontes e diversidade/correlação de erros;
- comparação com média igual, melhor componente e seleção guardada;
- fronteira de Pareto erro × custo × complexidade.

**Inferência estatística:** não tratar os 365 passos, folds sobrepostos ou múltiplas sementes de uma série como observações independentes. Para análises entre séries, usar série como unidade apropriada; registrar família de hipóteses, efeitos, intervalos e Holm quando aplicável. A estrutura de dependência e o número de unidades limitam a interpretação de Wilcoxon/DM e da seleção incremental.

### 7.4. Ablation studies

- Número de membros: `K=1,2,3` antes de 5/10.
- Média vs mediana vs NNLS; com/sem regularização de pesos.
- Reamostragem em blocos vs só várias sementes vs subspace.
- Bagging antes/depois de decomposição e com diferentes comprimentos de blocos.
- Boosting em lags vs boosting de resíduos; `m=1,2,3` estágios e `nu` pequeno/alto.
- Stacking sem/ com horizonte/contexto; modelos de baixa vs alta capacidade.
- OOF 1 passo vs OOF multipasso para A.
- Diversidade entre famílias vs somente candidatos de menor erro.
- Seleção padrão vs CV aninhada e controle negativo de ruído branco.

### 7.5. Hipóteses científicas, não resultados esperados

- **H1:** agregação simples pode reduzir variância relativamente ao melhor membro em séries heterogêneas.
- **H2:** pesos livres podem piorar frente a pesos uniformes com poucas origens OOF.
- **H3:** stacking por horizonte pode superar pesos globais em A de longo horizonte, quando houver dados suficientes.
- **H4:** boosting de previsores pode corrigir erros persistentes, mas também sobreajustar ruído branco.
- **H5:** ensemble de híbridos agrega valor quando há complementaridade residual real, não somente número de membros.
- **H6:** o ganho aparente diminui ao avaliar todo o procedimento com seleção aninhada.

## 8. Testes técnicos, invariantes e controles anti-leakage

### 8.1. Matriz de testes

| ID | Teste | Aceite |
|---|---|---|
| T01 | Prefix invariance | Alterar `y` depois da origem não altera previsão, features, pesos ou seleção naquela origem |
| T02 | Proveniência OOF | Cada previsão e erro tem origem, corte de treino e `available_at` válidos |
| T03 | Meta-model leakage | Meta-modelo não vê alvos da própria avaliação, `V_select` ou teste |
| T04 | A sem reveal | H passos não usam `y` de passos futuros intermediários |
| T05 | B com parâmetros congelados | Modelos-base não refazem `fit` a cada observação; pesos online são exceção explicitamente declarada |
| T06 | Bagging causal | Bootstrap e decomposição limitados ao histórico disponível |
| T07 | Forecast boosting | Pseudo-resíduos de cada estágio são OOF; sem usar fitted-in-sample para pontuar |
| T08 | Alinhamento | Série, frequência, escala, timestamp, origem e horizonte de todos os membros coincidem |
| T09 | Identidade e cache | Especificações equivalentes geram IDs idênticos; mudança de lógica/protocolo invalida cache |
| T10 | Composição recursiva | Grafo sem ciclos, clone não contamina estado, serialização preserva previsão |
| T11 | Teste selado | Resultados externos nunca alimentam pesos, hiperparâmetros, fallback ou seleção |
| T12 | Incompatibilidades | Modelos sem suporte A/B são rejeitados/registrados, não ajustados silenciosamente |
| T13 | Membros que falham | Política de fallback previamente definida, sem olhar verdadeiro da validação/teste |
| T14 | Painel global | Separar séries/grupos, scaler/embedding e datas de forma cronologicamente válida |
| T15 | Controle negativo | Taxa de falso aceite aferida com diversas sementes de ruído branco |
| T16 | Reprodutibilidade | Ledger, orçamento e seleção estáveis em execuções com mesmo ambiente e seeds |

CI: testes unitários sem extras; extras Nixtla/boosting em jobs independentes; testes de integração em CPU/horizonte curto, e GPU opcional. Não bloquear instalação do núcleo por TensorFlow/PyTorch ou Nixtla. Medir regressões de tempo/memória com baseline conhecido antes de impor thresholds.

## 9. Roadmap por entregas e critérios de aceite

As fases expressam **dependência técnica**, não duração estimada. Os itens são checklists do trabalho futuro.

### F0 — ADRs, contrato e rastreabilidade

- [ ] Redigir ADR da taxonomia (`combination` vs `bagging` vs `boosting` vs `residual_multistage`).
- [ ] Desenhar `CandidateGraphSpec`, identidade canônica, política de versão e compatibilidade legada.
- [ ] Especificar armazenamento OOF e `available_at`.
- [ ] Formalizar capacidade por horizonte, variante, treino local/global e entradas exógenas.
- [ ] Congelar controles, métricas, splits e protocolo inicial de pesquisa.

**Aceite:** ADR revisado; testes de identidade de pequeno grafo; contratos causais definidos; APIs legadas mantidas.

### F1 — Integrações de modelos e paridade

- [ ] Adaptar `StatsForecast` (AutoARIMA sazonal/não sazonal, AutoETS, AutoTheta/AutoCES).
- [ ] Incorporar `LightGBM` e depois XGBoost/CatBoost e AdaBoost.R2/HistGradientBoosting.
- [ ] Disponibilizar SVR, árvores e RandomForest como corretores de janelas.
- [ ] Criar previsor de **sequência residual** separado.
- [ ] Reproduzir paridade técnica dos sete híbridos de `projeto_st_replicacao` com configurações iguais.
- [ ] Validar suporte A/B individualmente para cada novo adaptador, em vez de presumir.

**Aceite:** testes de paridade, causa temporal, horizonte, serialização e avaliação com referência standalone para cada backend.

### F2 — Combinações básicas e ensembles de híbridos

- [ ] Média, mediana, média aparada, pesos inversos e NNLS com shrinkage.
- [ ] Permitir ensemble de bases, ensemble de híbridos e híbrido cuja base é um ensemble.
- [ ] Gerar OOF próprio de composições e comparar contra componentes originais.
- [ ] Diagnóstico da estabilidade dos pesos, cobertura e custo integral.

**Aceite:** todos os tipos funcionam com teste selado e A/B quando elegíveis; comparação com média simples e standalone; falhas registradas.

### F3 — Bagging, diversidade e algoritmos de boosting em lags

- [ ] Implementar moving-block bootstrap **no treino**, com controles de decomposição.
- [ ] Replicar tratamento bagging-ETS da literatura [R03].
- [ ] Random subspace, diversidade por semente e seleção gulosa de membros.
- [ ] Incluir AdaBoost.R2, LightGBM, XGBoost/CatBoost como preditores independentes ou corretores.
- [ ] Executar ablações `C4–C7` e testes `T01–T16` pertinentes.

**Aceite:** bagging comprovadamente causal, avaliação separando fontes de diversidade e comparação honesta com boosting como previsor isolado.

### F4 — Stacking e boosting de previsores completos

- [ ] Construir dataset OOF de membros e meta-modelos (NNLS/Ridge primeiro).
- [ ] Pesos por faixa de horizonte e meta-LightGBM somente depois de estabilizar Ridge.
- [ ] Especificar objetivo, gradiente, `learning_rate` e parada do forecast boosting.
- [ ] Comparar com `residual_multistage` do AutoHybTS, igualando base, estágios e orçamento.
- [ ] Exigir CV aninhada e ruído branco como controle de sobreajuste.

**Aceite:** zero *meta-leakage* nos testes; parâmetros dos estágios preservados; tratamento pode perder e ser rejeitado sem contornar a seleção.

### F5 — NeuralForecast, modelos globais e experts dinâmicos

- [ ] Incorporar NHITS, NBEATSx, PatchTST e avaliar LSTM/TCN/TFT.
- [ ] Integrar MLForecast como backend, incluindo engenharia causal de atributos.
- [ ] Introduzir contrato de painel global e avaliações entre séries.
- [ ] Estudar weighting por features tipo FFORMA, mixture of experts, DSEL generalizado e exponential weights em B.
- [ ] Separar pesos por contexto/horizonte, treinamento e custo de atualização.

**Aceite:** modelos locais/globais comparáveis no painel; teste de vazamento entre séries e de atualização B; matriz de versões e custos.

### F6 — Benchmark, documentação e divulgação

- [ ] Rodar M3, sintéticos, M4 e painel adicional/externo.
- [ ] Produzir resultados por categoria/horizonte, custos, cobertura e incerteza estatística.
- [ ] Confrontar H1–H6 com ablações e controles, sem selecionar só exemplos favoráveis.
- [ ] Adicionar guias, notebooks de ensembles e API de comparação, preservando API preexistente.
- [ ] Consolidar formato de resultados, versionamento de modelos e caminho de upgrade de artefatos.
- [ ] Avaliar probabilidade/reconciliação hierárquica **somente como extensão** posterior.

**Aceite:** o usuário compara individual, híbrido e ensemble no mesmo experimento e consegue auditar seleção, rejeições, custo e validade temporal.

## 10. Artefatos experimentais a produzir

Formato sugerido (não representa arquivos hoje existentes):

```text
artifacts/<run_id>/
  config.json
  manifest.json
  oof/
    predictions.parquet
    provenance.parquet
  search/
    candidate_graphs.jsonl
    trials.parquet
    ledger.parquet
    decisions.parquet
  ensembles/
    members.parquet
    weights.parquet
    stage_diagnostics.parquet
  external/
    predictions.parquet
    metrics.parquet
  reports/
    comparison.md
    ablations.md
    temporal_leakage.md
    cost_pareto.md
```

Registrar `run_id`, `candidate_id`, `trial_key`, hashes, versão de backend, seed, dispositivo, ordem temporal, peso e cobertura. **Não reutilizar artefatos históricos incompatíveis** nem usar o teste externo para seleção.

## 11. Riscos, ameaças à validade e decisões abertas

| Risco | Mitigação / decisão |
|---|---|
| Crescimento exponencial do espaço de busca | Fases S0–S5, diversidade de famílias, orçamento em ajustes e limites de topologia |
| Overfitting de pesos e meta-modelos | OOF interno separado de `V_select`, NNLS/Ridge, regularização, CV aninhada |
| Sobreposição temporal de origens | Dependência explicitada; série como unidade para comparação entre séries |
| Ruído residual sem estrutura | Controle negativo; `alpha=0` e rejeição de complexidade |
| Boosting confundido com correção em etapas | B1/B2/B3 definidos separadamente; ablação com `residual_multistage` |
| Resíduos de 1 passo usados para 365 passos | Treino OOF multipasso em A, estratificado por horizonte |
| Modelos neurais com horizonte fixo | Capacidades e impedimento de uso incompatível |
| Treino global com informação de série externa | Splits entre séries, cutoff temporal e rastreamento por grupo |
| Pesos/estados mutados no teste | Políticas pré-congeladas e teste de prefix invariance |
| Comparar treinamento completo de ensemble com inferência isolada | Custo da busca, geração OOF, ajuste, CPU/GPU e inferência |
| Teste M4 já inspecionado em pesquisa anterior | Identificar análises retrospectivas e reservar novo teste externo para confirmação |
| Mudança de comportamento do código legado | IDs/serialização versionados, fixtures de paridade e migração |
| Bibliotecas opcionais com dependências incompatíveis | Extras desacoplados, versões testadas e matriz CI |

**ADRs pendentes:** (i) grafo novo ou extensão do spec; (ii) esquema OOF e janela de meta-aprendizado; (iii) definição exata de B3 e perdas; (iv) políticas de fallback; (v) treino global/funcionalidade exógena; (vi) limites de orçamento por família. Resolver antes de codificar extensões avançadas.

## 12. Referências científicas e documentação técnica

> Trabalhos citados fornecem base conceitual e comparadores, **não evidência de desempenho do AutoHybTS**. O boosting de previsores completos (E11) é proposição experimental que deve ser verificada.

- **[R01]** Bates, J. M.; Granger, C. W. J. (1969). *The Combination of Forecasts*. Journal of the Operational Research Society, 20(4), 451–468. [DOI 10.1057/jors.1969.103](https://doi.org/10.1057/jors.1969.103).
- **[R02]** Breiman, L. (1996). *Bagging Predictors*. Machine Learning, 24, 123–140. [DOI 10.1007/BF00058655](https://doi.org/10.1007/BF00058655).
- **[R03]** Bergmeir, C.; Hyndman, R. J.; Benítez, J. M. (2016). *Bagging exponential smoothing methods using STL decomposition and Box–Cox transformation*. International Journal of Forecasting, 32(2), 303–312. [DOI 10.1016/j.ijforecast.2015.07.002](https://doi.org/10.1016/j.ijforecast.2015.07.002).
- **[R04]** Wolpert, D. H. (1992). *Stacked Generalization*. Neural Networks, 5(2), 241–259. [DOI 10.1016/S0893-6080(05)80023-1](https://doi.org/10.1016/S0893-6080(05)80023-1).
- **[R05]** Breiman, L. (1996). *Stacked Regressions*. Machine Learning, 24, 49–64. [DOI 10.1007/BF00117832](https://doi.org/10.1007/BF00117832).
- **[R06]** Caruana, R.; Niculescu-Mizil, A.; Crew, G.; Ksikes, A. (2004). *Ensemble Selection from Libraries of Models*. ICML. [DOI 10.1145/1015330.1015432](https://doi.org/10.1145/1015330.1015432).
- **[R07]** Friedman, J. H. (2001). *Greedy Function Approximation: A Gradient Boosting Machine*. Annals of Statistics, 29(5). [DOI 10.1214/aos/1013203451](https://doi.org/10.1214/aos/1013203451).
- **[R08]** Chen, T.; Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. KDD. [DOI 10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785).
- **[R09]** Ke, G. et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. NeurIPS 30. [Artigo](https://proceedings.neurips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html).
- **[R10]** Prokhorenkova, L. et al. (2018). *CatBoost: unbiased boosting with categorical features*. NeurIPS 31. [Artigo](https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html).
- **[R11]** Drucker, H. (1997). *Improving Regressors Using Boosting Techniques*. ICML, 107–115. [Artigo](https://mlanthology.org/icml/1997/drucker1997icml-improving/).
- **[R12]** Montero-Manso, P.; Athanasopoulos, G.; Hyndman, R. J.; Talagala, T. S. (2020). *FFORMA: Feature-based forecast model averaging*. International Journal of Forecasting, 36(1), 86–92. [DOI 10.1016/j.ijforecast.2019.02.011](https://doi.org/10.1016/j.ijforecast.2019.02.011).
- **[R13]** Makridakis, S.; Spiliotis, E.; Assimakopoulos, V. (2018). *The M4 Competition: Results, findings, conclusion and way forward*. International Journal of Forecasting, 34(4), 802–808. [DOI 10.1016/j.ijforecast.2018.06.001](https://doi.org/10.1016/j.ijforecast.2018.06.001).
- **[R14]** Claeskens, G.; Magnus, J. R.; Vasnev, A. L.; Wang, W. (2016). *The forecast combination puzzle: A simple theoretical explanation*. International Journal of Forecasting, 32(3), 754–762. [DOI 10.1016/j.ijforecast.2015.12.005](https://doi.org/10.1016/j.ijforecast.2015.12.005).
- **[R15]** Jacobs, R. A.; Jordan, M. I.; Nowlan, S. J.; Hinton, G. E. (1991). *Adaptive Mixtures of Local Experts*. Neural Computation, 3(1), 79–87. [DOI 10.1162/neco.1991.3.1.79](https://doi.org/10.1162/neco.1991.3.1.79).
- **[R16]** Cesa-Bianchi, N.; Lugosi, G. (2006). *Prediction, Learning, and Games*. Cambridge University Press. [DOI 10.1017/CBO9780511546921](https://doi.org/10.1017/CBO9780511546921).
- **[R17]** Smyl, S. (2020). *A hybrid method of exponential smoothing and recurrent neural networks for time series forecasting*. International Journal of Forecasting, 36(1), 75–85. [DOI 10.1016/j.ijforecast.2019.03.017](https://doi.org/10.1016/j.ijforecast.2019.03.017).
- **[R18]** Makridakis, S.; Spiliotis, E.; Assimakopoulos, V. (2022). *M5 accuracy competition: Results, findings, and conclusions*. International Journal of Forecasting, 38(4), 1346–1364. [DOI 10.1016/j.ijforecast.2021.11.013](https://doi.org/10.1016/j.ijforecast.2021.11.013).
- **[R19]** Bergmeir, C.; Hyndman, R. J.; Koo, B. (2018). *A note on the validity of cross-validation for evaluating autoregressive time series prediction*. Computational Statistics & Data Analysis, 120, 70–83. [DOI 10.1016/j.csda.2017.11.003](https://doi.org/10.1016/j.csda.2017.11.003).
- **[R20]** Hansen, P. R.; Lunde, A.; Nason, J. M. (2011). *The Model Confidence Set*. Econometrica, 79(2), 453–497. [DOI 10.3982/ECTA5771](https://doi.org/10.3982/ECTA5771).
- **Nixtla:** [StatsForecast](https://nixtlaverse.nixtla.io/statsforecast/), [MLForecast](https://nixtla.github.io/mlforecast/), [NeuralForecast](https://nixtlaverse.nixtla.io/neuralforecast/).

---

## Critério geral de conclusão

A ampliação estará consolidada quando seja possível construir e comparar **standalone, híbrido, ensemble, boosting e ensemble de híbridos**, aplicar o **mesmo protocolo temporal**, registrar a razão de cada seleção/rejeição e publicar resultados que expressem **ganho pareado, estabilidade, custos e ameaças à validade**. Isso inclui reconhecer quando a melhor decisão é não combinar modelos.
