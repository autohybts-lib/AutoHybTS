# Tutoriais da AutoHybTS

Notebooks no estilo dos tutoriais do PyCaret 4: curtos, numerados, um tema por notebook, usando a API orientada a objetos (`AutoHybridExperiment`). As cópias em inglês, célula a célula, estão em [`en/`](en/README.md); uma mudança num tutorial entra nos dois idiomas.

| Notebook | Conteúdo | Requer |
|---|---|---|
| [01_quickstart](01_quickstart.ipynb) | Carregar uma série, comparar modelos, prever e avaliar no teste selado | etapa I8 |
| [02_hybrid_selection](02_hybrid_selection.ipynb) | Busca em estágios, híbrido montado à mão, decisões de seleção, auditoria de tentativas, diagnósticos e eventos | etapas I8 e I11 |
| [03_finalize_and_persist](03_finalize_and_persist.ipynb) | Finalizar, avaliar uma vez, salvar e carregar com manifesto, previsão operacional sem teste selado | etapa I8 |
| [04_rolling_one_step](04_rolling_one_step.ipynb) | Variante B: previsão de um passo com parâmetros congelados e streaming de observações | etapa I10 |
| [06_compare_and_cross_validate](06_compare_and_cross_validate.ipynb) | Ranking das finalistas, validação cruzada temporal, gráficos de comparação, testes estatísticos, comparação no teste selado e CV aninhada | — |
| [07_panel](07_panel.ipynb) | Várias séries no mesmo experimento: ranking entre séries, seleção sobre o conjunto, um modelo por série e avaliação externa por série | — |
| [08_advanced_families_and_tuning](08_advanced_families_and_tuning.ipynb) | Estágios extras, decomposição, combinação fixa, busca com todas as famílias e `tune_model` | — |
| [09_ablation](09_ablation.ipynb) | Ablação do procedimento de seleção (§30.5): 4 procedimentos, mesmo orçamento e mesmos testes externos, em 12 séries sintéticas × 3 sementes, orçamento `small` | pesado (≈55 min) |
| [10_external_models](10_external_models.ipynb) | Modelos próprios: PyTorch (LSTM, Transformer, Mamba, GRU escrito na hora), TensorFlow/Keras (LSTM, Transformer, convolução), Holt-Winters do statsmodels e gradient boosting, como bases, corretores, na busca, no ajuste e na persistência | extras `torch` e `tensorflow`; pesado |
| [11_calibration](11_calibration.ipynb) | Calibração da regra de aceite em controles com resposta conhecida (ruído branco e resíduo previsível): falsos aceites, ganhos recusados, efeito do número de origens e do teste de Wilcoxon, conjunto de confirmação | — |
| [12_baselines_and_boosting](12_baselines_and_boosting.ipynb) | Baselines (média histórica, média em janela, média sazonal em janela, passeio aleatório com drift), boosting sobre lags (HistGradientBoosting, AdaBoost.R2, LightGBM) como bases e corretores, RMAE e contagem de ajustes | LightGBM no extra `boosting` |
| [13_ensembles](13_ensembles.ipynb) | Ensembles de candidatas completas (média, mediana, média aparada, inverso do MSE, NNLS com encolhimento), ensemble de híbridos, ensemble corrigido e ensembles na busca com a regra de aceite | — |
| [14_intermittent_demand](14_intermittent_demand.ipynb) | Demanda intermitente: Croston-SBA, TSB, ADIDA e IMAPA contra médias simples, por que a RMSSE em vez da MASE, mudança de taxa, demanda que para e a família `intermittent` na busca | — |
| [15_statsforecast](15_statsforecast.ipynb) | AutoARIMA, AutoETS, AutoTheta e AutoCES do StatsForecast contra statsmodels e baselines, parâmetros congelados (`forward`), os híbridos estatístico-ANN da replicação e a busca com esses modelos | extra `nixtla-stats` (traz pandas 2.x) |
| [17_benchmark_study](17_benchmark_study.ipynb) | 17 séries clássicas em três classes: cinco modelos (com o híbrido ARIMA–ANN) nos protocolos A e B, com 3, 6 e 12 lags e horizontes até 12 | extras `stats` e `plot`; rede na primeira execução |
| [18_sql_analysis](18_sql_analysis.ipynb) | Experimentos gravados em SQLite com `ExperimentStore` e analisados com consultas SQL | extras `stats` e `plot` |
| [05_benchmark_m4](05_benchmark_m4.ipynb) | M4 com o teste oficial selado, painel do M3 e Rdatasets (baixa os dados na primeira execução) | rede na 1ª vez; extra `datasets` para o M3 |

## Como executar

Para usar os tutoriais com a candidata publicada no PyPI:

```bash
python -m pip install "autohybts[stats,notebook]==1.0.0rc3"
```

Os notebooks 05, 10, 12 e 15 requerem os extras indicados na tabela. Consulte o
[guia de instalação](../documentation/installation.rst) para instalar outros extras.
Para desenvolver e executar os testes a partir de um clone local:

```bash
.venv/Scripts/python -m pip install -e ".[dev,stats,notebook]"
```

No VS Code, abra o notebook e selecione o kernel do `.venv`.

## Estado

| Notebook | Situação |
|---|---|
| 01_quickstart | roda (≈1–2 min com o orçamento padrão) |
| 03_finalize_and_persist | roda (≈3–6 min: dois `compare_models`) |
| 02_hybrid_selection | roda (≈1,5 min) |
| 04_rolling_one_step | roda (≈2,5–4,5 min) |
| 07_panel | roda |
| 08_advanced_families_and_tuning | roda (≈1 min) |
| 09_ablation | roda (≈55 min com o orçamento `small`); nos testes só com `AUTOHYBTS_RUN_HEAVY=1` |
| 05_benchmark_m4 | roda (≈25 s com cache; o primeiro download do M4 diário tem ~100 MB) |
| 06_compare_and_cross_validate | roda (≈50 s) |
| 10_external_models | roda (≈6 min); nos testes só com `AUTOHYBTS_RUN_HEAVY=1` |
| 11_calibration | roda (≈8 min) |
| 12_baselines_and_boosting | roda (≈1,5 min) |
| 13_ensembles | roda (≈4–5 min) |
| 14_intermittent_demand | roda (≈15 s) |
| 15_statsforecast | roda no ambiente com o extra `nixtla-stats` (pandas 2.x) |

O teste `tests/integration/test_notebooks.py` executa cada notebook assim que todos os nomes listados em `metadata.autohybts.requires` existem no pacote:

```bash
.venv/Scripts/python -m pytest tests/integration/test_notebooks.py -m slow -rs
```

Notebooks marcados `heavy` só rodam com `AUTOHYBTS_RUN_HEAVY=1`; um notebook cujos `extras` (declarados nos metadados) não estão instalados é pulado com o nome do que falta.

## Novo tutorial

O esqueleto no modelo de aula, com os metadados já no esquema, sai de:

```bash
.venv/Scripts/python notebooks/new_notebook.py 15_temporal_bagging "Bagging temporal" --guide user_guide/ensembles --requires autohybts.AutoHybridExperiment.compare_models
```

As regras (seções, metadados, execução e saídas) estão em `documentation/contributing.rst`.
