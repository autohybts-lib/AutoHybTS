# Tutoriais da AutoHybTS

Notebooks no estilo dos tutoriais do PyCaret 4: curtos, numerados, um tema por notebook, usando a API orientada a objetos (`AutoHybridExperiment`).

| Notebook | Conteúdo | Requer |
|---|---|---|
| [01_quickstart](01_quickstart.ipynb) | Carregar uma série, comparar modelos, prever e avaliar no teste selado | etapa I8 |
| [02_hybrid_selection](02_hybrid_selection.ipynb) | Busca em estágios, híbrido montado à mão, decisões de seleção, auditoria de tentativas, diagnósticos e eventos | etapas I8 e I11 |
| [03_finalize_and_persist](03_finalize_and_persist.ipynb) | Finalizar, avaliar uma vez, salvar e carregar com manifesto, previsão operacional sem teste selado | etapa I8 |
| [04_rolling_one_step](04_rolling_one_step.ipynb) | Variante B: previsão de um passo com parâmetros congelados e streaming de observações | etapa I10 |
| [06_compare_and_cross_validate](06_compare_and_cross_validate.ipynb) | Ranking das finalistas, validação cruzada temporal, gráficos de comparação, testes estatísticos, comparação no teste selado e CV aninhada | — |
| [07_panel](07_panel.ipynb) | Várias séries no mesmo experimento: ranking entre séries, seleção sobre o conjunto, um modelo por série e avaliação externa por série | — |
| [08_advanced_families_and_tuning](08_advanced_families_and_tuning.ipynb) | Estágios extras, decomposição, combinação fixa, busca com todas as famílias e `tune_model` | — |
| [05_benchmark_m4](05_benchmark_m4.ipynb) | Uma série do M4 com o teste oficial selado (baixa os dados na primeira execução) | etapa I9 |

## Como executar

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
| 05_benchmark_m4 | aguarda o catálogo de datasets (etapa I9) |
| 06_compare_and_cross_validate | roda (≈50 s) |

O teste `tests/integration/test_notebooks.py` executa cada notebook assim que todos os nomes listados em `metadata.autohybts.requires` existem no pacote:

```bash
.venv/Scripts/python -m pytest tests/integration/test_notebooks.py -m slow -rs
```
