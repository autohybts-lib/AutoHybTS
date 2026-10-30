# AutoHybTS tutorials (English)

English copies of the tutorials in `notebooks/` (Portuguese), cell by cell, with the same code, metadata and outputs. Each notebook is PyCaret-4 style: short, numbered, one topic per notebook, using the object-oriented API (`AutoHybridExperiment`).

| Notebook | Content | Requires |
|---|---|---|
| [01_quickstart](01_quickstart.ipynb) | Load a series, compare models, forecast and evaluate on the sealed test | — |
| [02_hybrid_selection](02_hybrid_selection.ipynb) | Staged search, a hand-built hybrid, selection decisions, trial audit, diagnostics and events | — |
| [03_finalize_and_persist](03_finalize_and_persist.ipynb) | Finalize, evaluate once, save and load with a manifest, operational forecasting without a sealed test | — |
| [04_rolling_one_step](04_rolling_one_step.ipynb) | Variant B: one-step forecasting with frozen parameters and streaming observations | — |
| [05_benchmark_m4](05_benchmark_m4.ipynb) | M4 with the official test sealed, an M3 panel and Rdatasets (downloads the data on the first run) | network the first time; `datasets` extra for M3 |
| [06_compare_and_cross_validate](06_compare_and_cross_validate.ipynb) | Ranking of the finalists, temporal cross-validation, comparison plots, statistical tests, comparison on the sealed test and nested CV | — |
| [07_panel](07_panel.ipynb) | Several series in the same experiment: ranking across series, selection over the whole set, one model per series and external evaluation per series | — |
| [08_advanced_families_and_tuning](08_advanced_families_and_tuning.ipynb) | Extra stages, decomposition, fixed combination, search with every family and `tune_model` | — |
| [09_ablation](09_ablation.ipynb) | Ablation of the selection procedure (§30.5): 4 procedures, same budget and same external tests, on 12 synthetic series × 3 seeds, `small` budget | heavy (≈55 min) |
| [10_external_models](10_external_models.ipynb) | Your own models: PyTorch (LSTM, Transformer, Mamba, a GRU written on the spot), TensorFlow/Keras (LSTM, Transformer, convolution), statsmodels' Holt-Winters and gradient boosting, as bases, correctors, in the search, in tuning and in persistence | `torch` and `tensorflow` extras; heavy |
| [11_calibration](11_calibration.ipynb) | Calibration of the acceptance rule on controls with a known answer (white noise and predictable residual): false acceptances, rejected gains, effect of the number of origins and of the Wilcoxon test, confirmation set | — |
| [12_baselines_and_boosting](12_baselines_and_boosting.ipynb) | Baselines, boosting on lags (HistGradientBoosting, AdaBoost.R2, LightGBM, XGBoost, CatBoost) as bases and correctors, RMAE and fit counts | `boosting` extra |
| [13_ensembles](13_ensembles.ipynb) | Ensembles of complete candidates (mean, median, trimmed mean, inverse MSE, NNLS with shrinkage), ensemble of hybrids, corrected ensemble and ensembles in the search with the acceptance rule | — |
| [14_intermittent_demand](14_intermittent_demand.ipynb) | Intermittent demand: Croston-SBA, TSB, ADIDA and IMAPA against simple means, why RMSSE instead of MASE, rate changes, demand that stops and the `intermittent` family in the search | — |
| [15_statsforecast](15_statsforecast.ipynb) | StatsForecast's AutoARIMA, AutoETS, AutoTheta and AutoCES against statsmodels and baselines, frozen parameters (`forward`), the statistical-ANN hybrids of the replication and the search with these models | `nixtla-stats` extra (brings pandas 2.x) |
| [17_benchmark_study](17_benchmark_study.ipynb) | 17 classical series in three classes: five models (with the ARIMA–ANN hybrid) in protocols A and B, with 3, 6 and 12 lags and horizons up to 12 | `stats` and `plot` extras; network on first run |
| [18_sql_analysis](18_sql_analysis.ipynb) | Experiments stored in SQLite with `ExperimentStore` and analysed with SQL queries | `stats` and `plot` extras |
| [19_mlops_flow](19_mlops_flow.ipynb) | MLOps flow with a commodity price: registration, promotion, daily forecasts, monitoring and gated retraining | `stats` and `plot` extras |
| [20_exogenous](20_exogenous.ipynb) | Series with exogenous variables: past × future, oracle, automatic selection and with × without covariates on benchmarks | `stats` and `plot` extras; network on first run |
| [21_mlops_exogenous](21_mlops_exogenous.ipynb) | MLOps with covariates on real data: Victoria electricity demand with weather forecast, holiday and weekend | `stats` and `plot` extras; network on first run |
| [22_m4_development_pilot](22_m4_development_pilot.ipynb) | A daily M4 project on official training data only: guarded selection against lowest error, logging and resume in SQLite and cost | `stats` and `plot` extras; network on first run |

## How to run

To use the tutorials with the candidate published on PyPI:

```bash
python -m pip install "autohybts[stats,notebook]==1.0.0rc5"
```

Notebooks 05, 10, 12 and 15 require the extras listed in the table. See the
[installation guide](../../documentation/en/installation.rst) for other extras.
To develop and run tests from a local clone:

```bash
.venv/Scripts/python -m pip install -e ".[dev,stats,notebook]"
```

In VS Code, open the notebook and select the `.venv` kernel. Run the notebooks from this folder (`notebooks/en`); notebook 15 needs the environment with the `nixtla-stats` extra.

A change to a tutorial goes into both languages; `tests/contracts/test_docs_translation.py` checks that every Portuguese notebook has an English copy with the same metadata. The rules (sections, metadata, execution and outputs) are in `documentation/en/contributing.rst`.
