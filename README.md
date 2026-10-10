<p align="center">
  <img src="figures/AutoHybTS.png" alt="AutoHybTS" width="260">
</p>

# AutoHybTS

**Automated Hybrid Time Series Forecasting.** Biblioteca Python que automatiza a comparação de modelos individuais de previsão e a descoberta de **sistemas híbridos residuais** (modelo-base + corretor do erro), aceitando complexidade adicional só quando critérios de validação temporal definidos pelo usuário são satisfeitos.

A interface é inspirada no [PyCaret](https://pycaret.org) — um experimento, métodos curtos, resultados prontos para tabela e gráfico —, mas o foco é outro: reunir num só fluxo, com validação sem vazamento, os modelos híbridos residuais para previsão de séries temporais e as estratégias de montá-los (correção aditiva, vários estágios, decomposição, combinação).

> **Status: pré-alfa (0.0.1).** A API abaixo funciona para uma série ou um painel de séries, nas variantes A (multipasso) e B (rolling de um passo com parâmetros congelados). Ainda não publicada no PyPI.

## Instalação

Requer Python 3.11 ou mais novo.

**Pelo pip** (quando a versão for publicada no PyPI):

```bash
pip install autohybts                     # núcleo: NumPy, pandas, scikit-learn
pip install "autohybts[stats,plot]"       # + ARIMA/STL (statsmodels) e gráficos (matplotlib)
pip install "autohybts[stats,plot,optuna]"  # + ajuste de hiperparâmetros com Optuna
```

**Pelo git** (versão em desenvolvimento; troque `<organizacao>` pelo endereço do repositório quando ele for publicado):

```bash
pip install "autohybts[stats,plot] @ git+https://github.com/<organizacao>/AutoHybTS.git"
# uma versão específica:
pip install "autohybts[stats,plot] @ git+https://github.com/<organizacao>/AutoHybTS.git@v0.1.0"
```

**A partir do código-fonte** (para desenvolver):

```bash
git clone https://github.com/<organizacao>/AutoHybTS.git
cd AutoHybTS
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev,stats,notebook]"   # Windows
# .venv/bin/python -m pip install -e ".[dev,stats,notebook]"     # Linux/macOS
```

Extras: `stats` (statsmodels: ARIMA e STL), `plot` (matplotlib), `optuna` (`tune_model` com TPE), `datasets` (ler o M3), `yaml`, `notebook` (rodar os tutoriais), `docs` (gerar a documentação), `dev` (testes e lint).

## Início rápido

```python
from autohybts import AutoHybridExperiment, load_model, save_model
from autohybts.datasets import get_data

y = get_data("synthetic_daily")  # pd.Series diária
exp = AutoHybridExperiment(fh=14, session_id=42).fit(y)  # sela os últimos 14 pontos

result = exp.compare_models()  # busca em V_search + seleção incremental em V_select
result.leaderboard  # todas as candidatas avaliadas na busca
result.decisions  # cada desafiante (ex.: arima-mlp): aceito ou não, e por quê

final = exp.finalize_model(result.best)  # reajuste no histórico anterior ao teste
exp.predict_model(final).predictions  # previsões indexadas por data
exp.evaluate_external(final).metrics  # teste selado, uma única vez

path = save_model(final, "artifacts/champion")
model = load_model(path, trusted=True)  # trusted=True: só para arquivos de confiança
```

### Estratégias de montar o sistema híbrido

```python
result = exp.compare_models(
    include=("standalone", "residual", "multistage", "decomposition", "combination")
)
result.strategies  # melhor e mediana de cada estratégia
result.plot("pareto")  # erro × custo; também x="complexity" e "strategies"

exp.create_hybrid("arima", "mlp")  # ŷ = b + α·r̂
exp.create_hybrid("arima", "mlp", extra_stages=["linear"])  # correção da correção
exp.create_decomposition("linear", "mlp", method="ma")  # suave + volátil
exp.create_combination(["arima", "knn"], weights="inverse_mse")
exp.tune_model("knn", n_trials=20)  # só em V_search; Optuna se instalado
```

### Comparação e validação cruzada temporal

```python
cv = exp.cross_validate(folds=8, step=7)  # mesmas origens para todos os modelos
cv.summary  # média, desvio, posto médio
cv.tests()  # Wilcoxon e Diebold-Mariano com Holm (relatório)
cv.plot("critical_difference")  # também: leaderboard, boxplot, win_rate, forecasts
exp.nested_cross_validate(outer_folds=3)  # o procedimento inteiro, dobra a dobra
```

### Dados incluídos

A biblioteca já traz datasets nomeados, prontos para usar — os nomes abaixo funcionam direto no código:

| Dataset | Fonte | O que é |
|---|---|---|
| `synthetic_daily` | sintética (offline) | 730 dias com tendência, sazonalidade semanal e ruído AR(1): estrutura residual que um corretor pode capturar |
| `white_noise_daily` | sintética (offline) | nível + ruído branco, sem padrão: controle negativo |
| `m4:Y1` … `m4:Y23000` | M4 yearly | 23.000 séries anuais |
| `m4:Q1` … `m4:Q24000` | M4 quarterly | 24.000 séries trimestrais |
| `m4:M1` … `m4:M1480` | M4 monthly | 1.480 séries mensais |
| `m4:W1` … `m4:W359` | M4 weekly | 359 séries semanais |
| `m4:D1` … `m4:D4227` | M4 daily | 4.227 séries diárias |
| `m4:H1` … `m4:H414` | M4 hourly | 414 séries horárias |
| `m3:N1402` e demais do catálogo | M3 | 3.003 séries nos subconjuntos `yearly` (645), `quarterly` (756), `monthly` (1428) e `other` (174); exige o extra `datasets` |
| `datasets::AirPassengers` e demais `<pacote>::<nome>` | Rdatasets | séries reais dos pacotes R (ex.: passageiros aéreos mensais) |

Nas séries de benchmark (M4/M3) o teste oficial da competição vem no final, selado. O **M5 não é suportado**: é hierárquico e o score oficial (WRMSSE) não está implementado — nenhum resultado desta biblioteca deve ser apresentado como benchmark M5.

```python
from autohybts.datasets import get_data, list_datasets, load_benchmark, load_dataset

y = get_data("synthetic_daily")  # offline, na hora

series = load_benchmark("m4", subset="daily", series_id="D1")  # baixa uma vez, com checksum
exp = AutoHybridExperiment(fh=series.descriptor.horizon, test_size=len(series.test)).fit(
    series.full()
)  # teste oficial selado
air = load_dataset(
    "rdatasets",
    package="datasets",
    name="AirPassengers",
    target="value",
    time_col="time",
    frequency="MS",
)
```

### Várias séries

```python
exp = AutoHybridExperiment(fh=14).fit({"loja_a": y_a, "loja_b": y_b})
result = exp.compare_models()  # ranking pela média dos postos entre séries
model = exp.finalize_model(result.best)  # PanelForecaster: um modelo por série
exp.evaluate_external(model).metrics  # uma linha por série + média
```

Tutoriais em [`notebooks/`](notebooks/README.md); documentação em [`documentation/`](documentation/) (`sphinx-build documentation documentation/_build/html`).

## Princípios

- Validade temporal antes de automação: nenhuma informação posterior à origem de previsão entra como feature, e o teste selado nunca alimenta busca ou seleção.
- Seleção incremental guardada: um desafiante só substitui o campeão atual se superar critérios pareados de ganho e vitórias; caso contrário, a estrutura anterior (inclusive um modelo individual) é mantida.
- Rastreabilidade: candidatas, tentativas e decisões têm identidade estável e ficam registradas, inclusive rejeições e falhas.

## Verificação

```bash
ruff check . && ruff format --check .
mypy
pytest -m "not slow"
```

## Limitações atuais

- Decomposição, combinação fixa e estágios múltiplos só entram na busca quando pedidos em `compare_models(include=...)`; a CV aninhada é só para uma série; a seleção dinâmica (`dynamic`) só existe na variante B; o M5 não é suportado.
- `compare_models` com o orçamento padrão (`small`) e o espaço completo leva alguns minutos numa série de ~700 pontos, sobretudo por causa do ARIMA com ordem automática; use `budget="smoke"` ou `base_models=[...]` para explorar.
- A regra de aceite do protocolo (ganho mediano ≥ 1 %, vitórias ≥ 60 %, variabilidade ≥ 0,05) não é controle de erro estatístico: em ruído branco ela aceitou híbridos espúrios em parte das sementes testadas.

## Créditos

Inspirada na ergonomia do [PyCaret](https://pycaret.org); o PyCaret não é dependência.
