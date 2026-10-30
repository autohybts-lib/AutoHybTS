<p align="center">
  <img src="https://raw.githubusercontent.com/autohybts-lib/AutoHybTS/main/figures/AutoHybTS.png" alt="AutoHybTS" width="260">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11%2B-blue.svg" alt="Python 3.11+">
  <a href="https://pypi.org/project/autohybts/"><img src="https://img.shields.io/pypi/v/autohybts.svg" alt="PyPI"></a>
  <a href="https://autohybts.readthedocs.io/"><img src="https://img.shields.io/readthedocs/autohybts/latest" alt="Documentação"></a>
  <a href="https://github.com/autohybts-lib/AutoHybTS/actions/workflows/ci.yml"><img src="https://github.com/autohybts-lib/AutoHybTS/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/autohybts-lib/AutoHybTS/blob/main/LICENSE"><img src="https://img.shields.io/badge/licen%C3%A7a-MIT-green.svg" alt="Licença MIT"></a>
</p>

# AutoHybTS

**Automated Hybrid Time Series Forecasting.** Biblioteca Python que automatiza a comparação de modelos individuais de previsão e a descoberta de **sistemas híbridos residuais** (modelo-base + corretor do erro), aceitando complexidade adicional só quando critérios de validação temporal definidos pelo usuário são satisfeitos.

A interface é inspirada no [PyCaret](https://pycaret.org) — um experimento, métodos curtos, resultados prontos para tabela e gráfico —, mas o foco é outro: reunir num só fluxo, com validação sem vazamento, os modelos híbridos residuais para previsão de séries temporais e as estratégias de montá-los (correção aditiva, vários estágios, decomposição, combinação).

> **Status: candidata à v1 (`1.0.0rc4`), [publicada no PyPI](https://pypi.org/project/autohybts/1.0.0rc4/).** A API funciona para uma série ou um painel de séries, nas variantes A (multipasso) e B (rolling de um passo com parâmetros congelados). Esta é uma pré-versão para validação; a `1.0.0` estável depende da conferência final e do estudo em séries reais. Veja as [notas e a política de compatibilidade](https://autohybts.readthedocs.io/pt-br/latest/releasing.html).

## Instalação

Requer Python 3.11 ou mais novo.

Instale pelo PyPI com os modelos estatísticos e gráficos usados nos exemplos:

```bash
python -m pip install "autohybts[stats,plot]==1.0.0rc4"
```

A versão está fixada porque `1.0.0rc4` é uma candidata. Para instalar apenas o núcleo,
use `python -m pip install "autohybts==1.0.0rc4"`. A instalação pelo PyPI não requer Git.

Para usar modelos próprios em PyTorch ou TensorFlow, acrescente os extras `torch` e/ou
`tensorflow` (ex.: `autohybts[stats,plot,torch]==1.0.0rc4`). Para notebooks, use o extra
`notebook`. Veja o [guia de instalação](https://autohybts.readthedocs.io/pt-br/latest/installation.html) para Jupyter,
extras e instalação de desenvolvimento.

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

Carregar um modelo executa código (pickle). Para artefatos que passam por outras mãos,
assine com `save_model(..., signing_key=chave)` e confira com
`load_model(..., signing_key=chave)`, ou fixe o hash com `load_model(..., sha256=...)`.

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

### Modelos próprios (PyTorch, TensorFlow, qualquer biblioteca)

```python
from autohybts.models import keras_model, keras_zoo, torch_model
from autohybts.models.torch_zoo import LSTMNet, MambaNet

exp.register_model(
    "lstm",
    torch_model(LSTMNet, hidden=32, lags=28, differencing=1),
    family="deep",
    requires={"torch": ">=2.2"},
    version="1",
)
exp.register_model(
    "mamba",
    torch_model(MambaNet, d_model=32, lags=28, differencing=1),
    family="deep",
    requires={"torch": ">=2.2"},
    version="1",
)
exp.register_model(
    "keras_lstm",
    keras_model(keras_zoo.lstm, units=32, lags=28),
    family="deep",
    requires={"tensorflow": ">=2.16"},
    version="1",
)
# depois, os nomes valem em base_models, correctors, create_hybrid, tune_model ...
```

### Várias séries

```python
exp = AutoHybridExperiment(fh=14).fit({"loja_a": y_a, "loja_b": y_b})
result = exp.compare_models()  # ranking pela média dos postos entre séries
model = exp.finalize_model(result.best)  # PanelForecaster: um modelo por série
exp.evaluate_external(model).metrics  # uma linha por série + média
```

## Datasets

Copie o nome e passe para `get_data`; o resultado é uma `pd.Series` pronta para o `fit`.

```python
from autohybts.datasets import get_data

y = get_data("ozone")
```

| Nome | O que é | Frequência | Pontos |
|---|---|---|---|
| `synthetic_daily` | sintética: tendência, sazonalidade semanal e ruído AR(1) (offline) | diária | 730 |
| `white_noise_daily` | sintética: nível + ruído branco, controle negativo (offline) | diária | 730 |
| `air_passengers` | passageiros aéreos, 1949–1960 | mensal | 144 |
| `nottem` | temperatura em Nottingham, 1920–1939 | mensal | 240 |
| `us_acc_deaths` | mortes acidentais nos EUA, 1973–1978 | mensal | 72 |
| `sunspots_monthly` | manchas solares, 1749–1983 | mensal | 2820 |
| `uk_gas` | consumo de gás no Reino Unido, 1960–1986 | trimestral | 108 |
| `johnson_johnson` | lucro por ação da Johnson & Johnson, 1960–1980 | trimestral | 84 |
| `ozone` | ozônio total em Halley Bay (Antártida), 1956–2000 | anual | 45 |
| `nile` | vazão do rio Nilo, 1871–1970 | anual | 100 |
| `lynx` | linces capturados no Canadá, 1821–1934 | anual | 114 |
| `lake_huron` | nível do lago Huron, 1875–1972 | anual | 98 |
| `sunspots` | manchas solares, 1700–1988 | anual | 289 |
| `m4:D1` | M4: troque `D1` por `Y…`, `Q…`, `M…`, `W…`, `D…` ou `H…` (100 mil séries) | várias | — |
| `m3:N1402` | M3: 3.003 séries; exige o extra `datasets` | várias | — |

As sintéticas são geradas na hora. As outras são baixadas na primeira vez e ficam em cache
(`~/.cache/autohybts/datasets`); as reais vêm do [Rdatasets](https://vincentarelbundock.github.io/Rdatasets/).

Nas séries do M4 e do M3 o teste oficial da competição vem no fim; sele-o com `test_size`:

```python
from autohybts.datasets import load_benchmark

series = load_benchmark("m4", subset="daily", series_id="D1")
exp = AutoHybridExperiment(fh=series.descriptor.horizon, test_size=len(series.test)).fit(
    series.full()
)
```

Qualquer outra série do Rdatasets entra por `load_dataset`, dizendo qual coluna é o tempo,
qual é o alvo e a frequência (veja a documentação, *Datasets*). A biblioteca não preenche
valores ausentes: trate-os antes do `fit`.

## Documentação e tutoriais

- Documentação: <https://autohybts.readthedocs.io/>
- Tutoriais em notebook, salvos com as saídas: [`notebooks/`](https://github.com/autohybts-lib/AutoHybTS/tree/main/notebooks)

Para configurar a publicação no Read the Docs, veja
[`Publicação no Read the Docs`](https://autohybts.readthedocs.io/pt-br/latest/maintainers.html).
O projeto em português (`/pt-br/latest/`) deve usar **Language: Portuguese (Brazil)**
e **Build configuration file: `documentation/.readthedocs.yaml`**. O YAML da raiz
(`.readthedocs.yaml`) compila a versão em inglês.

## Princípios

- Validade temporal antes de automação: nenhuma informação posterior à origem de previsão entra como feature, e o teste selado nunca alimenta busca ou seleção.
- Seleção incremental guardada: um desafiante só substitui o campeão atual se superar critérios pareados de ganho e vitórias; caso contrário, a estrutura anterior (inclusive um modelo individual) é mantida.
- Rastreabilidade: candidatas, tentativas e decisões têm identidade estável e ficam registradas, inclusive rejeições e falhas.

## Limitações atuais

- Decomposição, combinação fixa e estágios múltiplos só entram na busca quando pedidos em `compare_models(include=...)`; a CV aninhada funciona em uma série ou em painel, preservando origens e configurações por série; a seleção dinâmica (`dynamic`) só existe na variante B.
- `compare_models` com o orçamento padrão (`small`), o espaço completo e 7 origens por camada leva alguns minutos numa série de ~700 pontos, sobretudo por causa do ARIMA com ordem automática; use `budget="smoke"`, `fold=3` ou `base_models=[...]` para explorar.
- A regra de aceite (ganho mediano ≥ 5 %, vitórias ≥ 60 %, variabilidade ≥ 0,05, com 7 origens por camada quando a série comporta) foi calibrada em controles sintéticos: aceitou estruturas espúrias em 2–4 % das séries de ruído branco e ganhos reais em 54–56 % das séries com resíduo previsível. Não é controle de erro estatístico; em séries curtas, com menos origens, a taxa de falso aceite é maior (veja o guia de seleção).

## Créditos

Inspirada na ergonomia do [PyCaret](https://pycaret.org); o PyCaret não é dependência.

Desenvolvida por João Ataíde em coautoria com assistentes de IA: **Claude** (Anthropic, pelo Claude Code) e **Codex** (OpenAI). As decisões de arquitetura e de método, a revisão final e a responsabilidade pelo código são do autor; toda mudança passa pelos mesmos testes de contrato, causalidade e vazamento. Veja [Sobre](https://autohybts.readthedocs.io/pt-br/latest/about.html).

## Licença

[MIT](https://github.com/autohybts-lib/AutoHybTS/blob/main/LICENSE) © João Ataíde.
