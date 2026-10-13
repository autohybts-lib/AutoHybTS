"""Create the skeleton of a tutorial notebook in the standard lesson model.

Usage (from the repository root)::

    python notebooks/new_notebook.py 16_temporal_bagging "Bagging temporal" \\
        --guide user_guide/ensembles --extras plot \\
        --requires autohybts.AutoHybridExperiment.compare_models

The notebook gets the sections of the lesson model (documentation/contributing.rst) and
the metadata schema checked by ``tests/contracts/test_notebook_metadata.py``. Fill the
cells, run it from its folder (outputs are stored) and write the reading last, from the
numbers of that run. ``--lang en`` writes the English skeleton to ``notebooks/en/``;
every tutorial exists in both languages.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat

HERE = Path(__file__).parent
SETUP = '''import warnings

import matplotlib.pyplot as plt
import pandas as pd

import autohybts
from autohybts import AutoHybridExperiment
from autohybts.datasets import get_data

warnings.filterwarnings("ignore")
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)
SEED = 42


def clean(ax, grid_axis="both"):
    """Recessive grid and axes; text in neutral ink."""
    ax.grid(axis=grid_axis, color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors="#52514e")


print("AutoHybTS", autohybts.__version__)'''
SECTIONS_PT = (
    (
        "## 1. Dados e protocolo",
        "Origem, frequência, `fh`, `test_size`, `fold`, sementes e orçamento. A busca nunca "
        "toca o teste selado.",
        'y = get_data("synthetic_daily", n=500)\n'
        "exp = AutoHybridExperiment(fh=14, fold=3, session_id=SEED).fit(y)\n"
        "exp.splits",
    ),
    (
        "## 2. Exploração causal",
        "Série, sazonalidade e escalas, só com o que está disponível na origem.",
        'exp.plot_data("series")',
    ),
    (
        "## 3. Controles e modelos propostos",
        "Mesma métrica, mesmo horizonte, mesmas origens: o melhor modelo isolado e uma "
        "combinação simples são os controles.",
        "",
    ),
    (
        "## 4. Busca e seleção",
        "API pública, `V_search` para a busca e `V_select` para a decisão; custo em "
        "candidatas, unidades e ajustes.",
        "",
    ),
    (
        "## 5. Resultados e gráficos",
        "Previsões contra observações, erros, pesos, ganhos e decisões (aceites e recusas).",
        "",
    ),
    (
        "## 6. Diagnóstico e limites",
        "Riscos de vazamento, cobertura, falhas e custo.",
        "",
    ),
    (
        "## 7. Avaliação externa",
        "Uma única leitura do teste selado, depois de congelar a escolha.",
        "final = exp.finalize_model(result.best)\nexp.evaluate_external(final).metrics.round(3)",
    ),
)
SECTIONS_EN = (
    (
        "## 1. Data and protocol",
        "Source, frequency, `fh`, `test_size`, `fold`, seeds and budget. The search never "
        "touches the sealed test.",
        SECTIONS_PT[0][2],
    ),
    (
        "## 2. Causal exploration",
        "Series, seasonality and scales, using only what is available at the origin.",
        SECTIONS_PT[1][2],
    ),
    (
        "## 3. Controls and proposed models",
        "Same metric, same horizon, same origins: the best standalone model and a simple "
        "combination are the controls.",
        "",
    ),
    (
        "## 4. Search and selection",
        "Public API, `V_search` for the search and `V_select` for the decision; cost in "
        "candidates, units and fits.",
        "",
    ),
    (
        "## 5. Results and plots",
        "Forecasts against observations, errors, weights, gains and decisions (acceptances "
        "and rejections).",
        "",
    ),
    (
        "## 6. Diagnostics and limits",
        "Leakage risks, coverage, failures and cost.",
        "",
    ),
    (
        "## 7. External evaluation",
        "A single reading of the sealed test, after freezing the choice.",
        SECTIONS_PT[6][2],
    ),
)
TEXT = {
    "pt": {
        "intro": "**Objetivo e requisitos.** Tema, referência científica, fase do plano, "
        "variantes A/B suportadas, dependências e o que o leitor sai sabendo.",
        "sections": SECTIONS_PT,
        "reading": "## 8. Leitura\n\n"
        "Escreva depois da execução, com os números dela: o que venceu, contra quais "
        "controles, com quantas unidades em `V_select`, o custo e os limites. Não generalize "
        "além da série.",
    },
    "en": {
        "intro": "**Goal and requirements.** Topic, scientific reference, plan phase, "
        "supported A/B variants, dependencies and what the reader takes away.",
        "sections": SECTIONS_EN,
        "reading": "## 8. Reading\n\n"
        "Write it after the run, with its numbers: what won, against which controls, with "
        "how many units in `V_select`, the cost and the limits. Do not generalize beyond "
        "the series.",
    },
}


def skeleton(
    title: str,
    requires: list[str],
    guide: list[str],
    extras: list[str],
    heavy: bool,
    lang: str = "pt",
) -> nbformat.NotebookNode:
    text = TEXT[lang]
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3 (ipykernel)",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["autohybts"] = {
        "requires": requires,
        "network": False,
        "heavy": heavy,
        "extras": extras,
        "seeds": [42],
        "guide": guide,
        "template": True,
    }
    intro = f"# {title}\n\n{text['intro']}"
    cells = [nbformat.v4.new_markdown_cell(intro), nbformat.v4.new_code_cell(SETUP)]
    for heading, body, code in text["sections"]:
        cells.append(nbformat.v4.new_markdown_cell(f"{heading}\n\n{body}"))
        if code:
            cells.append(nbformat.v4.new_code_cell(code))
    cells.append(nbformat.v4.new_markdown_cell(text["reading"]))
    nb.cells = cells
    return nb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("name", help="nome do arquivo sem extensão, ex.: 15_temporal_bagging")
    parser.add_argument("title")
    parser.add_argument("--requires", nargs="+", required=True, help="nomes públicos usados")
    parser.add_argument("--guide", nargs="+", required=True, help="páginas do guia")
    parser.add_argument("--extras", nargs="*", default=["plot"])
    parser.add_argument("--heavy", action="store_true")
    parser.add_argument("--lang", choices=sorted(TEXT), default="pt")
    args = parser.parse_args()
    path = (HERE if args.lang == "pt" else HERE / args.lang) / f"{args.name}.ipynb"
    if path.exists():
        raise SystemExit(f"{path.name} já existe")
    notebook = skeleton(args.title, args.requires, args.guide, args.extras, args.heavy, args.lang)
    nbformat.write(notebook, path)
    print(f"criado {path.name}")


if __name__ == "__main__":
    main()
