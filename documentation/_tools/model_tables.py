"""Generate the model and search-space tables of the user guide from the code.

Called by ``conf.py`` at every build, so the tables cannot drift from the registry, the
default parameters, the built-in ``tune_model`` spaces and the budget presets. The files
go to ``user_guide/_generated/`` of each language tree (ignored by git and excluded as
documents; pages pull them in with ``.. include::``). ``language`` is ``"pt"`` (the
default, ``documentation/``) or ``"en"`` (``documentation/en/``).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from autohybts.adapters.sklearn_regression import DEFAULT_ESTIMATOR_PARAMS, ESTIMATORS
from autohybts.registry import default_registry
from autohybts.registry.registry import FAMILIES
from autohybts.search.budgets import PRESETS
from autohybts.search.spaces import DEFAULT_BASES, SearchSpace
from autohybts.search.tuning import LAG_SPACES, TUNING_SPACES, Distribution

SEASON = 7  # the tables show the defaults of a daily series

TEXT: dict[str, dict[str, str]] = {
    "pt": {
        "self_tuned": "— (escolhe sozinho ou não tem o que ajustar)",
        "family": "Família",
        "both": "A e B",
        "extra": "extra",
        "core": "núcleo",
        "name": "Nome",
        "defaults": "Parâmetros padrão",
        "in_search": "Na busca",
        "space": "Espaço do ``tune_model``",
        "variants": "Variantes; instalação",
        "sequence": "sequência",
        "window": "janela",
        "kind": "Tipo",
        "install": "Instalação",
        "total": "Total de candidatas",
        "per_family": "Por família",
        "per_stage": "Por etapa",
        "refinements": "Refinamentos por eixo",
        "bases": "Bases (S0)",
        "correctors": "Corretores (S2)",
        "windows": "Janelas de resíduos",
        "alphas": "Pesos da correção (α)",
        "sources": "Fonte de resíduos (S2)",
        "refine_sources": "Fontes no refinamento (S3)",
        "axis": "Eixo",
        "default": "Padrão",
    },
    "en": {
        "self_tuned": "— (chooses by itself or has nothing to tune)",
        "family": "Family",
        "both": "A and B",
        "extra": "extra",
        "core": "core",
        "name": "Name",
        "defaults": "Default parameters",
        "in_search": "In the search",
        "space": "``tune_model`` space",
        "variants": "Variants; installation",
        "sequence": "sequence",
        "window": "window",
        "kind": "Kind",
        "install": "Installation",
        "total": "Total candidates",
        "per_family": "Per family",
        "per_stage": "Per stage",
        "refinements": "Refinements per axis",
        "bases": "Bases (S0)",
        "correctors": "Correctors (S2)",
        "windows": "Residual windows",
        "alphas": "Correction weights (α)",
        "sources": "Residual source (S2)",
        "refine_sources": "Sources in refinement (S3)",
        "axis": "Axis",
        "default": "Default",
    },
}


def _value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, tuple | list):
        return "(" + ", ".join(_value(v) for v in value) + ("," if len(value) == 1 else "") + ")"
    return repr(value) if isinstance(value, str) else str(value)


def _params(params: Mapping[str, Any], name: str = "") -> str:
    # ``estimator`` and a ``model`` equal to the registry name are not settings.
    shown = {
        k: v
        for k, v in params.items()
        if v is not None and k != "estimator" and not (k == "model" and v == name)
    }
    if not shown:
        return "—"
    return ", ".join(f"``{k}={_value(v)}``" for k, v in shown.items())


def _distribution(dist: Distribution) -> str:
    if dist.kind == "categorical":
        return "{" + ", ".join(_value(c) for c in dist.choices) + "}"
    low, high = (
        _value(dist.low if dist.kind != "int" else int(dist.low)),
        _value(dist.high if dist.kind != "int" else int(dist.high)),
    )
    scale = " (log)" if dist.kind == "log_float" else ""
    return f"[{low}, {high}]{scale}"


def _space(name: str, text: Mapping[str, str]) -> str:
    if name not in TUNING_SPACES:
        return text["self_tuned"]
    parts = []
    if name in LAG_SPACES:
        parts.append(f"``lags`` [2, {min(60, max(14, 4 * SEASON))}]")
    parts += [f"``{k}`` {_distribution(d)}" for k, d in TUNING_SPACES[name].items()]
    return "; ".join(parts)


def _extra(requires_extra: str | None, text: Mapping[str, str]) -> str:
    return f"{text['extra']} ``{requires_extra}``" if requires_extra else text["core"]


def _table(header: Iterable[str], rows: list[list[str]], widths: str) -> list[str]:
    lines = [".. list-table::", "   :header-rows: 1", f"   :widths: {widths}", ""]
    for row in [list(header), *rows]:
        lines.append(f"   * - {row[0]}")
        lines += [f"     - {cell}" for cell in row[1:]]
    return [*lines, ""]


def base_tables(language: str = "pt") -> str:
    text = TEXT[language]
    registry = default_registry()
    space = SearchSpace(seasonal_period=SEASON)
    by_family: dict[str, list[str]] = {}
    for name in registry.names("base"):
        by_family.setdefault(registry.entry(name).family, []).append(name)
    order = [f for f in FAMILIES if f in by_family] + sorted(set(by_family) - set(FAMILIES))
    out: list[str] = []
    for family in order:
        heading = f"{text['family']} ``{family}``"
        out += [heading, "~" * len(heading), ""]
        rows = []
        for name in by_family[family]:
            entry = registry.entry(name)
            params = registry.make_base(name).get_params()
            if name in ESTIMATORS:
                params = {**params, "estimator_params": DEFAULT_ESTIMATOR_PARAMS.get(name)}
            caps = entry.capabilities
            variants = (
                text["both"]
                if caps.multi_step and caps.online_state_update
                else ("A" if caps.multi_step else "B")
            )
            rows.append(
                [
                    f"``{name}``" + (" ★" if name in DEFAULT_BASES else ""),
                    _params(params, name),
                    _params(space.default_params(name)),
                    _space(name, text),
                    f"{variants}; {_extra(caps.requires_extra, text)}",
                ]
            )
        out += _table(
            [text["name"], text["defaults"], text["in_search"], text["space"], text["variants"]],
            rows,
            "16 30 14 28 12",
        )
    return "\n".join(out)


def corrector_table(language: str = "pt") -> str:
    text = TEXT[language]
    registry = default_registry()
    defaults = SearchSpace().correctors
    rows = []
    for name in registry.names("corrector"):
        caps = registry.entry(name, "corrector").capabilities
        params = registry.make_corrector(name).get_params()
        if name in ESTIMATORS:
            params = {**params, "estimator_params": DEFAULT_ESTIMATOR_PARAMS.get(name)}
        kind = text["sequence"] if caps.multi_step else text["window"]
        rows.append(
            [
                f"``{name}``" + (" ★" if name in defaults else ""),
                kind,
                _params(params, name),
                _extra(caps.requires_extra, text),
            ]
        )
    header = [text["name"], text["kind"], text["defaults"], text["install"]]
    return "\n".join(_table(header, rows, "22 10 52 16"))


def budget_table(language: str = "pt") -> str:
    text = TEXT[language]
    names = (
        "max_trials_total",
        "max_trials_per_family",
        "max_candidates_per_stage",
        "max_refinements",
    )
    labels = (text["total"], text["per_family"], text["per_stage"], text["refinements"])
    rows = [
        [f"``{preset}``", *[str(PRESETS[preset]["guard"][n]) for n in names]]
        for preset in ("smoke", "small", "medium", "large")
    ]
    return "\n".join(_table(["Preset", *labels], rows, "16 21 21 21 21"))


def search_defaults_table(language: str = "pt") -> str:
    text = TEXT[language]
    space = SearchSpace(seasonal_period=SEASON)
    rows = [
        [text["bases"], ", ".join(f"``{b}``" for b in space.base_models)],
        [text["correctors"], ", ".join(f"``{c}``" for c in space.correctors)],
        [text["windows"], ", ".join(str(w) for w in space.residual_windows)],
        [text["alphas"], ", ".join(_value(a) for a in space.alphas)],
        [text["sources"], ", ".join(f"``{s}``" for s in space.residual_sources)],
        [text["refine_sources"], ", ".join(f"``{s}``" for s in space.refine_sources)],
    ]
    return "\n".join(_table([text["axis"], text["default"]], rows, "30 70"))


def write(target: Path, language: str = "pt") -> None:
    target.mkdir(parents=True, exist_ok=True)
    files = {
        "base_models.rst": base_tables(language),
        "correctors.rst": corrector_table(language),
        "budgets.rst": budget_table(language),
        "search_defaults.rst": search_defaults_table(language),
    }
    for name, content in files.items():
        path = target / name
        if not path.exists() or path.read_text(encoding="utf-8") != content:
            path.write_text(content, encoding="utf-8")
