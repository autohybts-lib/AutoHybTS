"""Capture the legacy residual hybrids as parity fixtures (F1e; run by hand, not by pytest).

Usage::

    python tests/regression/capture_legacy_hybrids.py <path to projeto_st_replicacao>

The four hybrids of the replication with a KNN residual corrector (linear, tree, SVR and
KNN bases) are rebuilt from the legacy sources: only the needed classes and functions are
read by name with ``ast`` and run on a deterministic series; their *forecasts* (variant A,
14 steps, and variant B, 14 one-step forecasts) are written to
``tests/regression/fixtures/legacy_hybrids.json``. No legacy code is copied into this
repository and the legacy package is never imported.

The three statistical-ANN hybrids (AutoETS, AutoARIMA, AutoSARIMA + MLP residual) use
StatsForecast, whose releases require pandas<3: they are captured only when it is
installed (run the script in an environment with the ``nixtla-stats`` extra, e.g. a second
venv with pandas 2.x; the KNN hybrids are captured in both).
"""

from __future__ import annotations
import __future__

import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import re
import sys
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Self

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

FIXTURE = Path(__file__).parent / "fixtures" / "legacy_hybrids.json"
SOURCES = {
    "src/split.py": {"lag_windows", "observed_lag_rows", "normalize_lags", "_as_series"},
    "src/models/base.py": {"Forecaster"},
    "src/models/lags.py": {
        "LagForecaster",
        "_final_estimator",
        "_input_transform",
        "_spectral_radius",
        "_stability",
        "LinearRegressionForecaster",
        "SVMForecaster",
        "KNNForecaster",
        "DecisionTreeForecaster",
        "MLPForecaster",
    },
    "src/models/residual.py": {
        "KNNResidualForecaster",
        "LinearKNNResidualForecaster",
        "TreeKNNResidualForecaster",
        "SVMKNNResidualForecaster",
        "KNNKNNResidualForecaster",
    },
}
# Settings shared by the four hybrids (the replication's grid values, shorter tail).
COMMON: dict[str, Any] = {
    "lags": 7,
    "residual_lags": 7,
    "n_neighbors": 5,
    "residual_tail": 120,
    "weights": "distance",
    "scale": True,
}
HYBRIDS: dict[str, tuple[str, dict[str, Any]]] = {
    "linear_knn_residual": ("LinearKNNResidualForecaster", {"base_scale": False}),
    "tree_knn_residual": (
        "TreeKNNResidualForecaster",
        {"base_scale": False, "random_state": 42, "min_samples_leaf": 5},
    ),
    "svm_knn_residual": (
        "SVMKNNResidualForecaster",
        {"base_scale": True, "kernel": "rbf", "gamma": "scale", "epsilon": 0.1, "C": 10},
    ),
    "knn_knn_residual": (
        "KNNKNNResidualForecaster",
        {"base_scale": True, "base_n_neighbors": 5, "base_weights": "distance"},
    ),
}
# Statistical-ANN hybrids: the replication's base settings and ANN settings
# (configs/models.yaml), residual lags 7, 5 hidden units, seed 42, shorter tail.
STATISTICAL_SOURCES = {
    "src/models/statistical.py": {
        "ARIMA_TRACE",
        "StatsForecastForecaster",
        "AutoARIMAForecaster",
        "AutoSARIMAForecaster",
        "AutoETSForecaster",
        "_min_root_modulus",
        "_parse_arima_trace",
    },
    "src/models/statistical_residual.py": {
        "StatisticalANNForecaster",
        "ETSANNForecaster",
        "ARIMAANNForecaster",
        "SARIMAANNForecaster",
    },
}
ANN: dict[str, Any] = {
    "residual_tail": 120,
    "residual_lags": 7,
    "hidden_units": 5,
    "seed": 42,
    "ann_params": {
        "activation": "logistic",
        "solver": "lbfgs",
        "max_iter": 2000,
        "alpha": 0.0001,
        "tol": 0.0001,
        "scale": "minmax",
    },
}
STATISTICAL: dict[str, tuple[str, dict[str, Any]]] = {
    "ets_ann_residual": ("ETSANNForecaster", {"season_length": 7, "model": "ZZZ"}),
    "arima_ann_residual": (
        "ARIMAANNForecaster",
        {
            "seasonal": False,
            "season_length": 1,
            "max_p": 5,
            "max_q": 5,
            "max_d": 2,
            "allowdrift": True,
            "allowmean": True,
            "ic": "aicc",
        },
    ),
    "sarima_ann_residual": (
        "SARIMAANNForecaster",
        {
            "seasonal": True,
            "season_length": 7,
            "max_d": 2,
            "max_P": 2,
            "max_Q": 2,
            "max_D": 1,
            "allowdrift": True,
            "allowmean": True,
            "ic": "aicc",
        },
    ),
}
N, TRAIN, HORIZON = 400, 386, 14


def load(path: Path, names: set[str], namespace: dict[str, Any]) -> None:
    """Run the named top-level functions and classes of ``path`` in ``namespace``."""

    tree = ast.parse(path.read_text(encoding="utf-8"))
    wanted = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef | ast.ClassDef) and node.name in names)
        or (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id in names for t in node.targets)
        )
    ]
    found = {
        node.name if not isinstance(node, ast.Assign) else node.targets[0].id  # type: ignore[attr-defined]
        for node in wanted
    }
    missing = names - found
    if missing:
        raise SystemExit(f"{path}: definitions not found: {sorted(missing)}")
    module = ast.Module(body=wanted, type_ignores=[])
    flags = __future__.annotations.compiler_flag  # the legacy files use postponed annotations
    exec(compile(module, str(path), "exec", flags=flags, dont_inherit=True), namespace)


def series() -> np.ndarray:
    rng = np.random.default_rng(11)
    noise = np.zeros(N)
    for i in range(1, N):
        noise[i] = 0.7 * noise[i - 1] + rng.normal()
    t = np.arange(N)
    return 50 + 0.02 * t + 3 * np.sin(2 * np.pi * t / 7) + noise


def main(root: Path) -> None:
    namespace: dict[str, Any] = {
        "np": np,
        "pd": pd,
        "warnings": warnings,
        "ABC": ABC,
        "abstractmethod": abstractmethod,
        "Any": Any,
        "Self": Self,
        "Path": Path,
        "RegressorMixin": RegressorMixin,
        "TransformedTargetRegressor": TransformedTargetRegressor,
        "ConvergenceWarning": ConvergenceWarning,
        "LinearRegression": LinearRegression,
        "KNeighborsRegressor": KNeighborsRegressor,
        "make_pipeline": make_pipeline,
        "MinMaxScaler": MinMaxScaler,
        "StandardScaler": StandardScaler,
        "SVR": SVR,
        "DecisionTreeRegressor": DecisionTreeRegressor,
        "MLPRegressor": MLPRegressor,
        "register": lambda name: lambda cls: cls,  # the legacy registry is not needed
    }
    sources = dict(SOURCES)
    statistical = importlib.util.find_spec("statsforecast") is not None
    if statistical:
        from statsforecast.models import AutoARIMA, AutoETS

        namespace.update(
            AutoARIMA=AutoARIMA,
            AutoETS=AutoETS,
            contextlib=contextlib,
            io=io,
            re=re,
            arch_model=None,  # GARCH is not loaded
        )
        sources.update(STATISTICAL_SOURCES)
    for relative, names in sources.items():
        load(root / relative, names, namespace)
    y = series()
    train, test = y[:TRAIN], y[TRAIN:]
    hybrids = {}
    for name, (class_name, params) in HYBRIDS.items():
        model = namespace[class_name](**COMMON, **params).fit(train)
        hybrids[name] = {
            "params": {**COMMON, **params},
            "variant_a": model.forecast(HORIZON).tolist(),
            "variant_b": model.one_step(test).tolist(),
        }
    if statistical:
        for name, (class_name, base_params) in STATISTICAL.items():
            with contextlib.redirect_stdout(io.StringIO()):  # AutoARIMA prints its trace
                model = namespace[class_name](base_params=base_params, **ANN).fit(train)
            hybrids[name] = {
                "params": {"base_params": base_params, **ANN},
                "variant_a": model.forecast(HORIZON).tolist(),
                "variant_b": model.one_step(test).tolist(),
            }
    else:
        print("statsforecast not installed: statistical-ANN hybrids not captured")
    fixture = {
        "source": {
            "project": "projeto_st_replicacao",
            "files": {
                relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
                for relative in sources
            },
        },
        "y": y.tolist(),
        "train": TRAIN,
        "horizon": HORIZON,
        "hybrids": hybrids,
    }
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(fixture, indent=1), encoding="utf-8")
    print(f"wrote {FIXTURE}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
