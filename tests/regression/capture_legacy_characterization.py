"""Capture the legacy series characterization as a parity fixture (run by hand, not by pytest).

Usage::

    python tests/regression/capture_legacy_characterization.py <path to projeto_st_replicacao>

The two rules of the replication project (ADF stationarity with the smallest passing
difference order; STL strength on the Box-Cox series for non-stationary series) are
rebuilt from ``src/analysis.py``: only ``adf_result``, ``kpss_result``,
``stationarity_profile`` and ``_seasonal_profile`` are read by name with ``ast`` and run
on the deterministic series of :func:`series`; the outputs are written to
``tests/regression/fixtures/legacy_characterization.json``. No legacy code is copied into
this repository and the legacy package is never imported.
"""

from __future__ import annotations

import ast
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np

NAMES = ("adf_result", "kpss_result", "stationarity_profile", "_seasonal_profile")
PERIODS = (7, 30)
FIXTURE = Path(__file__).parent / "fixtures" / "legacy_characterization.json"


def series() -> dict[str, np.ndarray]:
    """White noise, random walks, seasonal walks and trend plus a monthly cycle."""

    out = {}
    for seed in range(6):
        rng = np.random.default_rng(seed)
        t = np.arange(400)
        out[f"noise{seed}"] = 10 + rng.normal(size=400)
        out[f"walk{seed}"] = 50 + np.cumsum(rng.normal(size=400))
        seasonal = 6 * np.sin(2 * np.pi * t / 7)
        out[f"seasonal{seed}"] = 50 + np.cumsum(rng.normal(0, 0.3, 400)) + seasonal
        cycle = 2 * np.sin(2 * np.pi * t / 30)
        out[f"trend{seed}"] = 5 + 0.05 * t + rng.normal(size=400) + cycle
    return out


def legacy_functions(root: Path) -> dict[str, Any]:
    source = (root / "src" / "analysis.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in NAMES]
    missing = set(NAMES) - {n.name for n in nodes}
    if missing:
        raise SystemExit(f"Funções não encontradas no legado: {sorted(missing)}")
    from scipy.optimize import minimize_scalar
    from scipy.stats import boxcox, boxcox_llf
    from statsmodels.tools.sm_exceptions import InterpolationWarning
    from statsmodels.tsa.seasonal import STL
    from statsmodels.tsa.stattools import adfuller, kpss

    namespace: dict[str, Any] = {
        "Any": Any,
        "np": np,
        "warnings": warnings,
        "adfuller": adfuller,
        "kpss": kpss,
        "InterpolationWarning": InterpolationWarning,
        "STL": STL,
        "boxcox": boxcox,
        "boxcox_llf": boxcox_llf,
        "minimize_scalar": minimize_scalar,
    }
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(module, "legacy_analysis", "exec"), namespace)
    return namespace


def main(root: Path) -> None:
    legacy = legacy_functions(root)
    adf = {"alpha": 0.05, "autolag": "AIC", "regression": "c"}
    stl = {"robust": True, "boxcox_bounds": [-1, 2]}
    records = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, y in series().items():
            profile = legacy["stationarity_profile"](y, adf, max_diff=2)
            stationary = bool(profile["adf_stationary_d0"])
            strengths = {
                str(p): None if stationary else legacy["_seasonal_profile"](y, p, stl)[0]
                for p in PERIODS
            }
            records[name] = {
                "diff_order": int(profile["diff_order"]),
                "stationary": stationary,
                "fs": strengths,
            }
    FIXTURE.write_text(json.dumps(records, indent=1, sort_keys=True), encoding="utf-8")
    print(f"{len(records)} séries gravadas em {FIXTURE.name}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(Path(sys.argv[1]))
