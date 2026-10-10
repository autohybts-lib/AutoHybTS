"""Capture outputs of the legacy project as parity fixtures (I1; run by hand, not by pytest).

Usage::

    python tests/regression/capture_legacy.py <path to projeto_st_replicacao>

Only the needed functions are read from the legacy sources (by name, with ``ast``) and
run on deterministic inputs; their *outputs* are written to
``tests/regression/fixtures/legacy_composed.json``. No legacy code is copied into this
repository, and the legacy project's heavy dependencies are not imported.
"""

from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kurtosis

FIXTURE = Path(__file__).parent / "fixtures" / "legacy_composed.json"


def load_functions(path: Path, names: set[str], namespace: dict[str, Any]) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    wanted = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in names)
        or (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id in names for t in node.targets)
        )
    ]
    module = ast.Module(body=wanted, type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def series(seed: int, n: int, phi: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = phi * noise[i - 1] + rng.normal()
    return 10 + 0.05 * np.arange(n) + 2 * np.sin(2 * np.pi * np.arange(n) / 7) + noise


def main(root: Path) -> None:
    composed = root / "src" / "models" / "composed.py"
    compose = root / "src" / "compose.py"
    ns = load_functions(
        composed, {"moving_average", "kurtosis_window"}, {"np": np, "kurtosis": kurtosis}
    )
    cases = []
    for seed, n, phi, window, max_window in ((0, 60, 0.5, 3, 365), (1, 120, 0.9, 7, 30)):
        y = series(seed, n, phi)
        cases.append(
            {
                "seed": seed,
                "n": n,
                "phi": phi,
                "y": y.tolist(),
                "window": window,
                "moving_average": ns["moving_average"](y, window).tolist(),
                "max_window": max_window,
                "kurtosis_window": int(ns["kurtosis_window"](y, max_window)),
            }
        )

    rank_ns = load_functions(compose, {"_mean_ranks", "_gain", "BEST"}, {"np": np, "pd": pd})
    rows = []
    rng = np.random.default_rng(7)
    for m4_id in ("D1", "D2", "D3", "D4"):
        for structure, level in (("arima", 1.0), ("arima+mlp", 0.95), ("naive", 1.2)):
            rows.append(
                {"m4_id": m4_id, "structure": structure, "score": level + rng.normal(0, 0.05)}
            )
    best = pd.DataFrame(rows)
    ranks = rank_ns["_mean_ranks"](best, "score")
    gain = rank_ns["_gain"](best, "arima+mlp", "arima", "score")
    fixture = {
        "source": {
            "project": "projeto_st_replicacao",
            "files": {
                str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (composed, compose)
            },
        },
        "decomposition": cases,
        "ranking": {
            "rows": rows,
            "mean_ranks": {k: float(v) for k, v in ranks.items()},
            "gain_arima_mlp_over_arima": gain,
        },
    }
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(fixture, indent=1), encoding="utf-8")
    print(f"wrote {FIXTURE}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
