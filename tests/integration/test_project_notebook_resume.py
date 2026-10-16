"""The project tutorial resumes an interrupted run in a new interpreter, without downloads."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
WORKER = """
import hashlib
import json
import sys
import time
import warnings
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from autohybts import AutoHybridExperiment, ExperimentStore, load_model, read_manifest, save_model
from autohybts.benchmarks import synthetic_cases
from autohybts.domain.identity import stable_hash
from autohybts.exceptions import ArtifactCompatibilityError, ConfigurationError
from autohybts.persistence.manifest import environment, json_safe, source_fingerprint

notebook = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
mode = sys.argv[2]
cases = list(synthetic_cases(2, n=180, horizon=3))
H = 3
SETTINGS = dict(fh=H, test_size=H, fold=2, session_id=42, budget="smoke",
                seasonal_period=7, base_models=["naive", "linear"], correctors=["linear"])
if mode == "changed-config":
    SETTINGS["session_id"] = 43
if mode == "changed-data":
    cases[0].values.iloc[0] += 1
experiments = {case.series_id: AutoHybridExperiment(**SETTINGS).fit(case.values) for case in cases}
workflow = next("".join(cell["source"]) for cell in notebook["cells"]
                if cell["cell_type"] == "code" and "def run_project():" in "".join(cell["source"]))
exec(workflow)
original = AutoHybridExperiment.compare_models
called = []
def guarded(self, *args, **kwargs):
    if mode != "interrupt" and self is experiments[cases[0].series_id]:
        raise AssertionError("A completed case was searched again")
    if mode in {"reuse", "changed-config", "changed-data", "corrupt"}:
        raise AssertionError("Resume must not train in this mode")
    called.append(self)
    if mode == "interrupt" and len(called) == 2:
        raise KeyboardInterrupt
    return original(self, *args, **kwargs)
AutoHybridExperiment.compare_models = guarded
if mode == "interrupt":
    try:
        run_project()
    except KeyboardInterrupt:
        assert len(store.runs()) == 1
    else:
        raise AssertionError("Interruption not reached")
elif mode == "continue":
    assert run_project() == [cases[1].series_id]
    assert len(called) == 1
elif mode in {"changed-config", "changed-data", "corrupt"}:
    try:
        run_project()
    except (ConfigurationError, ArtifactCompatibilityError):
        assert not called
    else:
        raise AssertionError("Incompatible or corrupt state was reused")
else:
    before = store.query("SELECT run_id, logged_utc FROM runs ORDER BY run_id")
    assert run_project() == []
    pd.testing.assert_frame_equal(
        before, store.query("SELECT run_id, logged_utc FROM runs ORDER BY run_id")
    )
    assert len(store.query("SELECT * FROM external")) > 0
    assert len(store.query("SELECT * FROM predictions")) > 0
print(mode, "OK")
"""


@pytest.mark.parametrize("language", ["pt", "en"])
def test_project_resumes_in_new_process_and_refuses_incompatible_state(
    tmp_path: Path, language: str
) -> None:
    notebook = (
        ROOT / "notebooks" / ("en" if language == "en" else "") / "22_m4_development_pilot.ipynb"
    )
    worker = tmp_path / "worker.py"
    worker.write_text(WORKER, encoding="utf-8")
    env = dict(os.environ)
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        env[key] = "1"
    for mode in ("interrupt", "continue", "reuse", "changed-config", "changed-data"):
        result = subprocess.run(
            [sys.executable, str(worker), str(notebook), mode],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert f"{mode} OK" in result.stdout
    payload = next((tmp_path / "artifacts/m4_project/models").rglob("model.pkl"))
    payload.write_bytes(payload.read_bytes() + b"corrupt")
    result = subprocess.run(
        [sys.executable, str(worker), str(notebook), "corrupt"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "corrupt OK" in result.stdout
