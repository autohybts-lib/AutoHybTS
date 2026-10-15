"""Progress bars and logging of the experiment steps (reporting only)."""

from __future__ import annotations

import importlib
import logging
import subprocess
import sys
import warnings

import pandas as pd
import pytest

from autohybts import AutoHybridExperiment
from autohybts.datasets import get_data
from autohybts.exceptions import ConfigurationError, UnsupportedCapabilityError
from autohybts.progress import NullProgress, TqdmProgress, make_progress

SETTINGS = {
    "fh": 5,
    "fold": 2,
    "budget": "smoke",
    "base_models": ["naive", "linear"],
    "correctors": ["linear"],
}


def test_make_progress_chooses_the_implementation() -> None:
    assert isinstance(make_progress(False), NullProgress)
    assert isinstance(make_progress(None), NullProgress)
    pytest.importorskip("tqdm")
    assert isinstance(make_progress(True), TqdmProgress)
    with pytest.raises(ConfigurationError, match="progress"):
        make_progress("sim")


def test_missing_tqdm_gives_an_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    real = importlib.import_module

    def without_tqdm(name: str, *args: object) -> object:
        if name.startswith("tqdm"):
            raise ModuleNotFoundError(name)
        return real(name, *args)

    monkeypatch.setattr(importlib, "import_module", without_tqdm)
    with pytest.raises(UnsupportedCapabilityError, match=r"autohybts\[progress\]"):
        TqdmProgress()


def test_bars_without_ipywidgets_do_not_warn() -> None:
    pytest.importorskip("tqdm")

    class NotebookBarWithoutWidgets:
        def __init__(self, **kwargs: object) -> None:
            warnings.warn("IProgress not found.", stacklevel=2)

        def close(self) -> None:
            pass

    progress = TqdmProgress()
    progress._tqdm = NotebookBarWithoutWidgets
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with progress.bar(3, "busca"):
            pass


def test_bars_never_change_the_results(capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("tqdm")
    y = get_data("synthetic_daily", n=160)
    quiet = AutoHybridExperiment(**SETTINGS).fit(y)  # type: ignore[arg-type]
    loud = AutoHybridExperiment(**SETTINGS, progress=True).fit(y)  # type: ignore[arg-type]
    timing = ["fit_seconds", "predict_seconds"]
    pd.testing.assert_frame_equal(
        loud.compare_models().leaderboard.drop(columns=timing),
        quiet.compare_models().leaderboard.drop(columns=timing),
    )
    errors = capsys.readouterr().err
    assert "busca S0 bases" in errors
    assert "seleção V_select" in errors


def test_quiet_experiments_print_no_bars(capsys: pytest.CaptureFixture[str]) -> None:
    AutoHybridExperiment(**SETTINGS).fit(get_data("synthetic_daily", n=160)).compare_models()  # type: ignore[arg-type]
    captured = capsys.readouterr()
    assert "busca" not in captured.err + captured.out


def test_events_go_to_the_autohybts_logger(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="autohybts"):
        exp = AutoHybridExperiment(**SETTINGS).fit(get_data("synthetic_daily", n=160))  # type: ignore[arg-type]
        exp.compare_models()
    messages = [r.getMessage() for r in caplog.records if r.name == "autohybts"]
    assert any(m.startswith("experiment.fitted:") for m in messages)
    assert any(m.startswith("search.finished:") for m in messages)
    assert any(m.startswith("selection.finished:") for m in messages)


def test_importing_the_package_does_not_load_tqdm() -> None:
    code = "import sys, autohybts; print('tqdm' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False"
