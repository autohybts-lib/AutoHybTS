"""T20 (partial): the package imports without optional extras or heavy modules."""

from __future__ import annotations

import json
import subprocess
import sys

import autohybts

# Optional or heavy dependencies that `import autohybts` must never pull in.
HEAVY_MODULES = ("statsmodels", "torch", "optuna", "matplotlib", "plotly", "yaml", "sklearn")


def _modules_after_import(statement: str) -> set[str]:
    code = f"import json, sys; {statement}; print(json.dumps(sorted(sys.modules)))"
    output = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout
    return set(json.loads(output))


def test_version_is_exposed() -> None:
    assert isinstance(autohybts.__version__, str)
    assert autohybts.__version__.count(".") == 2


def test_import_does_not_load_heavy_modules() -> None:
    loaded = _modules_after_import("import autohybts")
    assert not {name for name in loaded if name.split(".")[0] in HEAVY_MODULES}


def test_core_submodules_do_not_load_heavy_modules() -> None:
    loaded = _modules_after_import(
        "import autohybts.exceptions, autohybts.types, autohybts.domain.identity"
    )
    assert not {name for name in loaded if name.split(".")[0] in HEAVY_MODULES}


def test_model_wrappers_do_not_load_frameworks_or_packaging() -> None:
    # packaging is an optional extra (ADR-0006): only checking a requirement imports it.
    loaded = _modules_after_import("import autohybts.models, autohybts.experiment")
    roots = {name.split(".")[0] for name in loaded}
    assert not roots & {"torch", "tensorflow", "keras", "packaging"}
