"""Exercise the installed wheel without optional extras, outside an editable install."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import autohybts

HEAVY = {"sklearn", "statsmodels", "torch", "tensorflow", "matplotlib", "optuna", "rdata"}


def main() -> None:
    assert not HEAVY.intersection(name.split(".")[0] for name in sys.modules)
    package = Path(autohybts.__file__).resolve()
    assert "site-packages" in package.parts, f"Expected an installed wheel, got {package}"

    from autohybts import AutoHybridExperiment, load_model, save_model
    from autohybts.datasets import get_data

    for variant in ("A", "B"):
        horizon = 3 if variant == "A" else 1
        exp = AutoHybridExperiment(
            fh=horizon,
            test_size=3,
            fold=2,
            variant=variant,
            budget="smoke",
            base_models=["naive", "linear"],
            correctors=["linear"],
        ).fit(get_data("synthetic_daily"))
        result = exp.compare_models()
        model = exp.finalize_model(result.best)
        predictions = exp.predict_model(model).predictions
        assert len(predictions) == horizon
        assert not exp.evaluate_external(model).metrics.empty
        with tempfile.TemporaryDirectory() as directory:
            key = b"wheel-smoke-signing-key-32-bytes!!"
            save_model(model, directory, signing_key=key)
            restored = load_model(directory, trusted=True, signing_key=key)
            assert exp.predict_model(restored).predictions.equals(predictions)
    print(f"Wheel {autohybts.__version__}: import, A/B forecasting and persistence OK")


if __name__ == "__main__":
    main()
