"""AutoHybTS: automated search and guarded selection of standalone and hybrid forecasters."""

__version__ = "1.0.0rc2"

from autohybts.api import AutoHybridForecaster
from autohybts.data.panel import PanelFrame
from autohybts.experiment import AutoHybridExperiment
from autohybts.forecasting.finalized import FinalizedForecaster, PanelForecaster
from autohybts.persistence.model_store import load_model, read_manifest, save_model

__all__ = [
    "AutoHybridExperiment",
    "AutoHybridForecaster",
    "FinalizedForecaster",
    "PanelForecaster",
    "PanelFrame",
    "__version__",
    "load_model",
    "read_manifest",
    "save_model",
]
