"""Models built outside AutoHybTS: PyTorch, TensorFlow/Keras, any regressor, plain functions.

Wrap the model, register it on the experiment and it is used by name everywhere:

.. code-block:: python

   from autohybts.models import torch_model
   from autohybts.models.torch_zoo import LSTMNet

   exp.register_model("lstm", torch_model(LSTMNet, hidden=32, lags=28),
                      family="deep", requires={"torch": ">=2.2"}, version="1")
   exp.compare_models()            # with base_models=[..., "lstm"] or correctors=[...]

Nothing here imports PyTorch or TensorFlow; they load when a model is fitted.
"""

from autohybts.models.custom import CustomForecaster, RegressorForecaster
from autohybts.models.keras_models import KerasCorrector, KerasForecaster, keras_model
from autohybts.models.lagged import LaggedModelForecaster, WindowCorrector
from autohybts.models.requirements import (
    check_specifier,
    framework_versions,
    installed_version,
    require,
)
from autohybts.models.torch_models import TorchCorrector, TorchForecaster, torch_model

__all__ = [
    "CustomForecaster",
    "KerasCorrector",
    "KerasForecaster",
    "LaggedModelForecaster",
    "RegressorForecaster",
    "TorchCorrector",
    "TorchForecaster",
    "WindowCorrector",
    "check_specifier",
    "framework_versions",
    "installed_version",
    "keras_model",
    "require",
    "torch_model",
]
