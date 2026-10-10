from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autohybts.exceptions import ConfigurationError
from autohybts.forecasting.forecast_frame import ForecastFrame


def rows(horizon: int = 3, origin: str = "2024-01-10", freq: str = "D") -> pd.DataFrame:
    stamps = pd.date_range(origin, periods=horizon + 1, freq=freq)[1:]
    return pd.DataFrame(
        {
            "series_id": "s",
            "candidate_id": "c",
            "fold_id": "select-0",
            "origin": pd.Timestamp(origin),
            "timestamp": stamps,
            "horizon": np.arange(1, horizon + 1),
            "y_pred": np.ones(horizon),
            "y_true": np.nan,
        }
    )


def test_valid_frame() -> None:
    frame = ForecastFrame(rows(), "D")
    assert len(frame) == 3
    data = frame.to_pandas()
    data.loc[0, "y_pred"] = 99.0
    assert frame.to_pandas().loc[0, "y_pred"] == 1.0  # copies, not views


def test_monthly_steps() -> None:
    assert len(ForecastFrame(rows(origin="2024-01-01", freq="MS"), "MS")) == 3


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d.drop(columns="y_pred"), "colunas"),
        (lambda d: d.assign(horizon=[1, 1, 3]), "Passos"),
        (lambda d: d.assign(horizon=[2, 3, 4]), "Passos"),
        (lambda d: d.assign(timestamp=d["timestamp"] + pd.Timedelta(days=1)), "incoerentes"),
        (lambda d: d.assign(y_pred=[1.0, np.nan, 1.0]), "não finitas"),
    ],
)
def test_invalid_frames(mutate: object, message: str) -> None:
    with pytest.raises(ConfigurationError, match=message):
        ForecastFrame(mutate(rows()), "D")


def test_concat() -> None:
    joined = ForecastFrame.concat(
        [ForecastFrame(rows(), "D"), ForecastFrame(rows(origin="2024-02-01"), "D")]
    )
    assert len(joined) == 6
    with pytest.raises(ConfigurationError, match="Frequências"):
        ForecastFrame.concat(
            [ForecastFrame(rows(), "D"), ForecastFrame(rows(origin="2024-01-01", freq="MS"), "MS")]
        )
    with pytest.raises(ConfigurationError):
        ForecastFrame.concat([])
