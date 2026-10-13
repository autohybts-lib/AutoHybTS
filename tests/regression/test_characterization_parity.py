"""Characterization parity with the replication project (capture_legacy_characterization.py).

Same difference order and stationarity decision (ADF rule) and the same STL strength on
the Box-Cox series, on white noise, random walks, seasonal walks and trended series.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("statsmodels")

from tests.regression.capture_legacy_characterization import PERIODS, series

from autohybts.data.characterization import characterize

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "legacy_characterization.json").read_text(
        encoding="utf-8"
    )
)

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


@pytest.mark.parametrize("name", sorted(FIXTURE))
def test_both_rules_match_the_legacy_analysis(name: str) -> None:
    expected = FIXTURE[name]
    record = characterize(series()[name], periods=PERIODS)
    assert record["diff_order"] == expected["diff_order"]
    assert record["stationary"] == expected["stationary"]
    for period in PERIODS:
        legacy = expected["fs"][str(period)]
        if legacy is None:
            assert np.isnan(record[f"fs_{period}"])
        else:
            assert record[f"fs_{period}"] == pytest.approx(legacy, abs=1e-9)
