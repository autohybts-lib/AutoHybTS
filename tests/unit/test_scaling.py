"""Per-column standardization (``scale="columns"``) of lag models and correctors."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from autohybts.adapters.sklearn_regression import LagRegressionForecaster
from autohybts.exceptions import ConfigurationError
from autohybts.residuals.correctors import SklearnCorrector
from autohybts.residuals.generation import local_context


def series(n: int = 200) -> np.ndarray:
    rng = np.random.default_rng(4)
    return 10 + np.sin(np.arange(n) / 2) * 3 + rng.normal(0, 0.4, n)


@pytest.mark.parametrize("build", [LagRegressionForecaster, SklearnCorrector])
def test_scale_accepts_only_bool_or_columns(build) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ConfigurationError, match="columns"):
        build(scale="rows")


def test_linear_models_do_not_depend_on_the_scaling() -> None:
    y = series()
    forecasts = [
        LagRegressionForecaster("linear", 7, scale=scale).fit(y).predict(10, local_context(y, 200))
        for scale in (False, True, "columns")
    ]
    np.testing.assert_allclose(forecasts[0], forecasts[1], atol=1e-8)
    np.testing.assert_allclose(forecasts[0], forecasts[2], atol=1e-8)


def test_columns_scaling_is_standard_scaler_on_x_and_y() -> None:
    rng = np.random.default_rng(1)
    windows = rng.normal(size=(60, 4)) * [1.0, 2.0, 3.0, 4.0]
    targets = rng.normal(size=60) * 5 + 2
    params = {"n_neighbors": 3, "weights": "distance"}
    corrector = SklearnCorrector("knn", scale="columns", estimator_params=params)
    corrector.fit(windows, targets)
    x_scaler, y_scaler = StandardScaler().fit(windows), StandardScaler().fit(targets[:, None])
    np.testing.assert_allclose(corrector.x_center_, x_scaler.mean_)
    np.testing.assert_allclose(corrector.x_spread_, x_scaler.scale_)
    assert corrector.center_ == pytest.approx(y_scaler.mean_[0])
    assert corrector.spread_ == pytest.approx(y_scaler.scale_[0])
    # Distance-based estimators see a different geometry than with one shared scale.
    shared = SklearnCorrector("knn", scale=True, estimator_params=params).fit(windows, targets)
    query = windows[:5] + 0.3
    assert not np.allclose(
        [corrector.predict_next(row) for row in query], [shared.predict_next(row) for row in query]
    )


def test_minmax_is_minmax_scaler_to_the_last_bit() -> None:
    from sklearn.preprocessing import MinMaxScaler

    from autohybts.adapters.sklearn_regression import column_scale, scale_values, unscale_values

    rng = np.random.default_rng(2)
    windows = rng.normal(size=(40, 3)) * [1.0, 10.0, 0.1] + 5
    center, spread = column_scale(windows, "minmax")
    scaler = MinMaxScaler().fit(windows)
    np.testing.assert_array_equal(
        scale_values(windows, center, spread, "minmax"), scaler.transform(windows)
    )
    np.testing.assert_allclose(
        unscale_values(scaler.transform(windows), center, spread, "minmax"), windows, atol=1e-12
    )


def test_feature_order_reverses_the_columns_the_estimator_sees() -> None:
    rng = np.random.default_rng(3)
    windows = rng.normal(size=(50, 4))
    targets = windows @ [0.1, 0.2, 0.3, 0.4] + rng.normal(0, 0.01, 50)
    params = {"hidden_layer_sizes": (4,), "solver": "lbfgs", "max_iter": 500}
    lag = SklearnCorrector("mlp", feature_order="lag", estimator_params=params).fit(
        windows, targets
    )
    flipped = SklearnCorrector("mlp", estimator_params=params).fit(windows[:, ::-1], targets)
    for row in windows[:5]:
        assert lag.predict_next(row) == pytest.approx(flipped.predict_next(row[::-1]))
    linear = [
        SklearnCorrector("linear", feature_order=o).fit(windows, targets)
        for o in ("chronological", "lag")
    ]
    assert linear[0].predict_next(windows[0]) == pytest.approx(linear[1].predict_next(windows[0]))
    with pytest.raises(ConfigurationError, match="feature_order"):
        SklearnCorrector(feature_order="random")
