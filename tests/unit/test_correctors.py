from __future__ import annotations

import warnings

import numpy as np
import pytest

from autohybts.exceptions import CandidateFitError, ConfigurationError, DataValidationError
from autohybts.residuals.correctors import ResidualCorrector, SklearnCorrector
from autohybts.residuals.recursive import recursive_correction, roll_forward


def ar_windows(n: int = 200, window: int = 3) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    r = np.zeros(n)
    for i in range(1, n):
        r[i] = 0.9 * r[i - 1] + rng.normal(0, 0.1)
    X = np.column_stack([r[k : n - window + k] for k in range(window)])
    return X, r[window:]


def test_linear_corrector_learns_an_ar1() -> None:
    X, y = ar_windows()
    corrector = SklearnCorrector("linear").fit(X, y)
    assert isinstance(corrector, ResidualCorrector)
    assert corrector.predict_next([0.0, 0.0, 1.0]) == pytest.approx(0.9, abs=0.1)


@pytest.mark.parametrize("estimator", ["linear", "knn", "mlp"])
def test_correctors_are_deterministic_and_clonable(estimator: str) -> None:
    X, y = ar_windows()
    params = {"max_iter": 300} if estimator == "mlp" else None
    a = SklearnCorrector(estimator, estimator_params=params).fit(X, y)
    b = a.clone().fit(X, y)
    assert a.predict_next(X[-1]) == b.predict_next(X[-1])


def test_window_shape_must_match() -> None:
    X, y = ar_windows()
    corrector = SklearnCorrector().fit(X, y)
    with pytest.raises(ConfigurationError, match="Janela"):
        corrector.predict_next([1.0, 2.0])


def test_predict_before_fit() -> None:
    with pytest.raises(ConfigurationError, match="fit antes"):
        SklearnCorrector().predict_next([1.0])


@pytest.mark.parametrize(
    ("X", "y"),
    [
        (np.ones(3), np.ones(3)),
        (np.ones((3, 2)), np.ones(2)),
        (np.full((3, 2), np.nan), np.ones(3)),
    ],
)
def test_invalid_training_data(X: np.ndarray, y: np.ndarray) -> None:
    with pytest.raises(DataValidationError):
        SklearnCorrector().fit(X, y)


def test_estimator_failure_is_a_candidate_error() -> None:
    X, y = ar_windows(n=6)
    with pytest.raises(CandidateFitError):
        SklearnCorrector("knn", estimator_params={"n_neighbors": 50}).fit(X, y).predict_next(X[0])


def test_fit_warnings_are_recorded_not_emitted() -> None:
    X, y = ar_windows()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        corrector = SklearnCorrector("mlp", estimator_params={"max_iter": 1}).fit(X, y)
    assert any(w.startswith("ConvergenceWarning") for w in corrector.fit_warnings_)
    assert SklearnCorrector("linear").fit(X, y).fit_warnings_ == []


def test_unknown_corrector() -> None:
    with pytest.raises(ConfigurationError, match="Corretor desconhecido"):
        SklearnCorrector("arima")


def test_roll_forward_and_recursion() -> None:
    assert roll_forward(np.array([1.0, 2.0, 3.0]), 9.0).tolist() == [2.0, 3.0, 9.0]

    class Last:
        def predict_next(self, window: np.ndarray) -> float:
            return float(window[-1] * 2)

    assert recursive_correction(Last(), np.array([0.0, 1.0]), 3).tolist() == [2.0, 4.0, 8.0]
