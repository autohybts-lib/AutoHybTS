"""The exception hierarchy of §11.3 is part of the public contract."""

from __future__ import annotations

import pytest

from autohybts import exceptions as exc

PUBLIC_ERRORS = (
    exc.ConfigurationError,
    exc.DataValidationError,
    exc.ProtocolConfigurationError,
    exc.InsufficientHistoryError,
    exc.TemporalLeakageError,
    exc.SealedTestAccessError,
    exc.UnsupportedCapabilityError,
    exc.CandidateFitError,
    exc.CoverageError,
    exc.ArtifactCompatibilityError,
    exc.DatasetUnavailableError,
    exc.DatasetIntegrityError,
)


@pytest.mark.parametrize("error", PUBLIC_ERRORS)
def test_every_error_derives_from_base(error: type[Exception]) -> None:
    assert issubclass(error, exc.AutoHybTSError)
    assert error.__name__ in exc.__all__


@pytest.mark.parametrize(
    ("error", "parent"),
    [
        (exc.ProtocolConfigurationError, exc.ConfigurationError),
        (exc.DataValidationError, exc.ConfigurationError),
        (exc.SealedTestAccessError, exc.TemporalLeakageError),
        (exc.ConfigurationError, ValueError),
        (exc.InsufficientHistoryError, ValueError),
        (exc.UnsupportedCapabilityError, NotImplementedError),
    ],
)
def test_specialized_errors_keep_their_parents(
    error: type[Exception], parent: type[Exception]
) -> None:
    assert issubclass(error, parent)


def test_sealed_test_access_is_caught_as_leakage() -> None:
    with pytest.raises(exc.TemporalLeakageError):
        raise exc.SealedTestAccessError("search tried to read the sealed test")
