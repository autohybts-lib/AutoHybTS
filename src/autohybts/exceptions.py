"""Exception hierarchy.

Configuration errors stop a run. Errors raised while fitting or predicting a single
candidate are caught by the search and recorded as auditable ``FAILED`` trials.
"""

from __future__ import annotations

__all__ = [
    "ArtifactCompatibilityError",
    "AutoHybTSError",
    "CandidateFitError",
    "ConfigurationError",
    "CoverageError",
    "DataValidationError",
    "DatasetIntegrityError",
    "DatasetUnavailableError",
    "InsufficientHistoryError",
    "ProtocolConfigurationError",
    "SealedTestAccessError",
    "TemporalLeakageError",
    "UnsupportedCapabilityError",
]


class AutoHybTSError(Exception):
    """Base class for every error raised by autohybts."""


class ConfigurationError(AutoHybTSError, ValueError):
    """Invalid or inconsistent configuration; stops the run."""


class DataValidationError(ConfigurationError):
    """Input series is invalid (unordered, duplicated, irregular, non-numeric or missing values).

    Never fixed silently: the library does not reorder, resample, deduplicate or impute.
    """


class ProtocolConfigurationError(ConfigurationError):
    """Configuration cannot honour the declared scientific protocol.

    Example: the ``research_strict`` profile without an explicitly defined
    variability criterion (open question A01).
    """


class InsufficientHistoryError(AutoHybTSError, ValueError):
    """Not enough observations for the requested split, window or horizon."""


class TemporalLeakageError(AutoHybTSError):
    """Information not available at the forecast origin would be used."""


class SealedTestAccessError(TemporalLeakageError):
    """Search, selection or tuning tried to read the sealed test segment."""


class UnsupportedCapabilityError(AutoHybTSError, NotImplementedError):
    """Model, strategy or API path not supported by this adapter or version."""


class CandidateFitError(AutoHybTSError):
    """A single candidate failed to fit or predict."""


class CoverageError(AutoHybTSError):
    """A candidate lacks valid forecasts for some required evaluation unit."""


class ArtifactCompatibilityError(AutoHybTSError):
    """A persisted artifact is incompatible, corrupted or untrusted."""


class DatasetUnavailableError(AutoHybTSError):
    """A dataset is not in the local cache and cannot be downloaded (offline or network)."""


class DatasetIntegrityError(AutoHybTSError):
    """A dataset failed a safety check: checksum, unapproved URL or unsafe archive."""
