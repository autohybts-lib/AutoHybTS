"""Stable identities for specs and trials (architecture §5.3 and §8.4).

``candidate_id`` and ``trial_key`` are built on these functions, and so are cache
reuse and deterministic tie-breaking. Changing the serialization changes every ID
and invalidates stored trials and snapshots, so treat it as a versioned contract.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import TypeAlias

import numpy as np

__all__ = ["JsonValue", "canonical_json", "derive_seed", "stable_hash"]

JsonValue: TypeAlias = bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"] | None


def canonical_json(obj: object) -> str:
    """Serialize ``obj`` deterministically: sorted keys, compact separators, normalized floats.

    Supported: ``None``, ``bool``, ``int``, ``float``, ``str``, mappings with string keys,
    lists, tuples, NumPy scalars and arrays. Ints and floats stay distinct (``1`` vs
    ``1.0``) because estimators such as scikit-learn give them different meanings.

    Raises:
        TypeError: unsupported type or non-string mapping key.
        ValueError: ``NaN`` or infinite float.
    """

    return json.dumps(
        _normalize(obj),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def stable_hash(obj: object) -> str:
    """SHA-256 hex digest of :func:`canonical_json`, stable across processes and platforms."""

    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def derive_seed(session_seed: int, *keys: object) -> int:
    """Seed in ``[0, 2**31)`` derived from the session seed and identifying keys (§18).

    Independent of execution order and of Python's hash randomization, so parallel and
    sequential runs draw the same seeds.
    """

    digest = hashlib.sha256(canonical_json([session_seed, *keys]).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % 2**31


def _normalize(obj: object) -> JsonValue:
    if obj is None:
        return None
    # bool is checked before int so that True never collapses into 1.
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, np.generic):
        return _normalize(obj.item())
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError(f"Non-finite float has no stable identity: {obj!r}")
        # -0.0 == 0.0 but serializes differently; both must map to one identity.
        return 0.0 if obj == 0.0 else float(obj)
    if isinstance(obj, str):
        return str(obj)
    if isinstance(obj, Mapping):
        normalized: dict[str, JsonValue] = {}
        for key, value in obj.items():
            if not isinstance(key, str):
                raise TypeError(f"Mapping keys must be str, got {type(key).__name__}: {key!r}")
            normalized[str(key)] = _normalize(value)
        return normalized
    if isinstance(obj, list | tuple):
        return [_normalize(item) for item in obj]
    if isinstance(obj, np.ndarray):
        return _normalize(obj.tolist())
    raise TypeError(f"Type {type(obj).__name__} has no canonical serialization: {obj!r}")
