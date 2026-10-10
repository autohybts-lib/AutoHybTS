from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from autohybts.domain.identity import canonical_json, stable_hash

SPEC = {
    "strategy": "residual_additive",
    "alpha": 0.5,
    "window": 7,
    "base": {"name": "linear", "params": {"lags": [1, 2, 7]}},
    "corrector": None,
    "flags": [True, False],
}

# Pinned serialization and digest: if either changes, every stored candidate_id and
# trial_key changes too.
SPEC_JSON = (
    '{"alpha":0.5,"base":{"name":"linear","params":{"lags":[1,2,7]}},'
    '"corrector":null,"flags":[true,false],"strategy":"residual_additive","window":7}'
)
SPEC_DIGEST = "5605516fb1d7f58b2aa0cbf05fab4a1bbc06744fc948c41d58a99579d915c19a"


def test_canonical_json_is_compact_and_sorted() -> None:
    assert canonical_json({"b": 1, "a": [1.5, None, "x"]}) == '{"a":[1.5,null,"x"],"b":1}'


def test_serialization_and_digest_are_pinned() -> None:
    assert canonical_json(SPEC) == SPEC_JSON
    assert stable_hash(SPEC) == hashlib.sha256(SPEC_JSON.encode("utf-8")).hexdigest()
    assert stable_hash(SPEC) == SPEC_DIGEST


def test_stable_hash_is_order_independent() -> None:
    reordered = dict(reversed(list(SPEC.items())))
    assert stable_hash(reordered) == stable_hash(SPEC)


@pytest.mark.parametrize("seed", ["0", "1", "12345"])
def test_stable_hash_is_identical_across_processes(seed: str) -> None:
    code = (
        "from autohybts.domain.identity import stable_hash; import json, sys; "
        "print(stable_hash(json.loads(sys.argv[1])))"
    )
    env = {**os.environ, "PYTHONHASHSEED": seed}
    result = subprocess.run(
        [sys.executable, "-c", code, json.dumps(SPEC)],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    assert result.stdout.strip() == stable_hash(SPEC)


def test_tuple_and_list_share_identity() -> None:
    assert stable_hash({"lags": (1, 2)}) == stable_hash({"lags": [1, 2]})


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (1, 1.0),  # sklearn: max_features=1 (one feature) vs 1.0 (all features)
        (True, 1),
        (False, 0),
        ("1", 1),
    ],
)
def test_distinct_types_keep_distinct_identities(left: object, right: object) -> None:
    assert stable_hash({"v": left}) != stable_hash({"v": right})


def test_negative_zero_equals_zero() -> None:
    assert stable_hash({"v": -0.0}) == stable_hash({"v": 0.0})


@pytest.mark.parametrize(
    ("numpy_value", "python_value"),
    [
        (np.int64(3), 3),
        (np.float32(0.5), 0.5),
        (np.float64(0.1), 0.1),
        (np.bool_(True), True),
        (np.array([1, 2, 3]), [1, 2, 3]),
    ],
)
def test_numpy_values_match_python_equivalents(numpy_value: object, python_value: object) -> None:
    assert canonical_json(numpy_value) == canonical_json(python_value)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), np.float64("nan")])
def test_non_finite_floats_are_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="Non-finite"):
        stable_hash({"v": value})


@pytest.mark.parametrize("value", [{1: "a"}, {("a",): 1}])
def test_non_string_keys_are_rejected(value: object) -> None:
    with pytest.raises(TypeError, match="keys must be str"):
        canonical_json(value)


@pytest.mark.parametrize("value", [{1, 2}, object(), b"bytes", 1 + 2j])
def test_unsupported_types_are_rejected(value: object) -> None:
    with pytest.raises(TypeError, match="no canonical serialization"):
        canonical_json(value)


scalars = (
    st.none()
    | st.booleans()
    | st.integers()
    | st.floats(allow_nan=False, allow_infinity=False)
    | st.text()
)
documents = st.recursive(
    scalars,
    lambda children: (
        st.lists(children, max_size=4) | st.dictionaries(st.text(), children, max_size=4)
    ),
    max_leaves=20,
)


@given(st.dictionaries(st.text(), documents, max_size=6))
def test_hash_ignores_insertion_order(document: dict[str, object]) -> None:
    reordered = dict(reversed(list(document.items())))
    assert stable_hash(reordered) == stable_hash(document)


@given(documents)
def test_canonical_json_round_trips(document: object) -> None:
    text = canonical_json(document)
    assert canonical_json(json.loads(text)) == text
