"""Evidence integrity, strict JSON and separation of export directories."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from autohybts.exceptions import ConfigurationError
from autohybts.reporting.evidence import write_evidence


def test_evidence_json_preserves_undefined_values_and_file_hashes(tmp_path):
    bundle = write_evidence(
        tmp_path / "bundle",
        experiment_id="protocol",
        tables={},
        documents={
            "metrics": {"undefined": np.array([np.nan, np.inf]), "date": pd.Timestamp("2026-01-01")}
        },
    )
    content = (bundle.path / "metrics.json").read_text()
    assert "NaN" not in content
    assert "Infinity" not in content
    assert json.loads(content) == {"undefined": [None, None], "date": "2026-01-01T00:00:00"}
    manifest = json.loads((bundle.path / "manifest.json").read_text())
    assert (
        manifest["sha256"]["metrics.json"]
        == hashlib.sha256((bundle.path / "metrics.json").read_bytes()).hexdigest()
    )
    with pytest.raises(ConfigurationError, match="vazia"):
        write_evidence(bundle.path, experiment_id="other", tables={}, documents={})


def test_evidence_rejects_paths_outside_bundle(tmp_path):
    with pytest.raises(ConfigurationError, match="locais"):
        write_evidence(
            tmp_path / "bundle", experiment_id="p", tables={}, documents={"../outside": {}}
        )
    assert not (tmp_path / "outside.json").exists()
