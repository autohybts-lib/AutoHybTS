"""Reproducible evidence bundle of an experiment (§34.6).

Writes tables and metadata, never the raw series values: forecasts and errors stay out
unless the caller exports them explicitly. Nothing here states statistical significance.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from autohybts.persistence.manifest import environment, utc_now

__all__ = ["EvidenceBundle", "write_evidence"]


@dataclass(frozen=True, slots=True)
class EvidenceBundle:
    path: Path
    files: tuple[str, ...]
    experiment_id: str


def write_evidence(
    path: str | Path,
    *,
    experiment_id: str,
    tables: Mapping[str, pd.DataFrame],
    documents: Mapping[str, Any],
    figures: Sequence[Any] = (),
    warnings: Sequence[str] = (),
) -> EvidenceBundle:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    files: list[str] = []
    for name, table in tables.items():
        file = f"{name}.csv"
        table.to_csv(target / file)
        files.append(file)
    for name, document in documents.items():
        file = f"{name}.json"
        (target / file).write_text(
            json.dumps(document, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )
        files.append(file)
    for artifact in figures:
        file = f"plot_{artifact.kind}.png"
        artifact.save(target / file)
        files.append(file)
    manifest = {
        "schema": "autohybts-evidence/1",
        "experiment_id": experiment_id,
        "created_utc": utc_now(),
        "environment": environment(),
        "files": sorted(files),
        "raw_series_values_included": False,
        "warnings": list(warnings),
    }
    (target / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return EvidenceBundle(target, tuple(sorted([*files, "manifest.json"])), experiment_id)
