"""Reproducible evidence bundle of an experiment (§34.6).

Writes tables and metadata; series values require explicit predictions or plots.
Nothing here states statistical significance.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from autohybts.exceptions import ConfigurationError
from autohybts.persistence.manifest import environment, json_safe, source_fingerprint, utc_now

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
    figure_formats: Sequence[str] = ("png",),
    include_predictions: bool = False,
    run_id: str | None = None,
) -> EvidenceBundle:
    formats = tuple(figure_formats)
    if not formats or len(set(formats)) != len(formats) or set(formats) - {"png", "pdf", "svg"}:
        raise ConfigurationError("figure_formats deve conter formatos únicos: png, pdf ou svg.")
    names = [*tables, *documents, *(f"plot_{a.kind}" for a in figures)]
    if any(
        not name or Path(name).name != name or "\\" in name or name in {".", ".."} for name in names
    ):
        raise ConfigurationError("Nomes de evidências devem ser nomes de arquivos locais.")
    target = Path(path)
    if target.exists() and any(target.iterdir()):
        raise ConfigurationError("Use uma pasta vazia para cada exportação de evidências.")
    target.mkdir(parents=True, exist_ok=True)
    files: list[str] = []
    for name, table in tables.items():
        file = f"{name}.csv"
        table.to_csv(target / file)
        files.append(file)
    for name, document in documents.items():
        file = f"{name}.json"
        (target / file).write_text(
            json.dumps(json_safe(document), indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        files.append(file)
    for artifact in figures:
        for extension in formats:
            file = f"plot_{artifact.kind}.{extension}"
            artifact.save(target / file)
            files.append(file)
    manifest = {
        "schema": "autohybts-evidence/2",
        "experiment_id": experiment_id,
        "protocol_hash": experiment_id,
        "run_id": run_id or uuid4().hex,
        "export_id": uuid4().hex,
        "created_utc": utc_now(),
        "environment": environment(),
        "files": sorted(files),
        "source_fingerprint": source_fingerprint(),
        "sha256": {
            file: hashlib.sha256((target / file).read_bytes()).hexdigest() for file in sorted(files)
        },
        "raw_series_values_included": include_predictions
        or any(
            artifact.kind in {"series", "cv_forecasts", "forecasts", "decomposition"}
            for artifact in figures
        ),
        "external_predictions_included": include_predictions,
        "figure_kinds": [artifact.kind for artifact in figures],
        "warnings": list(warnings),
    }
    (target / "manifest.json").write_text(
        json.dumps(json_safe(manifest), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return EvidenceBundle(target, tuple(sorted([*files, "manifest.json"])), experiment_id)
