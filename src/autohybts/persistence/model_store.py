"""Saving and loading finalized models (§34.7).

An artifact is a directory with ``manifest.json`` (plain data, safe to read) and
``model.pkl`` (executable on load). ``load_model`` refuses to unpickle unless the caller
declares trust explicitly, and even then checks the payload checksum and the schema.
``trusted=True`` is a declaration by the user, not a security mechanism: never use it with
files of unknown origin.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Final

from autohybts.domain.identity import stable_hash
from autohybts.exceptions import ArtifactCompatibilityError
from autohybts.forecasting.finalized import FinalizedForecaster, PanelForecaster
from autohybts.persistence.manifest import environment, utc_now

__all__ = ["MODEL_SCHEMA", "load_model", "read_manifest", "save_model"]

MODEL_SCHEMA: Final = "autohybts-model/1"
_MANIFEST: Final = "manifest.json"
_PAYLOAD: Final = "model.pkl"


def save_model(model: FinalizedForecaster | PanelForecaster, path: str | Path) -> Path:
    """Write ``model`` to the directory ``path`` (created if needed); returns the directory."""

    if not isinstance(model, FinalizedForecaster | PanelForecaster):
        raise ArtifactCompatibilityError("save_model espera um modelo finalizado.")
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    payload = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
    (target / _PAYLOAD).write_bytes(payload)
    manifest = {
        **dict(model.manifest),
        "schema": MODEL_SCHEMA,
        "saved_utc": utc_now(),
        "environment": environment(),
        "payload": {"file": _PAYLOAD, "sha256": hashlib.sha256(payload).hexdigest()},
    }
    (target / _MANIFEST).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return target


def read_manifest(path: str | Path) -> dict[str, Any]:
    """Inspect an artifact without executing anything."""

    file = Path(path) / _MANIFEST
    if not file.is_file():
        raise ArtifactCompatibilityError(f"Manifesto não encontrado em {file}.")
    try:
        manifest: dict[str, Any] = json.loads(file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ArtifactCompatibilityError(f"Manifesto inválido em {file}.") from error
    if manifest.get("schema") != MODEL_SCHEMA:
        raise ArtifactCompatibilityError(
            f"Esquema {manifest.get('schema')!r} não suportado; esperado {MODEL_SCHEMA!r}."
        )
    return manifest


def load_model(path: str | Path, *, trusted: bool = False) -> FinalizedForecaster | PanelForecaster:
    """Load a finalized model; requires ``trusted=True`` because unpickling runs code."""

    manifest = read_manifest(path)
    if not trusted:
        raise ArtifactCompatibilityError(
            "Carregar o modelo executa código (pickle). Confira a origem do arquivo e use "
            "load_model(path, trusted=True); para só inspecionar, use read_manifest(path)."
        )
    payload_info = manifest.get("payload", {})
    payload = (Path(path) / str(payload_info.get("file", _PAYLOAD))).read_bytes()
    if hashlib.sha256(payload).hexdigest() != payload_info.get("sha256"):
        raise ArtifactCompatibilityError("Checksum do modelo não confere com o manifesto.")
    # Executes code from the file: only reached with trusted=True and a matching checksum.
    model = pickle.loads(payload)
    if not isinstance(model, FinalizedForecaster | PanelForecaster):
        raise ArtifactCompatibilityError("O artefato não contém um modelo finalizado.")
    if stable_hash(model.spec.describe()) != stable_hash(manifest.get("spec", {}).get("describe")):
        raise ArtifactCompatibilityError("A especificação do modelo diverge do manifesto.")
    return model
