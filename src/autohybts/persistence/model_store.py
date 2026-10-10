"""Saving and loading finalized models (§34.7).

An artifact is a directory with ``manifest.json`` (plain data, safe to read) and
``model.pkl`` (executable on load). ``load_model`` refuses to unpickle unless the caller
declares trust explicitly, and even then checks the payload checksum and the schema.
``trusted=True`` is a declaration by the user, not a security mechanism: never use it with
files of unknown origin.

The checksum sits next to the payload, so it catches corruption, not tampering: whoever can
edit ``model.pkl`` can edit the manifest too. Two checks do resist tampering:

* ``sha256=``: the payload hash obtained from a trusted channel (e.g. the release notes).
* ``signing_key=``: an HMAC-SHA256 of the payload written by ``save_model`` with a secret
  key; ``load_model`` with the same key refuses an artifact without a valid signature.
"""

from __future__ import annotations

import hashlib
import hmac
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


def save_model(
    model: FinalizedForecaster | PanelForecaster,
    path: str | Path,
    *,
    signing_key: bytes | None = None,
) -> Path:
    """Write ``model`` to the directory ``path`` (created if needed); returns the directory.

    With ``signing_key`` (a secret kept outside the artifact), the manifest also records an
    HMAC-SHA256 of the payload that ``load_model(..., signing_key=...)`` requires.
    """

    if not isinstance(model, FinalizedForecaster | PanelForecaster):
        raise ArtifactCompatibilityError("save_model espera um modelo finalizado.")
    key = _key(signing_key)
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    payload = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
    file = target / _PAYLOAD
    if file.is_symlink():
        raise ArtifactCompatibilityError(f"{file} é um link simbólico; escolha outra pasta.")
    file.write_bytes(payload)
    payload_info = {"file": _PAYLOAD, "sha256": hashlib.sha256(payload).hexdigest()}
    if key is not None:
        payload_info["hmac_sha256"] = hmac.new(key, payload, hashlib.sha256).hexdigest()
    manifest = {
        **dict(model.manifest),
        "schema": MODEL_SCHEMA,
        "saved_utc": utc_now(),
        "environment": environment(),
        "payload": payload_info,
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


def load_model(
    path: str | Path,
    *,
    trusted: bool = False,
    sha256: str | None = None,
    signing_key: bytes | None = None,
) -> FinalizedForecaster | PanelForecaster:
    """Load a finalized model; requires ``trusted=True`` because unpickling runs code.

    ``sha256`` pins the payload hash and ``signing_key`` requires the HMAC written by
    ``save_model``; both are checked before anything is unpickled.
    """

    manifest = read_manifest(path)
    if not trusted:
        raise ArtifactCompatibilityError(
            "Carregar o modelo executa código (pickle). Confira a origem do arquivo e use "
            "load_model(path, trusted=True); para só inspecionar, use read_manifest(path)."
        )
    key = _key(signing_key)
    payload_info = manifest.get("payload")
    if not isinstance(payload_info, dict) or payload_info.get("file", _PAYLOAD) != _PAYLOAD:
        raise ArtifactCompatibilityError(f"O manifesto deve apontar para {_PAYLOAD!r}.")
    file = Path(path) / _PAYLOAD
    if file.is_symlink() or not file.is_file():
        raise ArtifactCompatibilityError(f"{file} ausente ou é um link simbólico.")
    payload = file.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if not hmac.compare_digest(digest, str(payload_info.get("sha256", ""))):
        raise ArtifactCompatibilityError("Checksum do modelo não confere com o manifesto.")
    if sha256 is not None and not hmac.compare_digest(digest, sha256.strip().lower()):
        raise ArtifactCompatibilityError(
            "O SHA-256 do modelo não confere com o valor fixado (sha256=); o arquivo não é o "
            "esperado."
        )
    if key is not None:
        expected = hmac.new(key, payload, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, str(payload_info.get("hmac_sha256", ""))):
            raise ArtifactCompatibilityError(
                "Assinatura HMAC ausente ou inválida: o artefato não foi salvo com esta chave "
                "ou foi alterado."
            )
    # Executes code from the file: only reached with trusted=True and matching checks.
    model = pickle.loads(payload)
    if not isinstance(model, FinalizedForecaster | PanelForecaster):
        raise ArtifactCompatibilityError("O artefato não contém um modelo finalizado.")
    if stable_hash(model.spec.describe()) != stable_hash(manifest.get("spec", {}).get("describe")):
        raise ArtifactCompatibilityError("A especificação do modelo diverge do manifesto.")
    return model


def _key(signing_key: bytes | None) -> bytes | None:
    if signing_key is None:
        return None
    if not isinstance(signing_key, bytes | bytearray) or len(signing_key) < 16:
        raise ArtifactCompatibilityError(
            "signing_key deve ser bytes com ao menos 16 bytes (ex.: secrets.token_bytes(32))."
        )
    return bytes(signing_key)
