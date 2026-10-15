"""Durable ablation execution records, separate from forecasting model tracking.

The store only reuses complete case results. It never supplies trials to model search
or selection. A process lock protects the single-writer contract, including on resume.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any, BinaryIO

import pandas as pd

from autohybts.domain.identity import canonical_json
from autohybts.exceptions import ConfigurationError
from autohybts.persistence.manifest import json_safe

SCHEMA = "autohybts-ablation-store/1"


def series_fingerprint(series: pd.Series) -> str:
    """Dataset identity only; values and timestamps are not written to the store."""

    hashed = pd.util.hash_pandas_object(series, index=True).to_numpy().tobytes()
    return hashlib.sha256(hashed).hexdigest()


class Checkpoint:
    """One immutable study identity with atomic case records and one writer."""

    def __init__(self, path: str | Path, identity: str, protocol: dict[str, Any], resume: bool):
        self.path = Path(path)
        self.identity = identity
        self.protocol = protocol
        self.resume = resume
        self.connection: sqlite3.Connection | None = None
        self.lock: BinaryIO | None = None

    def __enter__(self) -> Checkpoint:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = self.path.with_name(self.path.name + ".lock").open("a+b")
        if self.lock.seek(0, os.SEEK_END) == 0:
            self.lock.write(b"0")
            self.lock.flush()
        self.lock.seek(0)
        try:
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(self.lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self.lock.close()
            self.lock = None
            raise ConfigurationError("O checkpoint já está em uso por outra execução.") from error
        try:
            existed = self.path.exists()
            if existed and not self.resume:
                raise ConfigurationError("Checkpoint existente: use resume=True ou outro arquivo.")
            if not existed and self.resume:
                raise ConfigurationError("Não há checkpoint para retomar.")
            self.connection = sqlite3.connect(self.path)
            if existed:
                row = self.connection.execute(
                    "SELECT schema, identity, protocol FROM metadata"
                ).fetchone()
                if row is None or row[:2] != (SCHEMA, self.identity):
                    raise ConfigurationError(
                        "Checkpoint incompatível com protocolo, dados ou código."
                    )
                self.protocol = json.loads(row[2])
            else:
                with self.connection:
                    self.connection.execute(
                        "CREATE TABLE metadata (schema TEXT, identity TEXT, protocol TEXT)"
                    )
                    self.connection.execute(
                        "INSERT INTO metadata VALUES (?, ?, ?)",
                        (SCHEMA, self.identity, canonical_json(self.protocol)),
                    )
                    self.connection.execute(
                        "CREATE TABLE cases (unit_key TEXT PRIMARY KEY, payload TEXT, digest TEXT)"
                    )
            return self
        except sqlite3.DatabaseError as error:
            self.close()
            raise ConfigurationError("Checkpoint inválido ou corrompido.") from error
        except BaseException:
            self.close()
            raise

    def read(self) -> dict[str, dict[str, Any]]:
        assert self.connection is not None
        states = {}
        for key, payload, digest in self.connection.execute("SELECT * FROM cases"):
            if hashlib.sha256(payload.encode("utf-8")).hexdigest() != digest:
                raise ConfigurationError("Registro de checkpoint corrompido.")
            states[key] = json.loads(payload)
        return states

    def write(self, key: str, state: dict[str, Any]) -> None:
        assert self.connection is not None
        payload = canonical_json(json_safe(state))
        with self.connection:
            self.connection.execute(
                "INSERT OR REPLACE INTO cases VALUES (?, ?, ?)",
                (key, payload, hashlib.sha256(payload.encode("utf-8")).hexdigest()),
            )

    def close(self) -> None:
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        if self.lock is not None:
            # Closing the descriptor also releases the OS lock after a process exits.
            self.lock.close()
            self.lock = None

    def __exit__(self, *args: Any) -> None:
        self.close()
