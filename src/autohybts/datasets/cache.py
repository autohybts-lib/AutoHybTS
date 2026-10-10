"""Local cache of downloaded datasets with integrity checks (§32.4).

* Files live under ``AUTOHYBTS_DATA_DIR`` or ``~/.cache/autohybts/datasets``, one folder
  per provider. Nothing is downloaded on import or install (D10): only an explicit
  ``fetch`` touches the network.
* Only HTTPS URLs on the provider's approved domains are fetched (D11), with a timeout,
  a few retries and a size cap. Redirects are followed only to HTTPS URLs and the final
  URL must be on the same approved domains.
* Provider and file names are single path components (letters, digits, ``._+-``), so no
  name coming from a remote index can write outside the cache folder.
* Every file gets a manifest (URL, access time, SHA-256, size, license, version). A file
  whose bytes no longer match its manifest, or a pinned checksum, is refused (D03).
* ``offline=True`` serves from the cache only and says what is missing (D01, D02).
* Archives are extracted member by member, refusing absolute paths, ``..``, links and
  members larger than a size cap.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tarfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Collection
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import IO, Any, Final
from urllib.parse import urlparse

from autohybts.exceptions import (
    ConfigurationError,
    DatasetIntegrityError,
    DatasetUnavailableError,
)

__all__ = [
    "DEFAULT_CACHE_ENV",
    "CacheManager",
    "RemoteFile",
    "default_cache_dir",
    "safe_extract_member",
    "safe_name",
    "sha256_of",
]

DEFAULT_CACHE_ENV: Final = "AUTOHYBTS_DATA_DIR"
_CHUNK: Final = 1 << 20
_SAFE_NAME: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,199}")
Opener = Callable[[str, float], IO[bytes]]


def default_cache_dir() -> Path:
    configured = os.environ.get(DEFAULT_CACHE_ENV)
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".cache" / "autohybts" / "datasets"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class RemoteFile:
    """A file a provider may download; ``sha256`` pins its content when known."""

    url: str
    filename: str
    sha256: str | None = None
    max_bytes: int = 500 * 1024 * 1024


def safe_name(name: str, what: str = "nome") -> str:
    """``name`` if it is a single, plain path component; raises otherwise."""

    if not isinstance(name, str) or not _SAFE_NAME.fullmatch(name) or ".." in name:
        raise DatasetIntegrityError(
            f"{what} inseguro para o cache: {name!r} (use letras, dígitos e '._+-', sem "
            "separadores de caminho nem '..')."
        )
    return name


class _HttpsOnlyRedirects(urllib.request.HTTPRedirectHandler):
    """Refuses a redirect that leaves HTTPS; the final host is checked by ``fetch``."""

    def redirect_request(  # type: ignore[no-untyped-def]
        self, req, fp, code, msg, headers, newurl
    ):
        if urlparse(newurl).scheme != "https":
            raise DatasetIntegrityError(f"Redirecionamento para fora do HTTPS recusado: {newurl}.")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _urlopen(url: str, timeout: float) -> IO[bytes]:
    request = urllib.request.Request(url, headers={"User-Agent": "autohybts-datasets"})
    opener = urllib.request.build_opener(_HttpsOnlyRedirects)
    response: IO[bytes] = opener.open(request, timeout=timeout)
    return response


class CacheManager:
    def __init__(
        self,
        root: str | Path | None = None,
        *,
        offline: bool = False,
        timeout: float = 60.0,
        retries: int = 2,
        opener: Opener | None = None,
    ) -> None:
        if retries < 0 or timeout <= 0:
            raise ConfigurationError("timeout deve ser positivo e retries >= 0.")
        self.root = Path(root).expanduser() if root is not None else default_cache_dir()
        self.offline = offline
        self.timeout = timeout
        self.retries = retries
        self._open = opener or _urlopen

    def path(self, provider: str, filename: str) -> Path:
        return self.root / safe_name(provider, "provedor") / safe_name(filename, "arquivo")

    def manifest_path(self, provider: str, filename: str) -> Path:
        return self.path(provider, filename).with_name(f"{filename}.manifest.json")

    def read_manifest(self, provider: str, filename: str) -> dict[str, Any] | None:
        path = self.manifest_path(provider, filename)
        if not path.exists():
            return None
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise DatasetIntegrityError(
                f"Manifesto ilegível em {path}; apague-o ou baixe de novo com refresh=True."
            ) from error
        if not isinstance(loaded, dict):
            raise DatasetIntegrityError(f"Manifesto inválido em {path}.")
        return loaded

    def fetch(
        self,
        provider: str,
        remote: RemoteFile,
        *,
        allowed_domains: Collection[str],
        license: str | None = None,
        version: str | None = None,
        refresh: bool = False,
    ) -> Path:
        """Path of ``remote`` in the cache, downloading it first if needed and allowed."""

        _check_url(remote.url, allowed_domains)
        target = self.path(provider, remote.filename)
        if target.exists() and not refresh:
            self.verify(provider, remote)
            return target
        if self.offline:
            raise DatasetUnavailableError(
                f"{provider}/{remote.filename} não está no cache ({target.parent}) e o modo "
                "offline está ativo. Rode uma vez com offline=False para baixar."
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".part")
        self._download(remote, partial, allowed_domains)
        digest = sha256_of(partial)
        if remote.sha256 is not None and digest != remote.sha256:
            partial.unlink(missing_ok=True)
            raise DatasetIntegrityError(
                f"Checksum de {remote.url} não confere: esperado {remote.sha256}, obtido "
                f"{digest}. O arquivo de origem mudou ou o download foi adulterado."
            )
        partial.replace(target)
        manifest = {
            "provider": provider,
            "filename": remote.filename,
            "url": remote.url,
            "accessed_at": datetime.now(UTC).isoformat(),
            "sha256": digest,
            "size": target.stat().st_size,
            "pinned": remote.sha256 is not None,
            "license": license,
            "version": version,
        }
        self.manifest_path(provider, remote.filename).write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return target

    def verify(self, provider: str, remote: RemoteFile) -> None:
        """Refuse a cached file whose bytes differ from its manifest or pinned checksum."""

        target = self.path(provider, remote.filename)
        manifest = self.read_manifest(provider, remote.filename)
        if manifest is None:
            raise DatasetIntegrityError(
                f"{target} está no cache sem manifesto; apague-o ou baixe de novo com refresh=True."
            )
        digest = sha256_of(target)
        expected = remote.sha256 or manifest.get("sha256")
        if digest != expected or digest != manifest.get("sha256"):
            raise DatasetIntegrityError(
                f"{target} foi alterado depois do download (SHA-256 {digest}, esperado "
                f"{expected}). Apague-o ou baixe de novo com refresh=True."
            )

    def _download(
        self, remote: RemoteFile, destination: Path, allowed_domains: Collection[str]
    ) -> None:
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                with (
                    self._open(remote.url, self.timeout) as response,
                    destination.open("wb") as out,
                ):
                    # After redirects, the bytes must still come from an approved HTTPS host.
                    final = getattr(response, "url", None)
                    if isinstance(final, str) and final:
                        _check_url(final, allowed_domains)
                    written = 0
                    for chunk in iter(lambda: response.read(_CHUNK), b""):
                        written += len(chunk)
                        if written > remote.max_bytes:
                            raise DatasetIntegrityError(
                                f"{remote.url} passou de {remote.max_bytes} bytes; download "
                                "interrompido."
                            )
                        out.write(chunk)
                return
            except DatasetIntegrityError:
                destination.unlink(missing_ok=True)
                raise
            except (urllib.error.URLError, TimeoutError, OSError) as error:
                last = error
                destination.unlink(missing_ok=True)
                if attempt < self.retries:
                    time.sleep(min(2.0**attempt, 5.0))
        raise DatasetUnavailableError(
            f"Não foi possível baixar {remote.url} após {self.retries + 1} tentativa(s): "
            f"{last}. Verifique a conexão ou use um cache já preenchido (offline=True)."
        )


def _check_url(url: str, allowed_domains: Collection[str]) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise DatasetIntegrityError(f"Só downloads HTTPS são aceitos: {url}.")
    if parsed.hostname not in set(allowed_domains):
        raise DatasetIntegrityError(
            f"Domínio {parsed.hostname!r} não aprovado para este provedor "
            f"({sorted(allowed_domains)})."
        )


def safe_extract_member(
    archive: Path, member_name: str, destination: Path, *, max_bytes: int = 200 * 1024 * 1024
) -> Path:
    """Extract one regular file from a tar archive without escaping ``destination``."""

    destination = destination.resolve()
    with tarfile.open(archive) as tar:
        try:
            member = tar.getmember(member_name)
        except KeyError:
            raise DatasetIntegrityError(f"{member_name} não está em {archive.name}.") from None
        pure = PurePosixPath(member.name)
        if pure.is_absolute() or ".." in pure.parts or not member.isfile():
            raise DatasetIntegrityError(
                f"Membro inseguro em {archive.name}: {member.name!r} (caminho absoluto, '..' "
                "ou link)."
            )
        if member.size > max_bytes:
            raise DatasetIntegrityError(
                f"{member.name!r} tem {member.size} bytes, acima do limite de {max_bytes}."
            )
        target = (destination / Path(*pure.parts)).resolve()
        if destination not in target.parents:
            raise DatasetIntegrityError(f"{member.name!r} sairia de {destination}.")
        target.parent.mkdir(parents=True, exist_ok=True)
        source = tar.extractfile(member)
        if source is None:  # pragma: no cover - isfile() guarantees a file object
            raise DatasetIntegrityError(f"{member.name!r} não pôde ser lido.")
        with source, target.open("wb") as out:
            shutil.copyfileobj(source, out)
    return target
