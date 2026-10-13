"""Publication refuses mismatched tags, stale builds and invalid distribution metadata."""

from __future__ import annotations

import importlib.util
import io
import tarfile
import zipfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "check_release.py"
SPEC = importlib.util.spec_from_file_location("check_release", SCRIPT)
assert SPEC is not None
assert SPEC.loader is not None
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)


@pytest.mark.parametrize(("version", "prerelease"), [("1.0.0rc1", True), ("1.0.0", False)])
def test_tag_routes_candidate_and_stable_release(version: str, prerelease: bool) -> None:
    assert release.check_tag(f"v{version}", version) is prerelease


@pytest.mark.parametrize("tag", ["v1.0.0", "v0.0.1", "1.0.0rc1", "v1.0.0rc1\n"])
def test_tag_must_match_the_exact_package_version(tag: str) -> None:
    with pytest.raises(ValueError, match="does not match"):
        release.check_tag(tag, "1.0.0rc1")


@pytest.mark.parametrize("version", ["1.0", "01.0.0", "1.0.0dev1", "1.0.0rc0", "1.0.0+local"])
def test_unsupported_versions_cannot_be_published(version: str) -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        release.check_tag(f"v{version}", version)


def test_reading_the_version_does_not_execute_the_package(tmp_path: Path) -> None:
    source = tmp_path / "version.py"
    source.write_text('raise RuntimeError("do not import")\n__version__ = "1.0.0rc1"\n')
    assert release.read_version(source) == "1.0.0rc1"


def distributions(directory: Path, *, name: str = "autohybts", version: str = "1.0.0rc1") -> None:
    payload = f"Metadata-Version: 2.4\nName: {name}\nVersion: {version}\n".encode()
    with zipfile.ZipFile(directory / "autohybts-1.0.0rc1-py3-none-any.whl", "w") as wheel:
        wheel.writestr("autohybts-1.0.0rc1.dist-info/METADATA", payload)
    with tarfile.open(directory / "autohybts-1.0.0rc1.tar.gz", "w:gz") as sdist:
        member = tarfile.TarInfo("autohybts-1.0.0rc1/PKG-INFO")
        member.size = len(payload)
        sdist.addfile(member, io.BytesIO(payload))


def test_both_distribution_metadata_match(tmp_path: Path) -> None:
    distributions(tmp_path)
    release.check_distributions(tmp_path, "1.0.0rc1")


@pytest.mark.parametrize("extra", ["autohybts-0.0.1.tar.gz", "leftover.txt"])
def test_stale_builds_are_not_uploaded_with_the_release(tmp_path: Path, extra: str) -> None:
    distributions(tmp_path)
    (tmp_path / extra).write_bytes(b"stale")
    with pytest.raises(ValueError, match="exactly one"):
        release.check_distributions(tmp_path, "1.0.0rc1")


@pytest.mark.parametrize(("name", "version"), [("other", "1.0.0rc1"), ("autohybts", "0.0.1")])
def test_distribution_metadata_must_match_the_release(
    tmp_path: Path, name: str, version: str
) -> None:
    distributions(tmp_path, name=name, version=version)
    with pytest.raises(ValueError, match="does not describe"):
        release.check_distributions(tmp_path, "1.0.0rc1")
