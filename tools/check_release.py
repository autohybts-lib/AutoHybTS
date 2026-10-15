"""Check release tags and distribution metadata without importing the package."""

from __future__ import annotations

import argparse
import ast
import re
import tarfile
import zipfile
from email.parser import BytesParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION_FILE = ROOT / "src" / "autohybts" / "__init__.py"
RELEASE_VERSION = re.compile(r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(rc[1-9]\d*)?", re.ASCII)


def read_version(path: Path = VERSION_FILE) -> str:
    """Read the literal version assignment; never execute package code."""
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets
        ):
            version = ast.literal_eval(node.value)
            if isinstance(version, str):
                return version
    raise ValueError(f"No literal __version__ in {path}")


def check_tag(tag: str, version: str) -> bool:
    """Return whether this is an RC, refusing malformed or mismatched tags."""
    if RELEASE_VERSION.fullmatch(version) is None:
        raise ValueError(f"Unsupported release version: {version!r}; use X.Y.Z or X.Y.ZrcN")
    if tag != f"v{version}":
        raise ValueError(f"Tag {tag!r} does not match package version v{version}")
    return "rc" in version


def check_distributions(directory: Path, version: str) -> None:
    """Require one wheel and one sdist with matching package metadata."""
    wheels = sorted(directory.glob("*.whl"))
    sdists = sorted(directory.glob("*.tar.gz"))
    files = wheels + sdists
    if len(wheels) != 1 or len(sdists) != 1 or set(directory.iterdir()) != set(files):
        raise ValueError("Release directory must contain exactly one wheel and one sdist")
    for file in files:
        if file.suffix == ".whl":
            with zipfile.ZipFile(file) as wheel:
                names = [name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")]
                if len(names) != 1:
                    raise ValueError(f"Expected one wheel METADATA in {file.name}")
                payload = wheel.read(names[0])
        else:
            with tarfile.open(file, "r:gz") as sdist:
                members = [
                    member
                    for member in sdist.getmembers()
                    if len(Path(member.name).parts) == 2 and member.name.endswith("/PKG-INFO")
                ]
                if len(members) != 1 or not members[0].isfile():
                    raise ValueError(f"Expected one top-level PKG-INFO in {file.name}")
                stream = sdist.extractfile(members[0])
                if stream is None:
                    raise ValueError(f"Cannot read PKG-INFO in {file.name}")
                with stream:
                    payload = stream.read()
        metadata = BytesParser().parsebytes(payload)
        if metadata.get("Name") != "autohybts" or metadata.get("Version") != version:
            raise ValueError(f"Distribution {file.name} does not describe autohybts {version}")
        if not file.name.startswith(f"autohybts-{version}-") and file.name != (
            f"autohybts-{version}.tar.gz"
        ):
            raise ValueError(f"Distribution filename does not match version: {file.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--dist", type=Path)
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args()
    version = read_version()
    try:
        prerelease = check_tag(args.tag, version)
        if args.dist is not None:
            check_distributions(args.dist, version)
    except ValueError as error:
        parser.error(str(error))
    if args.github_output is not None:
        with args.github_output.open("a", encoding="utf-8") as output:
            output.write(f"version={version}\nprerelease={str(prerelease).lower()}\n")
    print(f"Verified {args.tag}; target=PyPI; prerelease={str(prerelease).lower()}")


if __name__ == "__main__":
    main()
