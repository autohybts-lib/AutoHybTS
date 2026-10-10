"""Provenance recorded with artifacts: the git commit is read without running ``git``."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from autohybts.persistence.manifest import git_commit

SHA = "0123456789abcdef0123456789abcdef01234567"


def fake_repo(root: Path, head: str) -> Path:
    git = root / ".git"
    (git / "refs" / "heads").mkdir(parents=True)
    (git / "HEAD").write_text(head + "\n", encoding="utf-8")
    return git


def test_loose_packed_and_detached_heads(tmp_path: Path) -> None:
    git = fake_repo(tmp_path, "ref: refs/heads/main")
    (git / "refs" / "heads" / "main").write_text(SHA + "\n", encoding="utf-8")
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    assert git_commit(nested) == SHA
    (git / "refs" / "heads" / "main").unlink()
    (git / "packed-refs").write_text(f"# pack-refs\n{SHA} refs/heads/main\n", encoding="utf-8")
    assert git_commit(tmp_path) == SHA
    (git / "HEAD").write_text(SHA, encoding="utf-8")
    assert git_commit(tmp_path) == SHA


def test_worktree_file_points_to_the_git_dir(tmp_path: Path) -> None:
    common = fake_repo(tmp_path / "main", "ref: refs/heads/main")
    (common / "refs" / "heads" / "feature").write_text(SHA, encoding="utf-8")
    private = common / "worktrees" / "wt"
    private.mkdir(parents=True)
    (private / "HEAD").write_text("ref: refs/heads/feature", encoding="utf-8")
    (private / "commondir").write_text("../..", encoding="utf-8")
    tree = tmp_path / "wt"
    tree.mkdir()
    (tree / ".git").write_text(f"gitdir: {private}", encoding="utf-8")
    assert git_commit(tree) == SHA


@pytest.mark.parametrize(
    "head", ["ref: refs/../../../secret", "ref: HEAD", "not a sha", "ref: refs/heads/missing"]
)
def test_malformed_heads_give_none(tmp_path: Path, head: str) -> None:
    (tmp_path / "secret").write_text(SHA, encoding="utf-8")
    fake_repo(tmp_path / "repo", head)
    assert git_commit(tmp_path / "repo") is None


def test_no_process_is_started(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # On Windows a bare "git" resolves in the current directory first: never run it.
    def refuse(*args: object, **kwargs: object) -> None:
        pytest.fail("subprocess used")

    monkeypatch.setattr(subprocess, "run", refuse)
    monkeypatch.setattr(subprocess, "Popen", refuse)
    git = fake_repo(tmp_path, "ref: refs/heads/main")
    (git / "refs" / "heads" / "main").write_text(SHA, encoding="utf-8")
    assert git_commit(tmp_path) == SHA


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_matches_git_in_this_repository() -> None:
    root = Path(__file__).resolve().parents[2]
    expected = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False
    ).stdout.strip()
    if not expected:
        pytest.skip("not a git work tree")
    assert git_commit(root) == expected
