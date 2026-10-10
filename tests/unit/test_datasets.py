"""Dataset catalogue offline (D01–D08, D10, D11): cache, integrity, providers, fixtures.

A stub opener serves fixture files by URL, so nothing touches the network.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
import tarfile
import urllib.error
from fractions import Fraction
from pathlib import Path

import pandas as pd
import pytest

from autohybts.datasets import catalog
from autohybts.datasets.cache import CacheManager, RemoteFile, safe_extract_member, sha256_of
from autohybts.datasets.providers import (
    M3Provider,
    M4Provider,
    RdatasetsProvider,
    m3_frames,
    regular_index,
)
from autohybts.exceptions import (
    ConfigurationError,
    DatasetIntegrityError,
    DatasetUnavailableError,
    DataValidationError,
)

FIXTURES = Path(__file__).parents[1] / "fixtures" / "datasets"
M4_BASE = "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/"
RD = "https://vincentarelbundock.github.io/Rdatasets/"
ROUTES = {
    M4_BASE + "M4-info.csv": FIXTURES / "m4" / "M4-info.csv",
    M4_BASE + "Train/Daily-train.csv": FIXTURES / "m4" / "Daily-train.csv",
    M4_BASE + "Test/Daily-test.csv": FIXTURES / "m4" / "Daily-test.csv",
    RD + "datasets.csv": FIXTURES / "rdatasets" / "datasets.csv",
    RD + "csv/datasets/AirPassengers.csv": FIXTURES / "rdatasets" / "AirPassengers.csv",
    RD + "csv/datasets/iris.csv": FIXTURES / "rdatasets" / "iris.csv",
}


class StubOpener:
    """Serves fixture bytes by URL and counts calls; ``down`` simulates no network."""

    def __init__(self, routes: dict[str, Path | bytes], down: bool = False) -> None:
        self.routes = routes
        self.down = down
        self.calls: list[str] = []

    def __call__(self, url: str, timeout: float) -> io.BytesIO:
        self.calls.append(url)
        if self.down:
            raise urllib.error.URLError("network unreachable")
        source = self.routes[url]
        return io.BytesIO(source if isinstance(source, bytes) else source.read_bytes())


def cache(tmp_path: Path, **kwargs: object) -> tuple[CacheManager, StubOpener]:
    opener = StubOpener(dict(ROUTES), down=bool(kwargs.pop("down", False)))
    return CacheManager(tmp_path, opener=opener, retries=0, **kwargs), opener  # type: ignore[arg-type]


M4 = M4Provider(pinned={})
REMOTE = RemoteFile(M4_BASE + "M4-info.csv", "M4-info.csv")
DOMAINS = {"raw.githubusercontent.com"}


# ------------------------------------------------------------------ cache
def test_download_writes_a_manifest_and_offline_reads_the_cache(tmp_path: Path) -> None:
    manager, opener = cache(tmp_path)
    path = manager.fetch("m4", REMOTE, allowed_domains=DOMAINS, license="L", version="v")
    manifest = manager.read_manifest("m4", "M4-info.csv")
    assert manifest is not None
    # D07: origin, license, version and checksum travel with the file.
    assert manifest["url"] == REMOTE.url
    assert manifest["license"] == "L"
    assert manifest["version"] == "v"
    assert manifest["sha256"] == sha256_of(path)
    # D01: offline opens the cached file without the network.
    offline = CacheManager(tmp_path, offline=True, opener=StubOpener({}, down=True))
    assert offline.fetch("m4", REMOTE, allowed_domains=DOMAINS) == path
    assert opener.calls == [REMOTE.url]


def test_missing_data_offline_or_without_network_gives_a_readable_error(tmp_path: Path) -> None:
    offline, _ = cache(tmp_path, offline=True)
    with pytest.raises(DatasetUnavailableError, match="offline"):
        offline.fetch("m4", REMOTE, allowed_domains=DOMAINS)
    down, _ = cache(tmp_path, down=True)
    with pytest.raises(DatasetUnavailableError, match="Não foi possível baixar"):
        down.fetch("m4", REMOTE, allowed_domains=DOMAINS)
    assert not (tmp_path / "m4" / "M4-info.csv").exists()


def test_changed_or_unexpected_content_is_refused(tmp_path: Path) -> None:
    manager, _ = cache(tmp_path)
    path = manager.fetch("m4", REMOTE, allowed_domains=DOMAINS)
    path.write_bytes(path.read_bytes() + b"tampered")
    # D03: a cached file that no longer matches its manifest.
    with pytest.raises(DatasetIntegrityError, match="alterado"):
        manager.fetch("m4", REMOTE, allowed_domains=DOMAINS)
    # A pinned checksum that the download does not match leaves nothing behind.
    pinned = RemoteFile(REMOTE.url, "pinned.csv", sha256="0" * 64)
    with pytest.raises(DatasetIntegrityError, match="Checksum"):
        manager.fetch("m4", pinned, allowed_domains=DOMAINS)
    assert not (tmp_path / "m4" / "pinned.csv").exists()
    # A file without manifest is not trusted either.
    (tmp_path / "m4" / "orphan.csv").write_text("x")
    with pytest.raises(DatasetIntegrityError, match="manifesto"):
        manager.fetch("m4", RemoteFile(REMOTE.url, "orphan.csv"), allowed_domains=DOMAINS)


def test_only_https_on_approved_domains_and_size_capped(tmp_path: Path) -> None:
    manager, _ = cache(tmp_path)
    # D11: unapproved URLs are refused before any request.
    with pytest.raises(DatasetIntegrityError, match="HTTPS"):
        manager.fetch(
            "m4",
            RemoteFile("http://raw.githubusercontent.com/x.csv", "x.csv"),
            allowed_domains=DOMAINS,
        )
    with pytest.raises(DatasetIntegrityError, match="não aprovado"):
        manager.fetch(
            "m4", RemoteFile("https://evil.example.com/x.csv", "x.csv"), allowed_domains=DOMAINS
        )
    small = RemoteFile(REMOTE.url, "capped.csv", max_bytes=10)
    with pytest.raises(DatasetIntegrityError, match="bytes"):
        manager.fetch("m4", small, allowed_domains=DOMAINS)


def tar_with(tmp_path: Path, name: str, payload: bytes = b"data") -> Path:
    archive = tmp_path / "archive.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo(name)
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    return archive


def test_archives_cannot_escape_the_destination(tmp_path: Path) -> None:
    good = tar_with(tmp_path, "Mcomp/data/M3.rda")
    target = safe_extract_member(good, "Mcomp/data/M3.rda", tmp_path / "out")
    assert target.read_bytes() == b"data"
    # D11: a malicious member path is refused.
    evil = tar_with(tmp_path, "../../escape.txt")
    with pytest.raises(DatasetIntegrityError, match="inseguro"):
        safe_extract_member(evil, "../../escape.txt", tmp_path / "out")
    with pytest.raises(DatasetIntegrityError, match="não está"):
        safe_extract_member(good, "missing.rda", tmp_path / "out")


def test_import_never_downloads() -> None:
    # D10: importing the package and the catalogue opens no URL.
    code = (
        "import urllib.request\n"
        "def boom(*a, **k): raise SystemExit('network used on import')\n"
        "urllib.request.urlopen = boom\n"
        "import autohybts, autohybts.datasets\n"
        "print('ok')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "ok"


# ------------------------------------------------------------------ M4
def test_m4_keeps_the_official_split_and_metadata(tmp_path: Path) -> None:
    manager, _ = cache(tmp_path)
    listing = M4.list(manager, "daily")
    assert listing["series_id"].tolist() == ["D1", "D2"]
    series = M4.load(manager, "daily", series_id="D1")
    # D04, D05, D08: official split, horizon and frequency from M4-info, fixture values.
    assert series.train.tolist() == [10, 11, 12, 13, 14, 15]
    assert series.test.tolist() == [16, 17, 18]
    assert series.descriptor.horizon == 3
    assert series.descriptor.split_kind == "official"
    assert series.train.index[0] == pd.Timestamp("1994-01-03")
    assert series.test.index[0] == series.train.index[-1] + pd.Timedelta(days=1)
    assert series.descriptor.extra["index_source"] == "source"
    assert series.full().tolist()[-3:] == [16, 17, 18]
    whole = M4.load(manager, "daily")
    assert whole.series_ids == ("D1", "D2")
    # Padding cells are dropped and a bad starting date gets a synthetic calendar.
    d2 = whole.series("D2")
    assert d2.train.tolist() == [20, 21, 22, 23]
    assert d2.descriptor.extra["index_source"] == "synthetic"
    panel = whole.panel()
    assert len(panel["D1"]) == 9
    with pytest.raises(ConfigurationError):
        M4.load(manager, "daily", series_id="D9")
    with pytest.raises(ConfigurationError):
        M4.load(manager, "secondly")


def test_benchmark_series_seals_the_official_test_in_an_experiment(tmp_path: Path) -> None:
    from autohybts import AutoHybridExperiment

    manager, _ = cache(tmp_path)
    series = M4.load(manager, "daily", series_id="D1")
    values = pd.Series(range(200), index=pd.date_range("2020-01-01", periods=200), dtype=float)
    stub = type(series)(values.iloc[:190], values.iloc[190:], series.descriptor)
    exp = AutoHybridExperiment(fh=10, test_size=len(stub.test), fold=1).fit(stub.full())
    assert exp.history_.end == stub.train.index[-1]


# ------------------------------------------------------------------ M3
def m3_records() -> dict[str, dict[str, object]]:
    monthly = pd.Series(
        [1.0, 2.0, 3.0], index=[Fraction(1990), Fraction(23881, 12), Fraction(11941, 6)]
    )
    return {
        "N1": {
            "sn": ["N1"],
            "period": ["MONTHLY"],
            "type": ["MICRO"],
            "h": [2.0],
            "x": monthly,
            "xx": pd.Series([4.0, 5.0]),
            "description": ["d"],
        },
        "N2": {
            "sn": ["N2"],
            "period": ["YEARLY"],
            "type": ["MACRO"],
            "h": [1.0],
            "x": pd.Series([1.0, 2.0], index=[1975, 1976]),
            "xx": pd.Series([3.0]),
        },
        "N3": {
            "sn": ["N3"],
            "period": ["OTHER"],
            "type": ["OTHER"],
            "h": [1.0],
            "x": pd.Series([5.0, 6.0], index=[1, 2]),
            "xx": pd.Series([7.0]),
        },
        "N4": {
            "sn": ["N4"],
            "period": ["QUARTERLY"],
            "type": ["MICRO"],
            "h": [1.0],
            "x": pd.Series([1.0, 2.0], index=[Fraction(1, 1), Fraction(5, 4)]),
            "xx": pd.Series([3.0]),
        },
    }


def test_m3_records_become_official_splits_with_calendars() -> None:
    train, test, info = m3_frames(m3_records())
    n1 = train[train["series_id"] == "N1"]
    assert n1["timestamp"].tolist() == [pd.Timestamp(f"1990-0{m}-01") for m in (1, 2, 3)]
    assert test[test["series_id"] == "N1"]["timestamp"].iloc[0] == pd.Timestamp("1990-04-01")
    sources = dict(zip(info["series_id"], info["index_source"], strict=True))
    # OTHER has no calendar; a quarterly series starting in year 1 is out of range.
    assert sources == {"N1": "source", "N2": "source", "N3": "synthetic", "N4": "synthetic"}
    assert info.set_index("series_id").loc["N1", "horizon"] == 2


def test_m3_provider_reads_the_archive_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tar_with(tmp_path, "Mcomp/data/M3.rda").read_bytes()
    url = "https://cran.r-project.org/src/contrib/Mcomp_2.8.tar.gz"
    opener = StubOpener({url: archive})
    manager = CacheManager(tmp_path / "cache", opener=opener, retries=0)
    reads: list[Path] = []
    monkeypatch.setattr(
        "autohybts.datasets.providers._read_rda", lambda path: reads.append(path) or m3_records()
    )
    provider = M3Provider(sha256=None)
    series = provider.load(manager, "monthly", series_id="N1")
    assert series.test.tolist() == [4.0, 5.0]
    assert provider.list(manager, "yearly")["series_id"].tolist() == ["N2"]
    assert len(reads) == 1  # parsed once, then served from the cached CSVs


# ------------------------------------------------------------------ Rdatasets
def test_rdatasets_needs_explicit_time_target_and_frequency(tmp_path: Path) -> None:
    manager, _ = cache(tmp_path)
    provider = RdatasetsProvider()
    assert provider.list(manager, "passenger")["Item"].tolist() == ["AirPassengers"]
    # D06: no inference of time or target.
    with pytest.raises(ConfigurationError, match="time_col"):
        provider.load(manager, package="datasets", name="AirPassengers")
    with pytest.raises(ConfigurationError, match="não existem"):
        provider.load(
            manager,
            package="datasets",
            name="iris",
            target="Sepal.Length",
            time_col="time",
            frequency="D",
        )
    with pytest.raises(ConfigurationError, match="anos decimais"):
        provider.load(
            manager,
            package="datasets",
            name="AirPassengers",
            target="value",
            time_col="time",
            frequency="D",
        )
    with pytest.raises(DataValidationError, match="regular"):
        provider.load(
            manager,
            package="datasets",
            name="AirPassengers",
            target="value",
            time_col="time",
            frequency="QS",
        )
    series = provider.load(
        manager,
        package="datasets",
        name="AirPassengers",
        target="value",
        time_col="time",
        frequency="MS",
    )
    assert series.train.index[1] == pd.Timestamp("1949-02-01")
    assert series.train.tolist()[:2] == [112.0, 118.0]
    assert series.descriptor.license is not None
    with pytest.raises(ConfigurationError, match="não está no índice"):
        provider.raw(manager, "datasets", "nope")


# ------------------------------------------------------------------ catalogue API
def test_catalogue_functions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager, _ = cache(tmp_path)
    monkeypatch.setitem(catalog.PROVIDERS, "m4", M4)
    providers = catalog.list_datasets()
    assert set(providers["provider"]) == {"synthetic", "m5", "m4", "m3", "rdatasets"}
    assert len(catalog.list_datasets("m4", subset="daily", cache=manager)) == 2
    descriptor = catalog.describe_dataset("m4", "daily", cache=manager)
    assert descriptor.horizon == 3
    assert descriptor.extra["n_series"] == 2
    loaded = catalog.load_benchmark("m4", "daily", series_id="D1", cache=manager)
    assert loaded.test.tolist() == [16, 17, 18]  # type: ignore[union-attr]
    raw = catalog.raw_dataset("rdatasets", package="datasets", name="iris", cache=manager)
    assert list(raw.columns) == ["rownames", "Sepal.Length", "Species"]
    with pytest.raises(ConfigurationError, match="Benchmark"):
        catalog.load_benchmark("m6", "daily", cache=manager)
    with pytest.raises(ConfigurationError, match="Provedor"):
        catalog.list_datasets("kaggle", cache=manager)
    with pytest.raises(ConfigurationError, match="search"):
        catalog.list_datasets("m4", search="x", cache=manager)


def test_get_data_names_benchmarks_and_rejects_unknown(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from autohybts.datasets import get_data

    manager, _ = cache(tmp_path)
    monkeypatch.setitem(catalog.PROVIDERS, "m4", M4)
    full = get_data("m4:D1", cache=manager)
    assert len(full) == 9
    with pytest.raises(ConfigurationError, match="m4:<id>"):
        get_data("airline")


# ------------------------------------------------------------------ calendars
def test_regular_index_falls_back_to_a_synthetic_calendar() -> None:
    index, source = regular_index("monthly", pd.Timestamp("1990-01-01"), 3)
    assert source == "source"
    assert index[-1] == pd.Timestamp("1990-03-01")
    _, source = regular_index("yearly", pd.Timestamp("2250-01-01"), 30)
    assert source == "synthetic"
    _, source = regular_index("daily", None, 5)
    assert source == "synthetic"
    hourly, _ = regular_index("hourly", pd.Timestamp("2015-07-01 12:00"), 3)
    assert hourly[1] - hourly[0] == pd.Timedelta(hours=1)
    with pytest.raises(DataValidationError):
        regular_index("yearly", None, 900)
    with pytest.raises(ConfigurationError):
        regular_index("fortnightly", None, 3)


@pytest.mark.network
def test_real_m4_series_downloads_and_loads(tmp_path: Path) -> None:
    from autohybts.datasets import load_benchmark

    series = load_benchmark("m4", "daily", series_id="D1", cache_dir=tmp_path)
    assert len(series.test) == 14  # type: ignore[union-attr]
    manifest = json.loads((tmp_path / "m4" / "Daily-test.csv.manifest.json").read_text())
    assert manifest["pinned"] is True


def test_m5_is_declared_unsupported_without_official_wrmsse() -> None:
    # D09: no partial M5 presented as the official benchmark.
    from autohybts.exceptions import UnsupportedCapabilityError

    for call in (
        lambda: catalog.load_benchmark("m5", "bottom"),
        lambda: catalog.list_datasets("m5"),
        lambda: catalog.load_dataset("m5"),
    ):
        with pytest.raises(UnsupportedCapabilityError, match="WRMSSE"):
            call()
    listed = catalog.list_datasets().set_index("provider")
    assert "WRMSSE" in listed.loc["m5", "kind"]
