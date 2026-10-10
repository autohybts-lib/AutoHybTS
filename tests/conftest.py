from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import settings

# Property tests check correctness, not speed: under coverage on a loaded machine a single
# example can exceed Hypothesis' default 200 ms deadline and fail spuriously.
settings.register_profile("autohybts", deadline=None)
settings.load_profile("autohybts")

SUITES = ("unit", "integration", "contracts", "leakage", "regression")


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-network",
        action="store_true",
        default=False,
        help="run tests marked 'network' (CI never does)",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    run_network = config.getoption("--run-network")
    skip_network = pytest.mark.skip(reason="needs --run-network")
    tests_root = Path(__file__).parent
    for item in items:
        # The suite marker comes from the folder, so `-m leakage` selects tests/leakage.
        parts = Path(item.path).relative_to(tests_root).parts
        if parts and parts[0] in SUITES:
            item.add_marker(getattr(pytest.mark, parts[0]))
        if "network" in item.keywords and not run_network:
            item.add_marker(skip_network)
