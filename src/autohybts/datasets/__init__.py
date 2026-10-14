"""Example and benchmark datasets (§32).

Offline synthetic series, and M4, M3 and Rdatasets downloaded on request into a local
cache with integrity checks. Nothing is downloaded on import.
"""

from autohybts.datasets.cache import CacheManager
from autohybts.datasets.catalog import (
    describe_dataset,
    list_datasets,
    load_benchmark,
    load_dataset,
    raw_dataset,
)
from autohybts.datasets.descriptor import BenchmarkDataset, BenchmarkSeries, DatasetDescriptor
from autohybts.datasets.exogenous import EXOG_DATASETS, list_exog_datasets, load_exog_dataset
from autohybts.datasets.named import NAMED_SERIES, NamedSeries
from autohybts.datasets.synthetic import SYNTHETIC, get_data

__all__ = [
    "EXOG_DATASETS",
    "NAMED_SERIES",
    "SYNTHETIC",
    "BenchmarkDataset",
    "BenchmarkSeries",
    "CacheManager",
    "DatasetDescriptor",
    "NamedSeries",
    "describe_dataset",
    "get_data",
    "list_datasets",
    "list_exog_datasets",
    "load_benchmark",
    "load_dataset",
    "load_exog_dataset",
    "raw_dataset",
]
