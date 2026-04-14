"""A unified dataset framework for mass spectrometry."""

__version__ = "0.1.0"

from msdatasets.download import download_dataset, load_dataset, load_repo_dataset
from msdatasets.exceptions import DatasetNotFoundError, DownloadError, ExtractionError
from msdatasets.models import Dataset, RepoSource

__all__ = [
    "Dataset",
    "DatasetNotFoundError",
    "DownloadError",
    "ExtractionError",
    "RepoSource",
    "download_dataset",
    "load_dataset",
    "load_repo_dataset",
    "__version__",
]
