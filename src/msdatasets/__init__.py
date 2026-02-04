"""A unified dataset framework for mass spectrometry."""

__version__ = "0.1.0"

from msdatasets.download import load_dataset
from msdatasets.exceptions import DatasetNotFoundError, DownloadError
from msdatasets.models import Dataset

__all__ = [
    "Dataset",
    "DatasetNotFoundError",
    "DownloadError",
    "load_dataset",
    "__version__",
]
