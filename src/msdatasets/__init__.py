"""A unified dataset framework for mass spectrometry."""

__version__ = "0.1.0"

from msdatasets.download import download_dataset, load_dataset
from msdatasets.exceptions import DatasetNotFoundError, DownloadError, ExtractionError
from msdatasets.models import Dataset

__all__ = [
    "Dataset",
    "DatasetNotFoundError",
    "DownloadError",
    "ExtractionError",
    "download_dataset",
    "load_dataset",
    "__version__",
]
