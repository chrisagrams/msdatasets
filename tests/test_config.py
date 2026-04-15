"""Tests for msdatasets.config."""

from pathlib import Path
from unittest.mock import patch

from msdatasets.config import get_api_url, get_cache_dir, get_dataset_dir


class TestGetApiUrl:
    """
    Tests for the get_api_url function.
    """

    def test_default(self):
        with patch.dict("os.environ", {}, clear=True):
            assert get_api_url() == "https://datasets.lab.gy"

    def test_env_override(self):
        with patch.dict("os.environ", {"MS_API_URL": "http://localhost:8000"}):
            assert get_api_url() == "http://localhost:8000"


class TestGetCacheDir:
    """
    Tests for the get_cache_dir function.
    """

    def test_default(self):
        with patch.dict("os.environ", {}, clear=True):
            result = get_cache_dir()
            assert result == Path.home() / ".ms" / "datasets"

    def test_ms_home(self):
        with patch.dict("os.environ", {"MS_HOME": "/custom/ms"}, clear=True):
            assert get_cache_dir() == Path("/custom/ms/datasets")

    def test_ms_datasets_cache(self):
        with patch.dict(
            "os.environ", {"MS_DATASETS_CACHE": "/direct/cache"}, clear=True
        ):
            assert get_cache_dir() == Path("/direct/cache")

    def test_ms_datasets_cache_takes_priority(self):
        with patch.dict(
            "os.environ",
            {
                "MS_DATASETS_CACHE": "/direct/cache",
                "MS_HOME": "/custom/ms",
            },
            clear=True,
        ):
            assert get_cache_dir() == Path("/direct/cache")


class TestGetDatasetDir:
    """
    Tests for the get_dataset_dir function.
    """

    def test_path_construction(self):
        with patch.dict("os.environ", {}, clear=True):
            result = get_dataset_dir("abc-123")
            assert result == Path.home() / ".ms" / "datasets" / "abc-123"
