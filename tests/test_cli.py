"""Tests for msdatasets.cli."""

import logging
from unittest.mock import patch

import pytest

from msdatasets.cli import _configure_logging, main
from msdatasets.exceptions import DatasetNotFoundError, DownloadError
from msdatasets.models import Dataset


@pytest.fixture(autouse=True)
def _reset_msdatasets_logger():
    """Remove handlers added by _configure_logging so tests stay isolated."""
    logger = logging.getLogger("msdatasets")
    original_handlers = list(logger.handlers)
    original_level = logger.level
    yield
    logger.handlers = original_handlers
    logger.setLevel(original_level)


class TestMain:
    """
    Tests for the main CLI function.
    """

    def test_no_args_prints_help(self, capsys):
        assert main([]) == 0

    def test_download_requires_dataset_id(self):
        # argparse exits with code 2 for missing required args
        try:
            main(["download"])
            assert False, "Should have raised SystemExit"
        except SystemExit as e:
            assert e.code == 2


class TestDownloadCommand:
    """
    Tests for the download command.
    """

    @patch("msdatasets.cli.download_dataset")
    def test_success(self, mock_load, tmp_path):
        mock_load.return_value = Dataset(
            dataset_id="abc",
            dataset_name="My Dataset",
            cache_dir=tmp_path,
            files=[tmp_path / "a.mszx"],
        )

        result = main(["download", "abc"])
        assert result == 0
        mock_load.assert_called_once_with(
            "abc", force_download=False, show_progress=True, max_workers=4
        )

    @patch("msdatasets.cli.download_dataset")
    def test_force_flag(self, mock_load, tmp_path):
        mock_load.return_value = Dataset(
            dataset_id="abc",
            dataset_name=None,
            cache_dir=tmp_path,
            files=[],
        )

        main(["download", "abc", "--force"])
        mock_load.assert_called_once_with(
            "abc", force_download=True, show_progress=True, max_workers=4
        )

    @patch("msdatasets.cli.download_dataset")
    def test_no_progress_flag(self, mock_load, tmp_path):
        mock_load.return_value = Dataset(
            dataset_id="abc",
            dataset_name=None,
            cache_dir=tmp_path,
            files=[],
        )

        main(["download", "abc", "--no-progress"])
        mock_load.assert_called_once_with(
            "abc", force_download=False, show_progress=False, max_workers=4
        )

    @patch("msdatasets.cli.download_dataset")
    def test_not_found_returns_1(self, mock_load):
        mock_load.side_effect = DatasetNotFoundError("not found")

        result = main(["download", "bad-id"])
        assert result == 1

    @patch("msdatasets.cli.download_dataset")
    def test_download_error_returns_1(self, mock_load):
        mock_load.side_effect = DownloadError("server error")

        result = main(["download", "some-id"])
        assert result == 1

    @patch("msdatasets.cli.download_dataset")
    def test_fallback_to_dataset_id_when_no_name(self, mock_load, tmp_path):
        mock_load.return_value = Dataset(
            dataset_id="abc-123",
            dataset_name=None,
            cache_dir=tmp_path,
            files=[],
        )

        result = main(["download", "abc-123"])
        assert result == 0


class TestConfigureLogging:
    """Tests for _configure_logging verbosity → log-level mapping."""

    def test_verbosity_zero_is_warning(self):
        _configure_logging(0)
        logger = logging.getLogger("msdatasets")
        assert logger.level == logging.WARNING
        assert len(logger.handlers) >= 1

    def test_verbosity_one_is_info(self):
        _configure_logging(1)
        assert logging.getLogger("msdatasets").level == logging.INFO

    def test_verbosity_two_is_debug(self):
        _configure_logging(2)
        assert logging.getLogger("msdatasets").level == logging.DEBUG

    def test_verbosity_above_two_is_debug(self):
        _configure_logging(5)
        assert logging.getLogger("msdatasets").level == logging.DEBUG


class TestMainVerboseFlag:
    """End-to-end: -v flag wires through to the logger."""

    @patch("msdatasets.cli.download_dataset")
    def test_verbose_flag_sets_info(self, mock_load, tmp_path):
        mock_load.return_value = Dataset(
            dataset_id="abc",
            dataset_name=None,
            cache_dir=tmp_path,
            files=[],
        )
        main(["-v", "download", "abc"])
        assert logging.getLogger("msdatasets").level == logging.INFO
