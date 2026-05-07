"""Tests for msdatasets.cli."""

import logging
from unittest.mock import patch

import pytest

from msdatasets.cli import _configure_logging, main
from msdatasets.exceptions import DatasetNotFoundError, DownloadError
from msdatasets.models import Dataset, RepoSource


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
            "abc",
            force_download=False,
            show_progress=True,
            max_workers=4,
            store_as="mszx",
            output_dir=None,
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
            "abc",
            force_download=True,
            show_progress=True,
            max_workers=4,
            store_as="mszx",
            output_dir=None,
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
            "abc",
            force_download=False,
            show_progress=False,
            max_workers=4,
            store_as="mszx",
            output_dir=None,
        )

    @patch("msdatasets.cli.download_dataset")
    def test_store_as_flag(self, mock_load, tmp_path):
        mock_load.return_value = Dataset(
            dataset_id="abc",
            dataset_name=None,
            cache_dir=tmp_path,
            files=[],
        )

        main(["download", "abc", "--store-as", "mzml"])
        assert mock_load.call_args.kwargs["store_as"] == "mzml"

    def test_store_as_rejects_invalid_value(self):
        # argparse exits with code 2 on invalid choice.
        with pytest.raises(SystemExit) as exc:
            main(["download", "abc", "--store-as", "bogus"])
        assert exc.value.code == 2

    @patch("msdatasets.cli.download_dataset")
    def test_output_flag(self, mock_load, tmp_path):
        mock_load.return_value = Dataset(
            dataset_id="abc",
            dataset_name=None,
            cache_dir=tmp_path,
            files=[],
        )

        target = tmp_path / "custom-out"
        main(["download", "abc", "-o", str(target)])
        assert mock_load.call_args.kwargs["output_dir"] == target

    @patch("msdatasets.cli.download_dataset")
    def test_output_long_flag(self, mock_load, tmp_path):
        mock_load.return_value = Dataset(
            dataset_id="abc",
            dataset_name=None,
            cache_dir=tmp_path,
            files=[],
        )

        target = tmp_path / "custom-out"
        main(["download", "abc", "--output", str(target)])
        assert mock_load.call_args.kwargs["output_dir"] == target

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


class TestDownloadRepoSpec:
    """Tests for CLI dispatch on repository specs (pride/MSV...)."""

    @patch("msdatasets.cli.download_repo_dataset")
    @patch("msdatasets.cli.download_dataset")
    def test_pride_accession_dispatches_to_repo(
        self, mock_download, mock_repo, tmp_path
    ):
        mock_repo.return_value = Dataset(
            dataset_id="ds-xyz",
            dataset_name="Repo",
            cache_dir=tmp_path,
            files=[],
        )
        result = main(["download", "pride/PXD075509"])
        assert result == 0
        mock_download.assert_not_called()
        mock_repo.assert_called_once_with(
            RepoSource.PRIDE,
            "PXD075509",
            filenames=None,
            force_download=False,
            show_progress=True,
            max_workers=4,
            store_as="mszx",
            output_dir=None,
        )

    @patch("msdatasets.cli.download_repo_dataset")
    def test_massive_accession_with_filenames(self, mock_repo, tmp_path):
        mock_repo.return_value = Dataset(
            dataset_id="ds-xyz",
            dataset_name=None,
            cache_dir=tmp_path,
            files=[],
        )
        main(["download", "massive/MSV000101460[a.raw, b.raw]"])
        mock_repo.assert_called_once_with(
            RepoSource.MASSIVE,
            "MSV000101460",
            filenames=["a.raw", "b.raw"],
            force_download=False,
            show_progress=True,
            max_workers=4,
            store_as="mszx",
            output_dir=None,
        )

    @patch("msdatasets.cli.download_repo_dataset")
    def test_store_as_forwarded_for_repo_spec(self, mock_repo, tmp_path):
        mock_repo.return_value = Dataset(
            dataset_id="ds-xyz",
            dataset_name=None,
            cache_dir=tmp_path,
            files=[],
        )
        main(["download", "pride/PXD075509", "--store-as", "msz"])
        assert mock_repo.call_args.kwargs["store_as"] == "msz"

    @patch("msdatasets.cli.download_repo_dataset")
    def test_repo_not_found_returns_1(self, mock_repo):
        mock_repo.side_effect = DatasetNotFoundError("nope")
        assert main(["download", "pride/PXD999999"]) == 1

    @patch("msdatasets.cli.download_repo_dataset")
    def test_repo_download_error_returns_1(self, mock_repo):
        mock_repo.side_effect = DownloadError("boom")
        assert main(["download", "pride/PXD075509"]) == 1


class TestDownloadHfSpec:
    """Tests for CLI dispatch on HuggingFace specs (hf/owner/repo...)."""

    @patch("msdatasets.cli.download_hf_dataset")
    @patch("msdatasets.cli.download_dataset")
    @patch("msdatasets.cli.download_repo_dataset")
    def test_hf_spec_dispatches_to_hf(
        self, mock_repo, mock_download, mock_hf, tmp_path
    ):
        mock_hf.return_value = Dataset(
            dataset_id="alice/bench",
            dataset_name="alice/bench",
            cache_dir=tmp_path,
            files=[],
        )
        result = main(["download", "hf/alice/bench"])
        assert result == 0
        mock_download.assert_not_called()
        mock_repo.assert_not_called()
        mock_hf.assert_called_once_with(
            "alice/bench",
            filenames=None,
            force_download=False,
            show_progress=True,
            output_dir=None,
        )

    @patch("msdatasets.cli.download_hf_dataset")
    def test_hf_spec_with_filenames(self, mock_hf, tmp_path):
        mock_hf.return_value = Dataset(
            dataset_id="alice/bench",
            dataset_name=None,
            cache_dir=tmp_path,
            files=[],
        )
        main(["download", "hf/alice/bench[a.mszx, b.mszx]", "--no-progress"])
        mock_hf.assert_called_once_with(
            "alice/bench",
            filenames=["a.mszx", "b.mszx"],
            force_download=False,
            show_progress=False,
            output_dir=None,
        )

    @patch("msdatasets.cli.download_hf_dataset")
    def test_hf_spec_with_output_dir(self, mock_hf, tmp_path):
        mock_hf.return_value = Dataset(
            dataset_id="alice/bench",
            dataset_name=None,
            cache_dir=tmp_path,
            files=[],
        )
        target = tmp_path / "out"
        main(["download", "hf/alice/bench", "-o", str(target)])
        assert mock_hf.call_args.kwargs["output_dir"] == target

    @patch("msdatasets.cli.download_hf_dataset")
    def test_hf_spec_rejects_store_as_msz(self, mock_hf, capsys):
        result = main(["download", "hf/alice/bench", "--store-as", "msz"])
        assert result == 1
        mock_hf.assert_not_called()
        captured = capsys.readouterr()
        assert "--store-as" in captured.err or "--store-as" in captured.out

    @patch("msdatasets.cli.download_hf_dataset")
    def test_hf_spec_rejects_store_as_mzml(self, mock_hf):
        result = main(["download", "hf/alice/bench", "--store-as", "mzml"])
        assert result == 1
        mock_hf.assert_not_called()

    @patch("msdatasets.cli.download_hf_dataset")
    def test_hf_not_found_returns_1(self, mock_hf):
        mock_hf.side_effect = DatasetNotFoundError("nope")
        assert main(["download", "hf/alice/missing"]) == 1

    @patch("msdatasets.cli.download_hf_dataset")
    def test_hf_download_error_returns_1(self, mock_hf):
        mock_hf.side_effect = DownloadError("boom")
        assert main(["download", "hf/alice/bench"]) == 1


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
