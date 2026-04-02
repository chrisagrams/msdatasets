"""Tests for msdatasets.download."""

from unittest.mock import MagicMock, patch

import pytest

from msdatasets.download import download_part, fetch_manifest, load_dataset
from msdatasets.exceptions import DatasetNotFoundError, DownloadError
from msdatasets.models import Dataset, DatasetPart, Manifest

# Sample manifest dictionary for testing
SAMPLE_MANIFEST_DICT = {
    "dataset_id": "550e8400-e29b-41d4-a716-446655440000",
    "dataset_name": "Test Dataset",
    "total_parts": 2,
    "parts": [
        {
            "part_index": 0,
            "item_id": "aaa",
            "filename": "sample_01.mszx",
            "num_indices": 100,
            "download_url": "/datasets/550e8400/parts/aaa",
        },
        {
            "part_index": 1,
            "item_id": "bbb",
            "filename": "sample_02.mszx",
            "num_indices": 200,
            "download_url": "/datasets/550e8400/parts/bbb",
        },
    ],
}


class TestManifest:
    """
    Tests for the Manifest model.
    """

    def test_from_dict(self):
        m = Manifest.from_dict(SAMPLE_MANIFEST_DICT)
        assert m.dataset_id == "550e8400-e29b-41d4-a716-446655440000"
        assert m.dataset_name == "Test Dataset"
        assert m.total_parts == 2
        assert len(m.parts) == 2

    def test_from_dict_null_name(self):
        data = {**SAMPLE_MANIFEST_DICT, "dataset_name": None}
        m = Manifest.from_dict(data)
        assert m.dataset_name is None

    def test_parts_are_datasetpart(self):
        m = Manifest.from_dict(SAMPLE_MANIFEST_DICT)
        p = m.parts[0]
        assert isinstance(p, DatasetPart)
        assert p.part_index == 0
        assert p.item_id == "aaa"
        assert p.filename == "sample_01.mszx"
        assert p.num_indices == 100


class TestDataset:
    """
    Tests for the Dataset model.
    """

    def test_len(self, tmp_path):
        ds = Dataset(
            dataset_id="x",
            dataset_name="X",
            cache_dir=tmp_path,
            files=[tmp_path / "a", tmp_path / "b"],
        )
        assert len(ds) == 2

    def test_indexing(self, tmp_path):
        f = tmp_path / "a"
        ds = Dataset(
            dataset_id="x",
            dataset_name=None,
            cache_dir=tmp_path,
            files=[f],
        )
        assert ds[0] == f

    def test_iteration(self, tmp_path):
        files = [tmp_path / "a", tmp_path / "b"]
        ds = Dataset(
            dataset_id="x",
            dataset_name=None,
            cache_dir=tmp_path,
            files=files,
        )
        assert list(ds) == files


class TestFetchManifest:
    """
    Tests for the fetch_manifest function.
    """

    def test_success(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = SAMPLE_MANIFEST_DICT

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response

        m = fetch_manifest("550e8400", client=mock_client)
        assert m.dataset_id == "550e8400-e29b-41d4-a716-446655440000"
        assert len(m.parts) == 2

    def test_404_raises_not_found(self):
        mock_response = MagicMock()
        mock_response.status_code = 404

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response

        with pytest.raises(DatasetNotFoundError, match="not found"):
            fetch_manifest("bad-id", client=mock_client)

    def test_500_raises_download_error(self):
        mock_response = MagicMock()
        mock_response.status_code = 500

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response

        with pytest.raises(DownloadError, match="500"):
            fetch_manifest("some-id", client=mock_client)


class TestDownloadPart:
    """
    Tests for the download_part function.
    """

    @patch("msdatasets.download.get_api_url", return_value="https://api.example.com")
    @patch("msdatasets.download.download_file")
    def test_downloads_file(self, mock_dl_file, mock_api_url, tmp_path):
        part = DatasetPart(
            part_index=0,
            item_id="aaa",
            filename="test.mszx",
            num_indices=10,
            download_url="/datasets/x/parts/aaa",
        )
        expected = tmp_path / "test.mszx"
        mock_dl_file.return_value = expected

        result = download_part("x", part, tmp_path)

        assert result == expected
        mock_dl_file.assert_called_once_with(
            "https://api.example.com/datasets/x/parts/aaa",
            expected,
            skip_existing=False,
            force=False,
        )

    @patch("msdatasets.download.get_api_url", return_value="https://api.example.com")
    @patch("msdatasets.download.download_file")
    def test_passes_skip_existing(self, mock_dl_file, mock_api_url, tmp_path):
        part = DatasetPart(
            part_index=0,
            item_id="aaa",
            filename="test.mszx",
            num_indices=10,
            download_url="/datasets/x/parts/aaa",
        )
        mock_dl_file.return_value = tmp_path / "test.mszx"

        download_part("x", part, tmp_path, skip_existing=True)

        mock_dl_file.assert_called_once_with(
            "https://api.example.com/datasets/x/parts/aaa",
            tmp_path / "test.mszx",
            skip_existing=True,
            force=False,
        )

    @patch("msdatasets.download.get_api_url", return_value="https://api.example.com")
    @patch("msdatasets.download.download_file")
    def test_passes_force(self, mock_dl_file, mock_api_url, tmp_path):
        part = DatasetPart(
            part_index=0,
            item_id="aaa",
            filename="test.mszx",
            num_indices=10,
            download_url="/datasets/x/parts/aaa",
        )
        mock_dl_file.return_value = tmp_path / "test.mszx"

        download_part("x", part, tmp_path, force=True)

        mock_dl_file.assert_called_once_with(
            "https://api.example.com/datasets/x/parts/aaa",
            tmp_path / "test.mszx",
            skip_existing=False,
            force=True,
        )

    @patch("msdatasets.download.get_api_url", return_value="https://api.example.com")
    @patch("msdatasets.download.download_file")
    def test_creates_dest_dir(self, mock_dl_file, mock_api_url, tmp_path):
        part = DatasetPart(
            part_index=0,
            item_id="aaa",
            filename="test.mszx",
            num_indices=10,
            download_url="/datasets/x/parts/aaa",
        )
        dest_dir = tmp_path / "nested" / "dir"
        mock_dl_file.return_value = dest_dir / "test.mszx"

        download_part("x", part, dest_dir)

        assert dest_dir.exists()


class TestLoadDataset:
    """
    Tests for the load_dataset function.
    """

    @patch("msdatasets.download.get_api_url", return_value="https://api.example.com")
    @patch("msdatasets.download.get_dataset_dir")
    @patch("msdatasets.download.fetch_manifest")
    @patch("msdatasets.download.download_batch")
    def test_downloads_all_parts(
        self, mock_batch, mock_fetch, mock_dir, mock_api_url, tmp_path
    ):
        ds_dir = tmp_path / "ds"
        mock_dir.return_value = ds_dir
        manifest = Manifest.from_dict(SAMPLE_MANIFEST_DICT)
        mock_fetch.return_value = manifest

        mock_batch.return_value = [
            ds_dir / "sample_01.mszx",
            ds_dir / "sample_02.mszx",
        ]

        ds = load_dataset("550e8400", show_progress=False)

        assert len(ds) == 2
        assert ds.dataset_name == "Test Dataset"
        mock_batch.assert_called_once()
        requests = mock_batch.call_args[0][0]
        assert len(requests) == 2
        assert requests[0].url == "https://api.example.com/datasets/550e8400/parts/aaa"
        assert requests[0].dest == ds_dir / "sample_01.mszx"
        assert requests[1].url == "https://api.example.com/datasets/550e8400/parts/bbb"
        assert requests[1].dest == ds_dir / "sample_02.mszx"

    @patch("msdatasets.download.get_api_url", return_value="https://api.example.com")
    @patch("msdatasets.download.get_dataset_dir")
    @patch("msdatasets.download.fetch_manifest")
    @patch("msdatasets.download.download_batch")
    def test_skips_existing_files(
        self, mock_batch, mock_fetch, mock_dir, mock_api_url, tmp_path
    ):
        ds_dir = tmp_path / "ds"
        ds_dir.mkdir()
        mock_dir.return_value = ds_dir
        manifest = Manifest.from_dict(SAMPLE_MANIFEST_DICT)
        mock_fetch.return_value = manifest

        # Pre-create one file
        (ds_dir / "sample_01.mszx").write_bytes(b"existing")

        mock_batch.return_value = [ds_dir / "sample_02.mszx"]

        ds = load_dataset("550e8400", show_progress=False)

        assert len(ds) == 2
        # Only the missing part should be in the batch request
        mock_batch.assert_called_once()
        requests = mock_batch.call_args[0][0]
        assert len(requests) == 1
        assert requests[0].dest == ds_dir / "sample_02.mszx"

    @patch("msdatasets.download.get_api_url", return_value="https://api.example.com")
    @patch("msdatasets.download.get_dataset_dir")
    @patch("msdatasets.download.fetch_manifest")
    @patch("msdatasets.download.download_batch")
    def test_force_redownloads(
        self, mock_batch, mock_fetch, mock_dir, mock_api_url, tmp_path
    ):
        ds_dir = tmp_path / "ds"
        ds_dir.mkdir()
        mock_dir.return_value = ds_dir
        manifest = Manifest.from_dict(SAMPLE_MANIFEST_DICT)
        mock_fetch.return_value = manifest

        (ds_dir / "sample_01.mszx").write_bytes(b"existing")

        mock_batch.return_value = [
            ds_dir / "sample_01.mszx",
            ds_dir / "sample_02.mszx",
        ]

        ds = load_dataset("550e8400", force_download=True, show_progress=False)

        assert len(ds) == 2
        # Both parts should be requested despite one existing
        requests = mock_batch.call_args[0][0]
        assert len(requests) == 2
