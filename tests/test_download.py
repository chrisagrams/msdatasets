"""Tests for msdatasets.download and msdatasets.client."""

import sys
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from msdatasets.client import ensure_extracted, fetch_manifest
from msdatasets.download import (
    download_dataset,
    download_part,
    load_dataset,
)
from msdatasets.exceptions import DatasetNotFoundError, DownloadError, ExtractionError
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
            "extract_url": "/datasets/550e8400/parts/aaa",
            "download_url": "/transfer/files/aaa.mszx",
        },
        {
            "part_index": 1,
            "item_id": "bbb",
            "filename": "sample_02.mszx",
            "num_indices": 200,
            "extract_url": "/datasets/550e8400/parts/bbb",
            "download_url": "/transfer/files/bbb.mszx",
        },
    ],
}


def _make_part(**overrides):
    """Create a DatasetPart with sensible defaults."""
    defaults = {
        "part_index": 0,
        "item_id": "aaa",
        "filename": "test.mszx",
        "num_indices": 10,
        "extract_url": "/datasets/x/parts/aaa",
        "download_url": "/transfer/files/aaa.mszx",
    }
    defaults.update(overrides)
    return DatasetPart(**defaults)


class TestManifest:
    """Tests for the Manifest model."""

    def test_from_dict(self):
        m = Manifest.model_validate(SAMPLE_MANIFEST_DICT)
        assert m.dataset_id == "550e8400-e29b-41d4-a716-446655440000"
        assert m.dataset_name == "Test Dataset"
        assert m.total_parts == 2
        assert len(m.parts) == 2

    def test_from_dict_null_name(self):
        data = {**SAMPLE_MANIFEST_DICT, "dataset_name": None}
        m = Manifest.model_validate(data)
        assert m.dataset_name is None

    def test_parts_are_datasetpart(self):
        m = Manifest.model_validate(SAMPLE_MANIFEST_DICT)
        p = m.parts[0]
        assert isinstance(p, DatasetPart)
        assert p.part_index == 0
        assert p.item_id == "aaa"
        assert p.filename == "sample_01.mszx"
        assert p.num_indices == 100
        assert p.extract_url == "/datasets/550e8400/parts/aaa"
        assert p.download_url == "/transfer/files/aaa.mszx"


class TestDataset:
    """Tests for the Dataset model."""

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
    """Tests for the fetch_manifest function."""

    @pytest.mark.asyncio
    async def test_success(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = SAMPLE_MANIFEST_DICT

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response

        m = await fetch_manifest("550e8400", client=mock_client)
        assert m.dataset_id == "550e8400-e29b-41d4-a716-446655440000"
        assert len(m.parts) == 2

    @pytest.mark.asyncio
    async def test_404_raises_not_found(self):
        mock_response = MagicMock()
        mock_response.status_code = 404

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response

        with pytest.raises(DatasetNotFoundError, match="not found"):
            await fetch_manifest("bad-id", client=mock_client)

    @pytest.mark.asyncio
    async def test_500_raises_download_error(self):
        mock_response = MagicMock()
        mock_response.status_code = 500

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response

        with pytest.raises(DownloadError, match="500"):
            await fetch_manifest("some-id", client=mock_client)


class TestEnsureExtracted:
    """Tests for ensure_extracted."""

    @pytest.mark.asyncio
    @patch("msdatasets.client.get_api_url", return_value="https://api.example.com")
    async def test_204_already_cached(self, mock_api_url):
        mock_response = MagicMock()
        mock_response.status_code = 204

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response

        part = _make_part()
        await ensure_extracted(mock_client, part)  # Should return without error

        mock_client.get.assert_called_once_with(
            "https://api.example.com/datasets/x/parts/aaa"
        )

    @pytest.mark.asyncio
    @patch("msdatasets.client.stream_task", new_callable=AsyncMock)
    @patch("msdatasets.client.get_api_url", return_value="https://api.example.com")
    async def test_202_streams_task(self, mock_api_url, mock_stream):
        mock_response = MagicMock()
        mock_response.status_code = 202
        mock_response.json.return_value = {
            "task_id": "task-123",
            "status_url": "/tasks/task-123",
        }

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response

        part = _make_part()
        await ensure_extracted(mock_client, part)

        mock_stream.assert_called_once_with(mock_client, "task-123")

    @pytest.mark.asyncio
    @patch("msdatasets.client.get_api_url", return_value="https://api.example.com")
    async def test_404_raises(self, mock_api_url):
        mock_response = MagicMock()
        mock_response.status_code = 404

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response

        part = _make_part()
        with pytest.raises(DownloadError, match="not found"):
            await ensure_extracted(mock_client, part)

    @pytest.mark.asyncio
    @patch("msdatasets.client.get_api_url", return_value="https://api.example.com")
    async def test_500_raises(self, mock_api_url):
        mock_response = MagicMock()
        mock_response.status_code = 500

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response

        part = _make_part()
        with pytest.raises(DownloadError, match="500"):
            await ensure_extracted(mock_client, part)


class TestDownloadPart:
    """Tests for the download_part function."""

    @patch("msdatasets.download.get_api_url", return_value="https://api.example.com")
    @patch("msdatasets.download.ensure_extracted", new_callable=AsyncMock)
    @patch("msdatasets.download.download_file")
    def test_downloads_file(self, mock_dl_file, mock_ensure, mock_api_url, tmp_path):
        part = _make_part()
        expected = tmp_path / "test.mszx"
        mock_dl_file.return_value = expected

        result = download_part(part, tmp_path)

        assert result == expected
        mock_ensure.assert_called_once()
        mock_dl_file.assert_called_once_with(
            "https://api.example.com/transfer/files/aaa.mszx",
            expected,
            skip_existing=False,
            force=False,
        )

    @patch("msdatasets.download.get_api_url", return_value="https://api.example.com")
    @patch("msdatasets.download.ensure_extracted", new_callable=AsyncMock)
    @patch("msdatasets.download.download_file")
    def test_passes_skip_existing(
        self, mock_dl_file, mock_ensure, mock_api_url, tmp_path
    ):
        part = _make_part()
        mock_dl_file.return_value = tmp_path / "test.mszx"

        download_part(part, tmp_path, skip_existing=True)

        mock_dl_file.assert_called_once_with(
            "https://api.example.com/transfer/files/aaa.mszx",
            tmp_path / "test.mszx",
            skip_existing=True,
            force=False,
        )

    @patch("msdatasets.download.get_api_url", return_value="https://api.example.com")
    @patch("msdatasets.download.ensure_extracted", new_callable=AsyncMock)
    @patch("msdatasets.download.download_file")
    def test_passes_force(self, mock_dl_file, mock_ensure, mock_api_url, tmp_path):
        part = _make_part()
        mock_dl_file.return_value = tmp_path / "test.mszx"

        download_part(part, tmp_path, force=True)

        mock_dl_file.assert_called_once_with(
            "https://api.example.com/transfer/files/aaa.mszx",
            tmp_path / "test.mszx",
            skip_existing=False,
            force=True,
        )

    @patch("msdatasets.download.get_api_url", return_value="https://api.example.com")
    @patch("msdatasets.download.ensure_extracted", new_callable=AsyncMock)
    @patch("msdatasets.download.download_file")
    def test_creates_dest_dir(self, mock_dl_file, mock_ensure, mock_api_url, tmp_path):
        part = _make_part()
        dest_dir = tmp_path / "nested" / "dir"
        mock_dl_file.return_value = dest_dir / "test.mszx"

        download_part(part, dest_dir)

        assert dest_dir.exists()


class TestDownloadDataset:
    """Tests for the download_dataset function."""

    @patch("msdatasets.download.get_api_url", return_value="https://api.example.com")
    @patch("msdatasets.download.get_dataset_dir")
    @patch("msdatasets.download.fetch_manifest", new_callable=AsyncMock)
    @patch("msdatasets.download.ensure_all_extracted", new_callable=AsyncMock)
    @patch("msdatasets.download.download_batch")
    def test_downloads_all_parts(
        self, mock_batch, mock_ensure, mock_fetch, mock_dir, mock_api_url, tmp_path
    ):
        ds_dir = tmp_path / "ds"
        mock_dir.return_value = ds_dir
        manifest = Manifest.model_validate(SAMPLE_MANIFEST_DICT)
        mock_fetch.return_value = manifest

        mock_batch.return_value = [
            ds_dir / "sample_01.mszx",
            ds_dir / "sample_02.mszx",
        ]

        ds = download_dataset("550e8400", show_progress=False)

        assert len(ds) == 2
        assert ds.dataset_name == "Test Dataset"
        # Extraction should have been triggered
        mock_ensure.assert_called_once()
        # Downloads go through mstransfer
        mock_batch.assert_called_once()
        requests = mock_batch.call_args[0][0]
        assert len(requests) == 2
        assert requests[0].url == "https://api.example.com/transfer/files/aaa.mszx"
        assert requests[0].dest == ds_dir / "sample_01.mszx"
        assert requests[1].url == "https://api.example.com/transfer/files/bbb.mszx"
        assert requests[1].dest == ds_dir / "sample_02.mszx"

    @patch("msdatasets.download.get_api_url", return_value="https://api.example.com")
    @patch("msdatasets.download.get_dataset_dir")
    @patch("msdatasets.download.fetch_manifest", new_callable=AsyncMock)
    @patch("msdatasets.download.ensure_all_extracted", new_callable=AsyncMock)
    @patch("msdatasets.download.download_batch")
    def test_skips_existing_files(
        self, mock_batch, mock_ensure, mock_fetch, mock_dir, mock_api_url, tmp_path
    ):
        ds_dir = tmp_path / "ds"
        ds_dir.mkdir()
        mock_dir.return_value = ds_dir
        manifest = Manifest.model_validate(SAMPLE_MANIFEST_DICT)
        mock_fetch.return_value = manifest

        # Pre-create one file
        (ds_dir / "sample_01.mszx").write_bytes(b"existing")

        mock_batch.return_value = [ds_dir / "sample_02.mszx"]

        ds = download_dataset("550e8400", show_progress=False)

        assert len(ds) == 2
        # Extraction should have been triggered for the missing part
        mock_ensure.assert_called_once()
        mock_batch.assert_called_once()
        requests = mock_batch.call_args[0][0]
        assert len(requests) == 1
        assert requests[0].dest == ds_dir / "sample_02.mszx"

    @patch("msdatasets.download.get_api_url", return_value="https://api.example.com")
    @patch("msdatasets.download.get_dataset_dir")
    @patch("msdatasets.download.fetch_manifest", new_callable=AsyncMock)
    @patch("msdatasets.download.ensure_all_extracted", new_callable=AsyncMock)
    @patch("msdatasets.download.download_batch")
    def test_force_redownloads(
        self, mock_batch, mock_ensure, mock_fetch, mock_dir, mock_api_url, tmp_path
    ):
        ds_dir = tmp_path / "ds"
        ds_dir.mkdir()
        mock_dir.return_value = ds_dir
        manifest = Manifest.model_validate(SAMPLE_MANIFEST_DICT)
        mock_fetch.return_value = manifest

        (ds_dir / "sample_01.mszx").write_bytes(b"existing")

        mock_batch.return_value = [
            ds_dir / "sample_01.mszx",
            ds_dir / "sample_02.mszx",
        ]

        ds = download_dataset("550e8400", force_download=True, show_progress=False)

        assert len(ds) == 2
        # Both parts should be extracted and requested despite one existing
        mock_ensure.assert_called_once()
        requests = mock_batch.call_args[0][0]
        assert len(requests) == 2

    @patch("msdatasets.download.get_api_url", return_value="https://api.example.com")
    @patch("msdatasets.download.get_dataset_dir")
    @patch("msdatasets.download.fetch_manifest", new_callable=AsyncMock)
    @patch("msdatasets.download.download_batch")
    def test_all_cached_skips_download(
        self, mock_batch, mock_fetch, mock_dir, mock_api_url, tmp_path
    ):
        ds_dir = tmp_path / "ds"
        ds_dir.mkdir()
        mock_dir.return_value = ds_dir
        manifest = Manifest.model_validate(SAMPLE_MANIFEST_DICT)
        mock_fetch.return_value = manifest

        # Pre-create all files
        (ds_dir / "sample_01.mszx").write_bytes(b"existing")
        (ds_dir / "sample_02.mszx").write_bytes(b"existing")

        ds = download_dataset("550e8400", show_progress=False)

        assert len(ds) == 2
        mock_batch.assert_not_called()

    @patch("msdatasets.download.get_api_url", return_value="https://api.example.com")
    @patch("msdatasets.download.get_dataset_dir")
    @patch("msdatasets.download.fetch_manifest", new_callable=AsyncMock)
    @patch("msdatasets.download.ensure_all_extracted", new_callable=AsyncMock)
    @patch("msdatasets.download.download_batch")
    def test_saves_manifest_json(
        self, mock_batch, mock_ensure, mock_fetch, mock_dir, mock_api_url, tmp_path
    ):
        ds_dir = tmp_path / "ds"
        mock_dir.return_value = ds_dir
        manifest = Manifest.model_validate(SAMPLE_MANIFEST_DICT)
        mock_fetch.return_value = manifest
        mock_batch.return_value = [
            ds_dir / "sample_01.mszx",
            ds_dir / "sample_02.mszx",
        ]

        download_dataset("550e8400", show_progress=False)

        manifest_file = ds_dir / "manifest.json"
        assert manifest_file.exists()
        import json

        saved = json.loads(manifest_file.read_text())
        assert saved["dataset_id"] == "550e8400-e29b-41d4-a716-446655440000"
        assert saved["total_parts"] == 2
        assert saved["parts"][0]["extract_url"] == "/datasets/550e8400/parts/aaa"
        assert saved["parts"][0]["download_url"] == "/transfer/files/aaa.mszx"


class TestLoadDataset:
    """Tests for the load_dataset function (returns MSCompressDataset)."""

    @patch("msdatasets.download.download_dataset")
    def test_returns_mscompress_dataset(self, mock_download):
        mock_ds = MagicMock()
        mock_ds.cache_dir = "/tmp/ds"
        mock_download.return_value = mock_ds

        mock_msc_cls = MagicMock()
        sentinel = MagicMock()
        mock_msc_cls.return_value = sentinel

        fake_module = ModuleType("mscompress.datasets.torch")
        fake_module.MSCompressDataset = mock_msc_cls

        with patch.dict(sys.modules, {"mscompress.datasets.torch": fake_module}):
            result = load_dataset(
                "abc-123", force_download=True, show_progress=False
            )

        mock_download.assert_called_once_with(
            "abc-123",
            force_download=True,
            show_progress=False,
            max_workers=4,
        )
        mock_msc_cls.assert_called_once_with("/tmp/ds")
        assert result is sentinel
