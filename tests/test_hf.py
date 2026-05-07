"""Tests for msdatasets.hf and the HuggingFace dispatch path."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from msdatasets.download import _parse_hf_spec
from msdatasets.exceptions import DatasetNotFoundError, DownloadError
from msdatasets.hf import (
    _MS_PATTERNS,
    _hf_dataset_dir,
    _hf_progress_disabled,
    _import_hf_hub,
    download_hf_dataset,
    load_hf_dataset,
)


class _FakeHfErrors:
    """Stand-in exception classes that mirror huggingface_hub.errors."""

    class RepositoryNotFoundError(Exception):
        pass

    class RevisionNotFoundError(Exception):
        pass

    class HfHubHTTPError(Exception):
        pass


def _patch_hf_hub(snapshot_side_effect):
    """Patch ``_import_hf_hub`` so callers don't need huggingface_hub installed."""
    snapshot = MagicMock(side_effect=snapshot_side_effect)
    return (
        patch(
            "msdatasets.hf._import_hf_hub",
            return_value=(
                snapshot,
                _FakeHfErrors.RepositoryNotFoundError,
                _FakeHfErrors.RevisionNotFoundError,
                _FakeHfErrors.HfHubHTTPError,
            ),
        ),
        snapshot,
    )


class TestParseHfSpec:
    """Tests for _parse_hf_spec."""

    def test_owner_repo(self):
        assert _parse_hf_spec("hf/myorg/proteomics-bench") == (
            "myorg/proteomics-bench",
            None,
        )

    def test_owner_repo_with_files(self):
        assert _parse_hf_spec("hf/myorg/proteomics-bench[a.mszx, b.mszx]") == (
            "myorg/proteomics-bench",
            ["a.mszx", "b.mszx"],
        )

    def test_single_segment_rejected(self):
        assert _parse_hf_spec("hf/myorg") is None

    def test_three_segment_rejected(self):
        assert _parse_hf_spec("hf/myorg/repo/extra") is None

    def test_unmatched_prefix(self):
        assert _parse_hf_spec("pride/PXD000001") is None

    def test_uuid_rejected(self):
        assert _parse_hf_spec("550e8400-e29b-41d4-a716-446655440000") is None

    def test_unclosed_bracket(self):
        assert _parse_hf_spec("hf/myorg/repo[a.mszx") is None


class TestHfDatasetDir:
    """Tests for the cache-path resolver."""

    @patch("msdatasets.hf.get_cache_dir")
    def test_default_uses_hf_subtree(self, mock_cache, tmp_path):
        mock_cache.return_value = tmp_path
        expected = tmp_path / "hf" / "alice" / "bench"
        assert _hf_dataset_dir("alice/bench", None) == expected

    def test_output_dir_overrides(self, tmp_path):
        target = tmp_path / "elsewhere"
        assert _hf_dataset_dir("alice/bench", target) == target

    def test_invalid_repo_id_raises(self):
        with pytest.raises(ValueError, match="Invalid HuggingFace repo_id"):
            _hf_dataset_dir("not-a-repo-id", None)


class TestHfProgressDisabled:
    """Tests for the HF_HUB_DISABLE_PROGRESS_BARS env toggle."""

    def test_show_progress_true_is_noop(self, monkeypatch):
        monkeypatch.delenv("HF_HUB_DISABLE_PROGRESS_BARS", raising=False)
        with _hf_progress_disabled(True):
            assert "HF_HUB_DISABLE_PROGRESS_BARS" not in os.environ

    def test_show_progress_false_sets_env(self, monkeypatch):
        monkeypatch.delenv("HF_HUB_DISABLE_PROGRESS_BARS", raising=False)
        with _hf_progress_disabled(False):
            assert os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] == "1"
        assert "HF_HUB_DISABLE_PROGRESS_BARS" not in os.environ

    def test_existing_value_is_restored(self, monkeypatch):
        monkeypatch.setenv("HF_HUB_DISABLE_PROGRESS_BARS", "preserved")
        with _hf_progress_disabled(False):
            assert os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] == "1"
        assert os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] == "preserved"


class TestDownloadHfDataset:
    """Tests for download_hf_dataset."""

    @patch("msdatasets.hf.get_cache_dir")
    def test_happy_path_collects_ms_files(self, mock_cache, tmp_path):
        mock_cache.return_value = tmp_path

        def populate(*args, **kwargs):
            local_dir = Path(kwargs["local_dir"])
            local_dir.mkdir(parents=True, exist_ok=True)
            (local_dir / "sample_02.mszx").write_bytes(b"x")
            (local_dir / "sample_01.mszx").write_bytes(b"x")
            (local_dir / "README.md").write_text("readme")
            return str(local_dir)

        ctx, snapshot = _patch_hf_hub(populate)
        with ctx:
            ds = download_hf_dataset("alice/bench", show_progress=False)

        snapshot.assert_called_once()
        kwargs = snapshot.call_args.kwargs
        assert kwargs["repo_id"] == "alice/bench"
        assert kwargs["repo_type"] == "dataset"
        assert kwargs["allow_patterns"] == _MS_PATTERNS
        assert kwargs["force_download"] is False
        assert kwargs["revision"] is None
        assert kwargs["token"] is None
        assert kwargs["local_dir"] == tmp_path / "hf" / "alice" / "bench"

        assert ds.dataset_id == "alice/bench"
        assert ds.dataset_name == "alice/bench"
        assert ds.cache_dir == tmp_path / "hf" / "alice" / "bench"
        # Only the .mszx files end up in the Dataset.files list (sorted).
        assert [f.name for f in ds.files] == ["sample_01.mszx", "sample_02.mszx"]

    @patch("msdatasets.hf.get_cache_dir")
    def test_filenames_forwarded_as_allow_patterns(self, mock_cache, tmp_path):
        mock_cache.return_value = tmp_path

        def noop(*args, **kwargs):
            Path(kwargs["local_dir"]).mkdir(parents=True, exist_ok=True)
            return str(kwargs["local_dir"])

        ctx, snapshot = _patch_hf_hub(noop)
        with ctx:
            download_hf_dataset(
                "alice/bench",
                filenames=["only-this.mszx"],
                show_progress=False,
            )

        assert snapshot.call_args.kwargs["allow_patterns"] == ["only-this.mszx"]

    def test_output_dir_overrides_cache(self, tmp_path):
        target = tmp_path / "one-off"

        def populate(*args, **kwargs):
            local_dir = Path(kwargs["local_dir"])
            local_dir.mkdir(parents=True, exist_ok=True)
            (local_dir / "a.mszx").write_bytes(b"x")
            return str(local_dir)

        ctx, snapshot = _patch_hf_hub(populate)
        with ctx, patch("msdatasets.hf.get_cache_dir") as mock_cache:
            ds = download_hf_dataset(
                "alice/bench",
                output_dir=target,
                show_progress=False,
            )

        # Cache resolver must not be consulted when output_dir is provided.
        mock_cache.assert_not_called()
        assert ds.cache_dir == target
        assert snapshot.call_args.kwargs["local_dir"] == target
        assert [f.name for f in ds.files] == ["a.mszx"]

    @patch("msdatasets.hf.get_cache_dir")
    def test_force_download_forwarded(self, mock_cache, tmp_path):
        mock_cache.return_value = tmp_path

        def noop(*args, **kwargs):
            Path(kwargs["local_dir"]).mkdir(parents=True, exist_ok=True)
            return str(kwargs["local_dir"])

        ctx, snapshot = _patch_hf_hub(noop)
        with ctx:
            download_hf_dataset(
                "alice/bench",
                force_download=True,
                revision="v1",
                token="hf_xxx",
                show_progress=False,
            )

        kwargs = snapshot.call_args.kwargs
        assert kwargs["force_download"] is True
        assert kwargs["revision"] == "v1"
        assert kwargs["token"] == "hf_xxx"

    @patch("msdatasets.hf.get_cache_dir")
    def test_show_progress_false_disables_hf_progress(self, mock_cache, tmp_path):
        mock_cache.return_value = tmp_path

        observed: dict[str, str | None] = {}

        def capture_env(*args, **kwargs):
            observed["env"] = os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS")
            Path(kwargs["local_dir"]).mkdir(parents=True, exist_ok=True)
            return str(kwargs["local_dir"])

        ctx, _ = _patch_hf_hub(capture_env)
        with ctx:
            download_hf_dataset("alice/bench", show_progress=False)

        assert observed["env"] == "1"


class TestHfErrorMapping:
    """Tests that HF errors map to the existing exceptions taxonomy."""

    @patch("msdatasets.hf.get_cache_dir")
    def test_repo_not_found_raises_dataset_not_found(self, mock_cache, tmp_path):
        mock_cache.return_value = tmp_path

        def boom(*args, **kwargs):
            raise _FakeHfErrors.RepositoryNotFoundError("nope")

        ctx, _ = _patch_hf_hub(boom)
        with ctx, pytest.raises(DatasetNotFoundError, match="alice/bench"):
            download_hf_dataset("alice/bench", show_progress=False)

    @patch("msdatasets.hf.get_cache_dir")
    def test_revision_not_found_raises_download_error(self, mock_cache, tmp_path):
        mock_cache.return_value = tmp_path

        def boom(*args, **kwargs):
            raise _FakeHfErrors.RevisionNotFoundError("nope")

        ctx, _ = _patch_hf_hub(boom)
        with ctx, pytest.raises(DownloadError, match="Revision"):
            download_hf_dataset("alice/bench", revision="v9", show_progress=False)

    @patch("msdatasets.hf.get_cache_dir")
    def test_http_error_raises_download_error(self, mock_cache, tmp_path):
        mock_cache.return_value = tmp_path

        def boom(*args, **kwargs):
            raise _FakeHfErrors.HfHubHTTPError("502 bad gateway")

        ctx, _ = _patch_hf_hub(boom)
        with ctx, pytest.raises(DownloadError, match="HuggingFace download failed"):
            download_hf_dataset("alice/bench", show_progress=False)


class TestImportHfHub:
    """Tests for the lazy huggingface_hub import helper."""

    def test_success_returns_tuple(self):
        snapshot = MagicMock()

        class FakeRepoNotFoundError(Exception):
            pass

        class FakeRevisionNotFoundError(Exception):
            pass

        class FakeHfHubHTTPError(Exception):
            pass

        hf_module = ModuleType("huggingface_hub")
        hf_module.snapshot_download = snapshot
        errors_module = ModuleType("huggingface_hub.errors")
        errors_module.HfHubHTTPError = FakeHfHubHTTPError
        errors_module.RepositoryNotFoundError = FakeRepoNotFoundError
        errors_module.RevisionNotFoundError = FakeRevisionNotFoundError
        hf_module.errors = errors_module

        with patch.dict(
            sys.modules,
            {"huggingface_hub": hf_module, "huggingface_hub.errors": errors_module},
        ):
            result = _import_hf_hub()

        assert result == (
            snapshot,
            FakeRepoNotFoundError,
            FakeRevisionNotFoundError,
            FakeHfHubHTTPError,
        )

    def test_missing_huggingface_hub_raises_helpful_error(self):
        real_import = (
            __builtins__["__import__"]
            if isinstance(__builtins__, dict)
            else __builtins__.__import__
        )

        def fake_import(name, *args, **kwargs):
            if name.startswith("huggingface_hub"):
                raise ImportError(f"No module named {name!r}")
            return real_import(name, *args, **kwargs)

        with patch.dict(sys.modules, {}, clear=False):
            sys.modules.pop("huggingface_hub", None)
            sys.modules.pop("huggingface_hub.errors", None)
            with patch("builtins.__import__", side_effect=fake_import):
                with pytest.raises(ImportError, match=r"msdatasets\[hf\]"):
                    _import_hf_hub()


class TestLoadHfDataset:
    """Tests for load_hf_dataset (HF → MSCompressDataset)."""

    @patch("msdatasets.hf.download_hf_dataset")
    def test_returns_mscompress_dataset(self, mock_download, tmp_path):
        mock_ds = MagicMock()
        mock_ds.cache_dir = tmp_path
        mock_download.return_value = mock_ds

        msc_cls = MagicMock(return_value="wrapped")
        fake_module = ModuleType("mscompress.datasets.torch")
        fake_module.MSCompressDataset = msc_cls

        with patch.dict(sys.modules, {"mscompress.datasets.torch": fake_module}):
            result = load_hf_dataset("alice/bench", show_progress=False)

        mock_download.assert_called_once_with(
            "alice/bench",
            filenames=None,
            revision=None,
            token=None,
            force_download=False,
            show_progress=False,
            output_dir=None,
        )
        msc_cls.assert_called_once_with(tmp_path, load_annotations=None)
        assert result == "wrapped"

    @patch("msdatasets.hf.download_hf_dataset")
    def test_forwards_load_annotations(self, mock_download, tmp_path):
        mock_ds = MagicMock()
        mock_ds.cache_dir = tmp_path
        mock_download.return_value = mock_ds
        sentinel = ["pseudo-AnnotationFormat"]

        msc_cls = MagicMock(return_value="wrapped")
        fake_module = ModuleType("mscompress.datasets.torch")
        fake_module.MSCompressDataset = msc_cls

        with patch.dict(sys.modules, {"mscompress.datasets.torch": fake_module}):
            load_hf_dataset(
                "alice/bench", show_progress=False, load_annotations=sentinel
            )

        msc_cls.assert_called_once_with(tmp_path, load_annotations=sentinel)
