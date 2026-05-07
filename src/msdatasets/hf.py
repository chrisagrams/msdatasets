"""HuggingFace Hub download flow.

Downloads `.mszx` (and other MS-format) files directly from a HuggingFace
dataset repo via `huggingface_hub.snapshot_download`.  HF traffic does not
go through the msdatasets server.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    from mscompress.datasets.torch import MSCompressDataset
    from mscompress.types import AnnotationFormat

from msdatasets.config import get_cache_dir
from msdatasets.exceptions import DatasetNotFoundError, DownloadError
from msdatasets.models import Dataset

log = logging.getLogger("msdatasets")

_MS_PATTERNS: list[str] = [
    "*.mszx",
    "*.msz",
    "*.mzML",
    "*.mzml",
    "manifest.json",
    "README.md",
]

_MS_GLOBS: tuple[str, ...] = ("**/*.mszx", "**/*.msz", "**/*.mzML", "**/*.mzml")


def _import_hf_hub() -> tuple[Any, type[Exception], type[Exception], type[Exception]]:
    """Lazy-import huggingface_hub, raising a helpful error if missing."""
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import (
            HfHubHTTPError,
            RepositoryNotFoundError,
            RevisionNotFoundError,
        )
    except ImportError:
        raise ImportError(
            "Loading a HuggingFace dataset requires the 'hf' extra. "
            "Install it with: pip install msdatasets[hf]"
        ) from None
    return (
        snapshot_download,
        RepositoryNotFoundError,
        RevisionNotFoundError,
        HfHubHTTPError,
    )


@contextmanager
def _hf_progress_disabled(show_progress: bool) -> Iterator[None]:
    """Toggle ``HF_HUB_DISABLE_PROGRESS_BARS`` for the duration of the call."""
    if show_progress:
        yield
        return
    key = "HF_HUB_DISABLE_PROGRESS_BARS"
    prev = os.environ.get(key)
    os.environ[key] = "1"
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = prev


def _hf_dataset_dir(repo_id: str, output_dir: Path | None) -> Path:
    """Resolve the on-disk destination for an HF dataset download."""
    if output_dir is not None:
        return output_dir
    owner, _, repo = repo_id.partition("/")
    if not owner or not repo:
        raise ValueError(f"Invalid HuggingFace repo_id: {repo_id!r}")
    return get_cache_dir() / "hf" / owner / repo


def _collect_ms_files(dataset_dir: Path) -> list[Path]:
    """Return MS-format files under *dataset_dir*, sorted by name."""
    files: list[Path] = []
    for pattern in _MS_GLOBS:
        files.extend(dataset_dir.glob(pattern))
    return sorted(set(files))


def download_hf_dataset(
    repo_id: str,
    *,
    filenames: list[str] | None = None,
    revision: str | None = None,
    token: str | None = None,
    force_download: bool = False,
    show_progress: bool = True,
    output_dir: Path | None = None,
) -> Dataset:
    """Download a HuggingFace dataset repo of MS files.

    Parameters
    ----------
    repo_id:
        HuggingFace dataset repo ID in ``owner/name`` form.
    filenames:
        Optional list of specific filenames to fetch. When provided, the names
        are passed through as ``allow_patterns`` to ``snapshot_download``.
    revision:
        Optional branch, tag, or commit. Defaults to the repo's default branch.
    token:
        Optional HF auth token. Falls back to ``HF_TOKEN`` and to the token
        stored by ``huggingface-cli login``.
    force_download:
        Re-download files even if HF's cache already has them.
    show_progress:
        When False, sets ``HF_HUB_DISABLE_PROGRESS_BARS=1`` for the call.
    output_dir:
        Optional destination directory. When set, files land here directly
        (no ``hf/owner/repo`` nesting). Otherwise the shared cache is used.

    Notes
    -----
    `--store-as` conversion (mszx → msz / mzml) is not supported in this
    version. ``MSCompressDataset`` reads ``.mszx`` natively, so the PyTorch
    path works end-to-end without conversion.
    """
    snapshot_download, RepoNotFound, RevNotFound, HfHubHTTPError = (  # noqa: N806
        _import_hf_hub()
    )

    dataset_dir = _hf_dataset_dir(repo_id, output_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    allow_patterns = filenames if filenames else _MS_PATTERNS
    log.info("Downloading HuggingFace dataset %s", repo_id)
    log.debug("HF dataset dir: %s", dataset_dir)
    log.debug("HF allow_patterns: %s", allow_patterns)

    try:
        with _hf_progress_disabled(show_progress):
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=dataset_dir,
                revision=revision,
                token=token,
                force_download=force_download,
                allow_patterns=allow_patterns,
            )
    except RepoNotFound as exc:
        raise DatasetNotFoundError(f"HuggingFace dataset not found: {repo_id}") from exc
    except RevNotFound as exc:
        raise DownloadError(f"Revision not found: {revision} in {repo_id}") from exc
    except HfHubHTTPError as exc:
        raise DownloadError(f"HuggingFace download failed: {exc}") from exc

    files = _collect_ms_files(dataset_dir)
    log.info("HF dataset ready: %d file(s) in %s", len(files), dataset_dir)
    return Dataset(
        dataset_id=repo_id,
        dataset_name=repo_id,
        cache_dir=dataset_dir,
        files=files,
    )


def load_hf_dataset(
    repo_id: str,
    *,
    filenames: list[str] | None = None,
    revision: str | None = None,
    token: str | None = None,
    force_download: bool = False,
    show_progress: bool = True,
    output_dir: Path | None = None,
    load_annotations: list[AnnotationFormat] | None = None,
) -> MSCompressDataset:
    """Download an HF dataset repo and return an `MSCompressDataset`.

    Convenience wrapper around `download_hf_dataset`. Requires PyTorch.

    *load_annotations* is forwarded to ``MSCompressDataset``.  When set, the
    dataset's ``__getitem__`` returns ``(mz, intensity, annotations_dict)``
    instead of just ``(mz, intensity)``.
    """
    from msdatasets.download import _import_torch_dataset

    dataset_cls = _import_torch_dataset()
    ds = download_hf_dataset(
        repo_id,
        filenames=filenames,
        revision=revision,
        token=token,
        force_download=force_download,
        show_progress=show_progress,
        output_dir=output_dir,
    )
    return dataset_cls(ds.cache_dir, load_annotations=load_annotations)
