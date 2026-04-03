"""Core download logic for fetching datasets from the server."""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mscompress.datasets.torch import MSCompressDataset

import httpx
from mstransfer.client import download_batch, download_file
from mstransfer.client.downloader import DownloadRequest
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TaskID,
    TextColumn,
    TransferSpeedColumn,
)

from rich.console import Console

from msdatasets.client import (
    ensure_all_extracted,
    ensure_extracted,
    fetch_manifest,
    trigger_pride_import,
)
from msdatasets.config import get_api_url, get_dataset_dir
from msdatasets.models import Dataset, DatasetPart, Manifest, PrideImportStatus

_STATUS_LABELS: dict[PrideImportStatus, str] = {
    PrideImportStatus.PENDING: "Pending",
    PrideImportStatus.DOWNLOADING: "Downloading from PRIDE",
    PrideImportStatus.CONVERTING: "Converting to MSZX",
    PrideImportStatus.INDEXING: "Indexing",
    PrideImportStatus.COMPLETE: "Complete",
}

log = logging.getLogger("msdatasets")

_PRIDE_PATTERN = re.compile(r"^pride/(PXD\d+)(?:\[([^\]]+)\])?$")


def _import_torch_dataset() -> type[MSCompressDataset]:
    """Import and return MSCompressDataset, raising a helpful error if torch is missing."""
    try:
        from mscompress.datasets.torch import MSCompressDataset
    except ImportError:
        raise ImportError(
            "Loading a dataset as a PyTorch Dataset requires the 'torch' extra. "
            "Install it with: pip install msdatasets[torch]"
        ) from None
    return MSCompressDataset


def download_part(
    part: DatasetPart,
    dest_dir: Path,
    *,
    skip_existing: bool = False,
    force: bool = False,
) -> Path:
    """Download a single dataset part to *dest_dir* via mstransfer.

    If the server needs to extract the file first (202 Accepted), polls
    the task endpoint until the file is ready, then downloads via the
    mstransfer endpoint.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    download_url = f"{get_api_url()}{part.download_url}"
    dest = dest_dir / part.filename
    log.debug("Downloading %s via %s", part.filename, download_url)

    async def _extract() -> None:
        timeout = httpx.Timeout(10.0, read=None)
        async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
            await ensure_extracted(client, part)

    asyncio.run(_extract())

    result: Path = download_file(
        download_url, dest, skip_existing=skip_existing, force=force
    )
    return result


class _RichBatchProgress:
    """Adapts a rich :class:`Progress` bar to mstransfer's
    :class:`BatchDownloadProgress` callback protocol."""

    def __init__(self, progress: Progress) -> None:
        self._progress = progress
        self._tasks: dict[str, TaskID] = {}

    def on_file_start(self, filename: str, total_bytes: int | None) -> None:
        self._tasks[filename] = self._progress.add_task(filename, total=total_bytes)

    def on_file_progress(self, filename: str, bytes_delta: int) -> None:
        if filename in self._tasks:
            self._progress.update(self._tasks[filename], advance=bytes_delta)

    def on_file_complete(self, filename: str) -> None:
        pass

    def on_file_error(self, filename: str, error: Exception) -> None:
        if filename in self._tasks:
            self._progress.update(
                self._tasks[filename],
                description=f"[red]{filename} (error)",
            )


def download_dataset(
    dataset_id: str,
    *,
    force_download: bool = False,
    show_progress: bool = True,
    max_workers: int = 4,
    filenames: list[str] | None = None,
) -> Dataset:
    """Download a dataset and return a :class:`Dataset` pointing to local files.

    Parameters
    ----------
    dataset_id:
        Server-side dataset identifier (UUID).
    force_download:
        Re-download parts even if they already exist on disk.
    show_progress:
        Show a ``rich`` progress bar during download.
    max_workers:
        Maximum number of parallel downloads.
    filenames:
        Optional list of filenames to include. When provided, the server
        returns a manifest containing only matching parts.
    """
    log.info("Downloading dataset %s", dataset_id)
    dataset_dir = get_dataset_dir(dataset_id)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    log.debug("Dataset directory: %s", dataset_dir)

    async def _fetch_and_extract() -> tuple[Manifest, list[Path], list[DatasetPart]]:
        timeout = httpx.Timeout(10.0, read=None)
        async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
            manifest_ = await fetch_manifest(
                dataset_id, filenames=filenames, client=client
            )

            # Persist manifest locally for offline inspection
            manifest_path = dataset_dir / "manifest.json"
            manifest_path.write_text(manifest_.model_dump_json(indent=2))

            cached: list[Path] = []
            to_download: list[DatasetPart] = []

            for part in manifest_.parts:
                dest = dataset_dir / part.filename
                if dest.exists() and not force_download:
                    log.debug("Cached, skipping: %s", part.filename)
                    cached.append(dest)
                else:
                    to_download.append(part)

            if to_download:
                log.info(
                    "Downloading %d/%d part(s)",
                    len(to_download),
                    manifest_.total_parts,
                )
                await ensure_all_extracted(client, to_download)
            else:
                log.info("All %d part(s) already cached", manifest_.total_parts)

            return manifest_, cached, to_download

    manifest, files, parts_to_download = asyncio.run(_fetch_and_extract())

    # Download all ready files via mstransfer.
    if parts_to_download:
        base_url = get_api_url()
        requests = [
            DownloadRequest(
                url=f"{base_url}{part.download_url}",
                dest=dataset_dir / part.filename,
            )
            for part in parts_to_download
        ]

        progress_bar: Progress | None = None
        batch_progress: _RichBatchProgress | None = None
        if show_progress:
            progress_bar = Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                DownloadColumn(),
                TransferSpeedColumn(),
            )
            batch_progress = _RichBatchProgress(progress_bar)

        ctx = progress_bar if progress_bar is not None else _NullContext()
        with ctx:
            downloaded = download_batch(
                requests,
                parallel=max_workers,
                progress=batch_progress,
            )
            files.extend(downloaded)

    # Ensure files are ordered by part_index
    file_map = {p.name: p for p in files}
    ordered_files = [
        file_map[part.filename] for part in manifest.parts if part.filename in file_map
    ]

    ds = Dataset(
        dataset_id=manifest.dataset_id,
        dataset_name=manifest.dataset_name,
        cache_dir=dataset_dir,
        files=ordered_files,
    )
    log.info("Dataset ready: %d file(s) in %s", len(ds), dataset_dir)
    return ds


def load_pride_dataset(
    accession: str,
    *,
    filenames: list[str] | None = None,
    force_download: bool = False,
    show_progress: bool = True,
    max_workers: int = 4,
) -> MSCompressDataset:
    """Trigger a PRIDE import and return an :class:`MSCompressDataset` once ready.

    Posts to ``/pride/{accession}/dataset`` to create a dataset from a
    PRIDE project.  The endpoint is idempotent—calling it for an
    already-imported project returns the existing dataset and job statuses.

    Parameters
    ----------
    accession:
        PRIDE project accession (e.g. ``PXD075509``).
    filenames:
        Optional list of specific mzML filenames to import. When *None*,
        all files in the project are imported.
    force_download:
        Re-download parts even if they already exist on disk.
    show_progress:
        Show a ``rich`` progress bar during download.
    max_workers:
        Maximum number of parallel downloads.
    """
    MSCompressDataset = _import_torch_dataset()
    console = Console(stderr=True)

    async def _import() -> str:
        timeout = httpx.Timeout(10.0, read=None)
        async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
            if show_progress:
                with console.status(
                    f"[bold blue]PRIDE import {accession}: pending…",
                    spinner="dots",
                ) as spinner:

                    def _on_status(
                        file_name: str, status: PrideImportStatus
                    ) -> None:
                        label = _STATUS_LABELS.get(status, status.value)
                        spinner.update(
                            f"[bold blue]{file_name}: {label}…"
                        )

                    result = await trigger_pride_import(
                        accession,
                        filenames=filenames,
                        client=client,
                        on_status=_on_status,
                    )
            else:
                result = await trigger_pride_import(
                    accession, filenames=filenames, client=client
                )
            return result.dataset_id

    dataset_id = asyncio.run(_import())

    ds = download_dataset(
        dataset_id,
        force_download=force_download,
        show_progress=show_progress,
        max_workers=max_workers,
        filenames=filenames,
    )
    return MSCompressDataset(ds.cache_dir)


def load_dataset(
    dataset_id: str,
    *,
    force_download: bool = False,
    show_progress: bool = True,
    max_workers: int = 4,
) -> MSCompressDataset:
    """Download a dataset and return an :class:`MSCompressDataset`.

    Convenience wrapper around :func:`download_dataset` that loads the
    downloaded files into an :class:`mscompress.datasets.torch.MSCompressDataset`
    ready for iteration.  Requires PyTorch to be installed.

    If *dataset_id* matches the pattern ``pride/<accession>`` (e.g.
    ``pride/PXD075509``), the PRIDE import flow is used instead.

    Parameters
    ----------
    dataset_id:
        Server-side dataset identifier, or a PRIDE specifier like
        ``pride/PXD075509``.
    force_download:
        Re-download parts even if they already exist on disk.
    show_progress:
        Show a ``rich`` progress bar during download.
    max_workers:
        Maximum number of parallel downloads.
    """
    match = _PRIDE_PATTERN.match(dataset_id)
    if match:
        accession = match.group(1)
        filenames = (
            [f.strip() for f in match.group(2).split(",")]
            if match.group(2)
            else None
        )
        return load_pride_dataset(
            accession,
            filenames=filenames,
            force_download=force_download,
            show_progress=show_progress,
            max_workers=max_workers,
        )

    MSCompressDataset = _import_torch_dataset()

    ds = download_dataset(
        dataset_id,
        force_download=force_download,
        show_progress=show_progress,
        max_workers=max_workers,
    )
    return MSCompressDataset(ds.cache_dir)


class _NullContext:
    """Minimal context manager for Python 3.10 compatibility."""

    def __enter__(self) -> _NullContext:
        return self

    def __exit__(self, *exc: object) -> None:
        pass
