"""Core download logic for fetching datasets from the server."""

from __future__ import annotations

import json
import logging
from pathlib import Path

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

from msdatasets.config import get_api_url, get_dataset_dir
from msdatasets.exceptions import DatasetNotFoundError, DownloadError
from msdatasets.models import Dataset, DatasetPart, Manifest

log = logging.getLogger("msdatasets")


def fetch_manifest(dataset_id: str, *, client: httpx.Client | None = None) -> Manifest:
    """Fetch and parse the manifest for *dataset_id* from the server."""
    url = f"{get_api_url()}/datasets/{dataset_id}/manifest"
    log.debug("Fetching manifest from %s", url)

    def _get(c: httpx.Client) -> httpx.Response:
        resp = c.get(url)
        if resp.status_code == 404:
            raise DatasetNotFoundError(f"Dataset not found: {dataset_id}")
        if resp.status_code >= 400:
            raise DownloadError(
                f"Server error {resp.status_code} fetching manifest for {dataset_id}"
            )
        return resp

    if client is not None:
        data = _get(client).json()
    else:
        with httpx.Client() as c:
            data = _get(c).json()

    manifest = Manifest.from_dict(data)
    log.debug(
        "Manifest: %s (%d parts)",
        manifest.dataset_name or dataset_id,
        manifest.total_parts,
    )
    return manifest


def download_part(
    dataset_id: str,
    part: DatasetPart,
    dest_dir: Path,
    *,
    skip_existing: bool = False,
    force: bool = False,
) -> Path:
    """Download a single dataset part to *dest_dir* via mstransfer.

    Thin wrapper around :func:`mstransfer.client.download_file`.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    url = f"{get_api_url()}{part.download_url}"
    dest = dest_dir / part.filename
    log.debug("Downloading %s from %s", part.filename, url)
    result: Path = download_file(url, dest, skip_existing=skip_existing, force=force)
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


def load_dataset(
    dataset_id: str,
    *,
    force_download: bool = False,
    show_progress: bool = True,
    max_workers: int = 4,
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
    """
    log.info("Loading dataset %s", dataset_id)
    dataset_dir = get_dataset_dir(dataset_id)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    log.debug("Dataset directory: %s", dataset_dir)

    timeout = httpx.Timeout(10.0, read=300.0)
    with httpx.Client(follow_redirects=True, timeout=timeout) as client:
        manifest = fetch_manifest(dataset_id, client=client)

    # Persist manifest locally for offline inspection
    manifest_path = dataset_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_id": manifest.dataset_id,
                "dataset_name": manifest.dataset_name,
                "total_parts": manifest.total_parts,
                "parts": [
                    {
                        "part_index": p.part_index,
                        "item_id": p.item_id,
                        "filename": p.filename,
                        "num_indices": p.num_indices,
                        "download_url": p.download_url,
                    }
                    for p in manifest.parts
                ],
            },
            indent=2,
        )
    )

    base_url = get_api_url()
    files: list[Path] = []
    parts_to_download: list[DatasetPart] = []

    for part in manifest.parts:
        dest = dataset_dir / part.filename
        if dest.exists() and not force_download:
            log.debug("Cached, skipping: %s", part.filename)
            files.append(dest)
        else:
            parts_to_download.append(part)

    if parts_to_download:
        log.info(
            "Downloading %d/%d part(s)",
            len(parts_to_download),
            manifest.total_parts,
        )
    else:
        log.info("All %d part(s) already cached", manifest.total_parts)

    if parts_to_download:
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


class _NullContext:
    """Minimal context manager for Python 3.10 compatibility."""

    def __enter__(self) -> _NullContext:
        return self

    def __exit__(self, *exc: object) -> None:
        pass
