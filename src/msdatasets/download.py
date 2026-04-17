"""Core download logic for fetching datasets from the server."""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from mscompress.datasets.torch import MSCompressDataset

import httpx
from mstransfer.client import download_batch, download_file
from mstransfer.client.downloader import DownloadRequest
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from msdatasets.client import (
    ensure_all_extracted,
    ensure_extracted,
    fetch_manifest,
    trigger_repo_import,
)
from msdatasets.config import get_api_url, get_dataset_dir
from msdatasets.models import (
    Dataset,
    DatasetPart,
    Manifest,
    RepoImportEvent,
    RepoImportStatus,
    RepoSource,
)

StoreFormat = Literal["mszx", "msz", "mzml"]

_STORE_FORMAT_EXT: dict[StoreFormat, str] = {
    "mszx": ".mszx",
    "msz": ".msz",
    "mzml": ".mzML",
}


def _target_filename(src_filename: str, store_as: StoreFormat) -> str:
    """Return the on-disk filename for *src_filename* after format conversion."""
    return str(Path(src_filename).with_suffix(_STORE_FORMAT_EXT[store_as]))


_STATUS_LABELS: dict[RepoImportStatus, str] = {
    RepoImportStatus.PENDING: "Pending",
    RepoImportStatus.DOWNLOADING: "Downloading",
    RepoImportStatus.CONVERTING: "Converting to MSZX",
    RepoImportStatus.INDEXING: "Indexing",
    RepoImportStatus.COMPLETE: "Complete",
}

log = logging.getLogger("msdatasets")

_REPO_PATTERN = re.compile(
    r"^(?P<source>pride|massive)/(?P<accession>[^\[\]/]+)(?:\[(?P<files>[^\]]+)\])?$"
)


def _parse_repo_spec(
    dataset_id: str,
) -> tuple[RepoSource, str, list[str] | None] | None:
    """Parse a ``{source}/{accession}[files]`` spec.

    Returns ``(source, accession, filenames)`` if *dataset_id* matches the
    repository pattern, otherwise ``None``.
    """
    match = _REPO_PATTERN.match(dataset_id)
    if not match:
        return None
    source = RepoSource(match.group("source"))
    accession = match.group("accession")
    files_group = match.group("files")
    filenames = [f.strip() for f in files_group.split(",")] if files_group else None
    return source, accession, filenames


def _import_torch_dataset() -> type[MSCompressDataset]:
    """Import MSCompressDataset, raising a helpful error if torch is missing."""
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
    store_as: StoreFormat = "mszx",
    skip_existing: bool = False,
    force: bool = False,
) -> Path:
    """Download a single dataset part to *dest_dir* via mstransfer.

    If the server needs to extract the file first (202 Accepted), polls
    the task endpoint until the file is ready, then downloads via the
    mstransfer endpoint.  When *store_as* is ``"msz"`` or ``"mzml"``,
    the downloaded ``.mszx`` is converted client-side by mstransfer.
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
        download_url,
        dest,
        store_as=store_as,
        skip_existing=skip_existing,
        force=force,
    )
    return result


class _RichImportProgress:
    """Adapts a rich `Progress` bar to the repo-import callback protocol.

    One Progress task is created lazily per job (keyed by job_id).  The
    task's description is updated on status transitions, and the bar
    advances on every DOWNLOADING event with a populated ``bytes_downloaded``.
    Status events that aren't DOWNLOADING (CONVERTING, INDEXING, COMPLETE)
    peg the bar at 100% and update the description.
    """

    def __init__(self, progress: Progress) -> None:
        self._progress = progress
        self._tasks: dict[str, TaskID] = {}

    def _ensure_task(
        self,
        job_id: str,
        file_name: str,
        total_bytes: int | None,
    ) -> TaskID:
        task_id = self._tasks.get(job_id)
        if task_id is None:
            task_id = self._progress.add_task(
                self._describe(file_name, RepoImportStatus.PENDING),
                total=total_bytes,
                start=False,
            )
            self._tasks[job_id] = task_id
        return task_id

    @staticmethod
    def _describe(file_name: str, status: RepoImportStatus) -> str:
        label = _STATUS_LABELS.get(status, status.value)
        return f"{file_name}: {label}"

    def on_status(self, file_name: str, status: RepoImportStatus) -> None:
        # on_status is transition-only; we may not have a task_id yet if
        # PENDING is the initial state, so look up by file_name as a fallback.
        job_id = self._find_job_by_file(file_name) or file_name
        task_id = self._tasks.get(job_id)
        if task_id is None:
            task_id = self._progress.add_task(
                self._describe(file_name, status), total=None, start=False
            )
            self._tasks[job_id] = task_id
        if status == RepoImportStatus.DOWNLOADING:
            self._progress.start_task(task_id)
        self._progress.update(task_id, description=self._describe(file_name, status))
        if status == RepoImportStatus.COMPLETE:
            # Pin at 100% if we have a total; otherwise just mark the task done.
            task = self._progress.tasks[task_id]
            if task.total is not None:
                self._progress.update(task_id, completed=task.total)

    def on_progress(self, event: RepoImportEvent) -> None:
        if (
            event.status != RepoImportStatus.DOWNLOADING
            or event.bytes_downloaded is None
        ):
            return
        file_name = event.file_name or event.job_id
        task_id = self._ensure_task(event.job_id, file_name, event.total_bytes)
        # Start the task (so speed/ETA render) and jump to the absolute byte count.
        task = self._progress.tasks[task_id]
        if not task.started:
            self._progress.start_task(task_id)
        update_kwargs: dict[str, object] = {"completed": event.bytes_downloaded}
        # Servers without Content-Length send total_bytes=None; if it shows
        # up later (unlikely), patch the total in.
        if event.total_bytes is not None and task.total != event.total_bytes:
            update_kwargs["total"] = event.total_bytes
        self._progress.update(task_id, **update_kwargs)

    def _find_job_by_file(self, file_name: str) -> str | None:
        # on_status only has file_name, but our tasks are keyed by job_id.
        # Walk the Progress tasks to find one whose description matches.
        for job_id, task_id in self._tasks.items():
            task = self._progress.tasks[task_id]
            if task.description.startswith(f"{file_name}:"):
                return job_id
        return None


class _RichBatchProgress:
    """Adapts a rich `Progress` bar to mstransfer's
    `BatchDownloadProgress` callback protocol."""

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
    store_as: StoreFormat = "mszx",
    output_dir: Path | None = None,
) -> Dataset:
    """Download a dataset and return a `Dataset` pointing to local files.

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
    store_as:
        On-disk format for downloaded parts.  Defaults to ``"mszx"`` (the
        raw archive shipped by the server).  Set to ``"msz"`` to extract
        the inner MSZ, or ``"mzml"`` to decompress fully to mzML.
        Conversion is handled by mstransfer.
    output_dir:
        Optional destination directory.  When set, files are written
        directly here (no ``{dataset_id}`` subdir) and the cache root
        from `get_dataset_dir` is bypassed.  Useful for one-off
        downloads outside the shared cache.
    """
    log.info("Downloading dataset %s", dataset_id)
    dataset_dir = output_dir if output_dir is not None else get_dataset_dir(dataset_id)
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

            # Skip files that are already on disk, unless *force_download* is True.
            # Cache is keyed by the target on-disk filename, so switching
            # --store-as triggers a re-download rather than using a stale
            # artifact in a different format.
            for part in manifest_.parts:
                dest = dataset_dir / _target_filename(part.filename, store_as)
                if dest.exists() and not force_download:
                    log.debug("Cached, skipping: %s", dest.name)
                    cached.append(dest)
                else:
                    to_download.append(part)

            # For files that need to be downloaded,
            # ensure they are extracted and ready on the server.
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
                store_as=store_as,
                parallel=max_workers,
                progress=batch_progress,
            )
            files.extend(downloaded)

    # Ensure files are ordered by part_index.  `files` entries use the
    # target extension (set by mstransfer), so key the lookup by that name.
    file_map = {p.name: p for p in files}
    ordered_files = [
        file_map[target]
        for part in manifest.parts
        if (target := _target_filename(part.filename, store_as)) in file_map
    ]

    ds = Dataset(
        dataset_id=manifest.dataset_id,
        dataset_name=manifest.dataset_name,
        cache_dir=dataset_dir,
        files=ordered_files,
    )
    log.info("Dataset ready: %d file(s) in %s", len(ds), dataset_dir)
    return ds


def download_repo_dataset(
    source: RepoSource | str,
    accession: str,
    *,
    filenames: list[str] | None = None,
    force_download: bool = False,
    show_progress: bool = True,
    max_workers: int = 4,
    store_as: StoreFormat = "mszx",
    output_dir: Path | None = None,
) -> Dataset:
    """Trigger a repository import and download the resulting dataset.

    Posts to ``/repositories/{source}/projects/{accession}/dataset`` to create
    a dataset from a PRIDE or MassIVE project.  The endpoint is
    idempotent—calling it for an already-imported project returns the
    existing dataset and job statuses.

    Parameters
    ----------
    source:
        Repository source (``"pride"`` or ``"massive"``).
    accession:
        Project accession (e.g. ``PXD075509`` for PRIDE, ``MSV000078787`` for
        MassIVE).
    filenames:
        Optional list of specific filenames to import. When *None*, all
        files in the project are imported.
    force_download:
        Re-download parts even if they already exist on disk.
    show_progress:
        Show a ``rich`` progress bar during download.
    max_workers:
        Maximum number of parallel downloads.
    store_as:
        On-disk format for downloaded parts.  See `download_dataset`.
    output_dir:
        Optional destination directory.  See `download_dataset`.
    """
    source = RepoSource(source)
    console = Console(stderr=True)

    async def _import() -> str:
        timeout = httpx.Timeout(10.0, read=None)
        async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
            if show_progress:
                progress = Progress(
                    TextColumn("[bold blue]{task.description}"),
                    BarColumn(),
                    DownloadColumn(),
                    TransferSpeedColumn(),
                    TimeRemainingColumn(),
                    console=console,
                    transient=False,
                )
                import_progress = _RichImportProgress(progress)
                with progress:
                    result = await trigger_repo_import(
                        source,
                        accession,
                        filenames=filenames,
                        client=client,
                        on_status=import_progress.on_status,
                        on_progress=import_progress.on_progress,
                    )
            else:
                result = await trigger_repo_import(
                    source, accession, filenames=filenames, client=client
                )
            return result.dataset_id

    dataset_id = asyncio.run(_import())

    return download_dataset(
        dataset_id,
        force_download=force_download,
        show_progress=show_progress,
        max_workers=max_workers,
        filenames=filenames,
        store_as=store_as,
        output_dir=output_dir,
    )


def load_repo_dataset(
    source: RepoSource | str,
    accession: str,
    *,
    filenames: list[str] | None = None,
    force_download: bool = False,
    show_progress: bool = True,
    max_workers: int = 4,
) -> MSCompressDataset:
    """Trigger a repository import and return an `MSCompressDataset` once ready.

    Convenience wrapper around `download_repo_dataset` that loads the
    downloaded files into an `mscompress.datasets.torch.MSCompressDataset`.
    Requires PyTorch to be installed.
    """
    dataset_cls = _import_torch_dataset()
    ds = download_repo_dataset(
        source,
        accession,
        filenames=filenames,
        force_download=force_download,
        show_progress=show_progress,
        max_workers=max_workers,
    )
    return dataset_cls(ds.cache_dir)


def load_dataset(
    dataset_id: str,
    *,
    force_download: bool = False,
    show_progress: bool = True,
    max_workers: int = 4,
) -> MSCompressDataset:
    """Download a dataset and return an `MSCompressDataset`.

    Convenience wrapper around `download_dataset` that loads the
    downloaded files into an `mscompress.datasets.torch.MSCompressDataset`
    ready for iteration.  Requires PyTorch to be installed.

    If *dataset_id* matches the pattern ``{source}/<accession>`` (e.g.
    ``pride/PXD075509`` or ``massive/MSV000078787``), the repository import
    flow is used instead.  A specific filename subset may be specified in
    square brackets: ``pride/PXD000001[file1.raw,file2.mzML]``.

    Parameters
    ----------
    dataset_id:
        Server-side dataset identifier, or a repository specifier like
        ``pride/PXD075509`` or ``massive/MSV000078787``.
    force_download:
        Re-download parts even if they already exist on disk.
    show_progress:
        Show a ``rich`` progress bar during download.
    max_workers:
        Maximum number of parallel downloads.
    """
    repo_spec = _parse_repo_spec(dataset_id)
    if repo_spec is not None:
        source, accession, filenames = repo_spec
        return load_repo_dataset(
            source,
            accession,
            filenames=filenames,
            force_download=force_download,
            show_progress=show_progress,
            max_workers=max_workers,
        )

    dataset_cls = _import_torch_dataset()

    ds = download_dataset(
        dataset_id,
        force_download=force_download,
        show_progress=show_progress,
        max_workers=max_workers,
    )
    return dataset_cls(ds.cache_dir)


class _NullContext:
    """Minimal context manager for Python 3.10 compatibility."""

    def __enter__(self) -> _NullContext:
        return self

    def __exit__(self, *exc: object) -> None:
        pass
