"""Async HTTP client for the msdatasets API."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from typing import TypeVar

import httpx
from httpx_sse import aconnect_sse

from msdatasets.config import get_api_url
from msdatasets.exceptions import DatasetNotFoundError, DownloadError, ExtractionError
from msdatasets.models import (
    DatasetPart,
    Manifest,
    NotificationEvent,
    RepoDatasetRequest,
    RepoDatasetResponse,
    RepoImportEvent,
    RepoImportStatus,
    RepoSource,
    TaskEvent,
)

log = logging.getLogger("msdatasets")

T = TypeVar("T", bound=NotificationEvent)


async def _iter_sse_events(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    model: type[T],
) -> AsyncIterator[tuple[str, T | None]]:
    """Connect to an SSE endpoint and yield ``(event_type, event)`` pairs.

    Events whose type is ``"done"`` yield ``None`` for the event payload
    since the terminal sentinel carries no schema-relevant data.  All
    other events are validated against *model*.
    """
    async with aconnect_sse(client, method, url) as source:
        if source.response.status_code != 200:
            raise DownloadError(
                f"SSE stream failed for {url}: HTTP {source.response.status_code}"
            )
        # Parse events as they arrive, yielding validated models
        async for sse in source.aiter_sse():
            if sse.event == "done":
                yield sse.event, None
                continue
            yield sse.event, model.model_validate_json(sse.data)


async def fetch_manifest(
    dataset_id: str,
    *,
    filenames: list[str] | None = None,
    client: httpx.AsyncClient,
) -> Manifest:
    """Fetch and parse the manifest for *dataset_id* from the server."""

    # Construct the URL for the manifest endpoint
    url = f"{get_api_url()}/datasets/{dataset_id}/manifest"

    # Optionally query for specific filenames if provided
    params = {"filenames": filenames} if filenames else None
    log.debug("Fetching manifest from %s", url)

    # Make the GET request to fetch the manifest
    resp = await client.get(url, params=params)

    if resp.status_code == 404:
        raise DatasetNotFoundError(f"Dataset not found: {dataset_id}")
    if resp.status_code >= 400:
        raise DownloadError(
            f"Server error {resp.status_code} fetching manifest for {dataset_id}"
        )

    # Validate and parse the manifest JSON into a Manifest model
    manifest = Manifest.model_validate(resp.json())
    log.debug(
        "Manifest: %s (%d parts)",
        manifest.dataset_name or dataset_id,
        manifest.total_parts,
    )
    return manifest


async def stream_task(client: httpx.AsyncClient, task_id: str) -> None:
    """Stream SSE events for an extraction task until it reaches a terminal state."""
    # Construct the URL for the task SSE endpoint
    url = f"{get_api_url()}/tasks/{task_id}/stream"
    async for event_type, event in _iter_sse_events(client, "GET", url, TaskEvent):
        # Terminal events have no payload, so *event* is None and we can stop
        if event_type == "done" or event is None:
            return
        if event.state == "complete":
            return
        if event.state == "failed":
            raise ExtractionError(
                f"Server extraction failed: {event.error or 'unknown error'}"
            )
        log.debug("Task %s: %s", task_id, event.state)


async def ensure_extracted(client: httpx.AsyncClient, part: DatasetPart) -> None:
    """Ensure a dataset part is extracted and ready for download.

    Hits the extraction endpoint.  If the server returns **204** the
    file is already cached.  If the server returns **202 Accepted**,
    the JSON payload is read to obtain the ``task_id`` and the task is
    streamed via SSE until extraction completes.
    """
    # Construct the URL for the extraction endpoint for this part
    url = f"{get_api_url()}{part.extract_url}"
    log.debug("Checking extraction status for %s", part.filename)

    resp = await client.get(url)

    # Already extracted and ready
    if resp.status_code == 204:
        log.debug("Already extracted: %s", part.filename)
        return

    # Extraction queued, stream task events until complete
    if resp.status_code == 202:
        task_info = resp.json()
        task_id = task_info["task_id"]
        log.info("Extraction queued for %s (task %s)", part.filename, task_id)
        await stream_task(client, task_id)
        log.debug("Extraction complete: %s", part.filename)
        return

    # Handle errors
    if resp.status_code == 404:
        raise DownloadError(f"Part not found: {part.filename}")

    raise DownloadError(f"Server error {resp.status_code} for {part.filename}")


async def ensure_all_extracted(
    client: httpx.AsyncClient,
    parts: list[DatasetPart],
) -> None:
    """Ensure all parts are extracted, concurrently."""
    await asyncio.gather(*(ensure_extracted(client, part) for part in parts))


async def _stream_repo_import(
    client: httpx.AsyncClient,
    result: RepoDatasetResponse,
    *,
    on_status: Callable[[str, RepoImportStatus], None] | None = None,
    on_progress: Callable[[RepoImportEvent], None] | None = None,
) -> None:
    """Stream SSE events for a repository import until all jobs complete.

    ``on_status`` fires once per status transition per job.
    ``on_progress`` fires on every event, including the ~2s download
    progress ticks, and receives the full event (with ``bytes_downloaded``,
    ``total_bytes``, ``speed_bps`` populated during DOWNLOADING).
    """
    url = f"{get_api_url()}/repositories/datasets/{result.dataset_id}/stream"

    # Seed from the initial POST response so the loop's completion check
    # covers every known job, even if the SSE stream never emits an event
    # for some of them (e.g. silent connection drop mid-stream).
    job_states: dict[str, RepoImportStatus] = {
        j.job_id: j.status for j in result.jobs if j.job_id is not None
    }
    expected_job_ids = set(job_states)
    done_received = False

    async for event_type, event in _iter_sse_events(
        client, "GET", url, RepoImportEvent
    ):
        if event_type == "done" or event is None:
            done_received = True
            break

        file_name = event.file_name or "unknown"

        # Always let progress observers see every event, including DOWNLOADING ticks.
        if on_progress is not None:
            on_progress(event)

        # If a job fails, we currently have no way to recover.
        # TODO: Consider retrying failed jobs
        if event.status == RepoImportStatus.FAILED:
            raise DownloadError(
                f"Repository import failed for {file_name}: "
                f"{event.error_message or 'unknown error'}"
            )

        # Only fire on_status + the summary log when a job actually changes
        # status — otherwise downloading ticks would spam both.
        previous = job_states.get(event.job_id)
        is_transition = previous != event.status
        job_states[event.job_id] = event.status

        if is_transition and on_status is not None:
            on_status(file_name, event.status)

        if event.status == RepoImportStatus.DOWNLOADING and event.bytes_downloaded:
            if event.total_bytes:
                log.debug(
                    "Repo import %s: %.1f%% (%d/%d bytes) %.2f MiB/s",
                    file_name,
                    100.0 * event.bytes_downloaded / event.total_bytes,
                    event.bytes_downloaded,
                    event.total_bytes,
                    (event.speed_bps or 0.0) / (1 << 20),
                )
            else:
                log.debug(
                    "Repo import %s: %d bytes %.2f MiB/s",
                    file_name,
                    event.bytes_downloaded,
                    (event.speed_bps or 0.0) / (1 << 20),
                )
        elif is_transition and event.status == RepoImportStatus.COMPLETE:
            complete = sum(
                1 for s in job_states.values() if s == RepoImportStatus.COMPLETE
            )
            log.info(
                "Repository import: %d/%d files ready",
                complete,
                len(job_states),
            )

        # Once all jobs are complete, we can stop streaming.
        if all(s == RepoImportStatus.COMPLETE for s in job_states.values()):
            break

    # Guard against a silent stream close (connection drop, server restart,
    # proxy buffer, etc.).  Without this, a half-finished import would
    # surface as "Done! 0 file(s)" in the CLI.  An explicit ``done`` event
    # from the server is authoritative — trust it even if our local state
    # is behind.
    if not done_received:
        unfinished = [
            jid
            for jid in expected_job_ids
            if job_states.get(jid) != RepoImportStatus.COMPLETE
        ]
        if unfinished:
            raise DownloadError(
                "Repository import stream closed before all jobs completed "
                f"({len(unfinished)}/{len(expected_job_ids)} still pending). "
                "Inspect /repositories/imports to see the latest status."
            )


async def trigger_repo_import(
    source: RepoSource | str,
    accession: str,
    *,
    filenames: list[str] | None = None,
    client: httpx.AsyncClient,
    on_status: Callable[[str, RepoImportStatus], None] | None = None,
    on_progress: Callable[[RepoImportEvent], None] | None = None,
) -> RepoDatasetResponse:
    """Trigger a repository import and wait until all jobs complete.

    Posts to ``/repositories/{source}/projects/{accession}/dataset`` and
    streams SSE events until all import jobs reach a terminal state.

    ``on_status`` fires once per status transition per job.
    ``on_progress`` receives every event, including ~2s download progress
    ticks carrying ``bytes_downloaded`` / ``total_bytes`` / ``speed_bps``.

    Returns the final `RepoDatasetResponse`.
    """
    source = RepoSource(source)
    url = f"{get_api_url()}/repositories/{source.value}/projects/{accession}/dataset"
    log.info("Triggering %s import for %s", source.value, accession)

    # Construct a request for a repository dataset import
    body = RepoDatasetRequest(filenames=filenames)
    if filenames:
        log.info("Requesting specific files: %s", filenames)

    # Trigger the import by POSTing to the server.
    resp = await client.post(url, json=body.model_dump(exclude_none=True))
    if resp.status_code == 404:
        raise DatasetNotFoundError(f"{source.value} project not found: {accession}")
    if resp.status_code >= 400:
        raise DownloadError(
            f"Failed to trigger {source.value} import for {accession}: "
            f"HTTP {resp.status_code}"
        )

    # Parse the initial response,
    # which includes the status of all import jobs at the time of creation.
    result = RepoDatasetResponse.model_validate(resp.json())

    # Check if already done
    if all(j.status == RepoImportStatus.COMPLETE for j in result.jobs):
        log.info(
            "%s import complete for %s (dataset %s)",
            source.value,
            accession,
            result.dataset_id,
        )
        return result

    # If any jobs have already failed, we can stop early without streaming
    failed = [j for j in result.jobs if j.status == RepoImportStatus.FAILED]
    if failed:
        details = ", ".join(
            f"{j.file_name or j.job_id or 'unknown'}: "
            f"{j.error_message or 'unknown error'}"
            for j in failed
        )
        raise DownloadError(f"{source.value} import failed — {details}")

    # Stream SSE events until all jobs are complete.
    await _stream_repo_import(
        client, result, on_status=on_status, on_progress=on_progress
    )

    log.info(
        "%s import complete for %s (dataset %s)",
        source.value,
        accession,
        result.dataset_id,
    )
    return result
