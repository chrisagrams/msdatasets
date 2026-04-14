"""Async HTTP client for the msdatasets API."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx

from msdatasets.config import get_api_url
from msdatasets.exceptions import DatasetNotFoundError, DownloadError, ExtractionError
from msdatasets.models import (
    DatasetPart,
    Manifest,
    RepoDatasetRequest,
    RepoDatasetResponse,
    RepoImportStatus,
    RepoSource,
)

log = logging.getLogger("msdatasets")


async def _iter_sse_events(
    response: httpx.Response,
) -> AsyncIterator[tuple[str | None, dict[str, Any]]]:
    """Parse SSE events from an httpx streaming response."""
    event_type: str | None = None
    data_lines: list[str] = []
    async for line in response.aiter_lines():
        if line.startswith("event:"):
            event_type = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].strip())
        elif line == "":
            if data_lines:
                data = "\n".join(data_lines)
                yield event_type, json.loads(data)
                event_type = None
                data_lines = []
        # Lines starting with ":" are keep-alive pings — ignore

    # Flush any pending event when the stream closes
    if data_lines:
        data = "\n".join(data_lines)
        yield event_type, json.loads(data)


async def fetch_manifest(
    dataset_id: str,
    *,
    filenames: list[str] | None = None,
    client: httpx.AsyncClient,
) -> Manifest:
    """Fetch and parse the manifest for *dataset_id* from the server."""
    url = f"{get_api_url()}/datasets/{dataset_id}/manifest"
    params = {"filenames": filenames} if filenames else None
    log.debug("Fetching manifest from %s", url)

    resp = await client.get(url, params=params)

    if resp.status_code == 404:
        raise DatasetNotFoundError(f"Dataset not found: {dataset_id}")
    if resp.status_code >= 400:
        raise DownloadError(
            f"Server error {resp.status_code} fetching manifest for {dataset_id}"
        )

    manifest = Manifest.model_validate(resp.json())
    log.debug(
        "Manifest: %s (%d parts)",
        manifest.dataset_name or dataset_id,
        manifest.total_parts,
    )
    return manifest


async def stream_task(client: httpx.AsyncClient, task_id: str) -> None:
    """Stream SSE events for an extraction task until it reaches a terminal state."""
    url = f"{get_api_url()}/tasks/{task_id}/stream"
    async with client.stream("GET", url) as response:
        if response.status_code != 200:
            raise DownloadError(
                f"SSE stream failed for task {task_id}: HTTP {response.status_code}"
            )
        async for event_type, data in _iter_sse_events(response):
            if event_type == "done":
                return
            state = data.get("state")
            if state == "complete":
                return
            if state == "failed":
                raise ExtractionError(
                    f"Server extraction failed: {data.get('error', 'unknown error')}"
                )
            log.debug("Task %s: %s", task_id, state)


async def ensure_extracted(client: httpx.AsyncClient, part: DatasetPart) -> None:
    """Ensure a dataset part is extracted and ready for download.

    Hits the extraction endpoint.  If the server returns **204** the
    file is already cached.  If the server returns **202 Accepted**,
    the JSON payload is read to obtain the ``task_id`` and the task is
    streamed via SSE until extraction completes.
    """
    url = f"{get_api_url()}{part.extract_url}"
    log.debug("Checking extraction status for %s", part.filename)

    resp = await client.get(url)

    if resp.status_code == 204:
        log.debug("Already extracted: %s", part.filename)
        return

    if resp.status_code == 202:
        task_info = resp.json()
        task_id = task_info["task_id"]
        log.info("Extraction queued for %s (task %s)", part.filename, task_id)
        await stream_task(client, task_id)
        log.debug("Extraction complete: %s", part.filename)
        return

    if resp.status_code == 404:
        raise DownloadError(f"Part not found: {part.filename}")

    raise DownloadError(f"Server error {resp.status_code} for {part.filename}")


async def ensure_all_extracted(
    client: httpx.AsyncClient,
    parts: list[DatasetPart],
) -> None:
    """Ensure all parts are extracted, concurrently."""
    import asyncio

    await asyncio.gather(*(ensure_extracted(client, part) for part in parts))


async def _stream_repo_import(
    client: httpx.AsyncClient,
    result: RepoDatasetResponse,
    *,
    on_status: Callable[[str, RepoImportStatus], None] | None = None,
) -> None:
    """Stream SSE events for a repository import until all jobs complete."""
    url = f"{get_api_url()}/repositories/datasets/{result.dataset_id}/stream"
    job_states: dict[str, RepoImportStatus] = {}

    async with client.stream("GET", url) as response:
        if response.status_code != 200:
            raise DownloadError(
                f"SSE stream failed for repository import: HTTP {response.status_code}"
            )
        async for event_type, data in _iter_sse_events(response):
            if event_type == "done":
                return

            status = RepoImportStatus(data["status"])
            file_name = data.get("file_name", "unknown")

            if status == RepoImportStatus.FAILED:
                error = data.get("error_message") or "unknown error"
                raise DownloadError(
                    f"Repository import failed for {file_name}: {error}"
                )

            job_states[data["job_id"]] = status

            if on_status is not None:
                on_status(file_name, status)

            complete = sum(
                1 for s in job_states.values() if s == RepoImportStatus.COMPLETE
            )
            log.info(
                "Repository import: %d/%d files ready",
                complete,
                len(job_states),
            )

            if all(s == RepoImportStatus.COMPLETE for s in job_states.values()):
                return


async def trigger_repo_import(
    source: RepoSource | str,
    accession: str,
    *,
    filenames: list[str] | None = None,
    client: httpx.AsyncClient,
    on_status: Callable[[str, RepoImportStatus], None] | None = None,
) -> RepoDatasetResponse:
    """Trigger a repository import and wait until all jobs complete.

    Posts to ``/repositories/{source}/projects/{accession}/dataset`` and
    streams SSE events until all import jobs reach a terminal state.

    Returns the final :class:`RepoDatasetResponse`.
    """
    source = RepoSource(source)
    url = f"{get_api_url()}/repositories/{source.value}/projects/{accession}/dataset"
    log.info("Triggering %s import for %s", source.value, accession)

    body = RepoDatasetRequest(filenames=filenames)
    if filenames:
        log.info("Requesting specific files: %s", filenames)

    resp = await client.post(url, json=body.model_dump(exclude_none=True))
    if resp.status_code == 404:
        raise DatasetNotFoundError(f"{source.value} project not found: {accession}")
    if resp.status_code >= 400:
        raise DownloadError(
            f"Failed to trigger {source.value} import for {accession}: "
            f"HTTP {resp.status_code}"
        )

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

    failed = [j for j in result.jobs if j.status == RepoImportStatus.FAILED]
    if failed:
        details = ", ".join(
            f"{j.file_name or j.job_id or 'unknown'}: "
            f"{j.error_message or 'unknown error'}"
            for j in failed
        )
        raise DownloadError(f"{source.value} import failed — {details}")

    await _stream_repo_import(client, result, on_status=on_status)

    log.info(
        "%s import complete for %s (dataset %s)",
        source.value,
        accession,
        result.dataset_id,
    )
    return result
