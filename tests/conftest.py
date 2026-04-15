"""Pytest configuration and fixtures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from msdatasets.models import (
    DatasetPart,
    Manifest,
    RepoDatasetResponse,
    RepoImportJob,
    RepoImportStatus,
    RepoSource,
)


@dataclass
class FakeServerSentEvent:
    """Stand-in for ``httpx_sse.ServerSentEvent`` with just the attrs we use."""

    event: str
    data: str = ""


class FakeEventSource:
    """Stand-in for ``httpx_sse.EventSource`` returned by ``aconnect_sse``."""

    def __init__(
        self,
        events: list[FakeServerSentEvent],
        status_code: int = 200,
    ) -> None:
        self._events = events
        self.response = type("Resp", (), {"status_code": status_code})()

    async def aiter_sse(self):
        for ev in self._events:
            yield ev


class FakeSseContextManager:
    """Async context manager that yields a :class:`FakeEventSource`."""

    def __init__(self, source: FakeEventSource) -> None:
        self._source = source

    async def __aenter__(self) -> FakeEventSource:
        return self._source

    async def __aexit__(self, *_exc: Any) -> None:
        return None


@pytest.fixture
def fake_sse_stream():
    """Factory that builds a patch target for ``aconnect_sse``.

    Usage::

        with patch(
            "msdatasets.client.aconnect_sse",
            fake_sse_stream([("update", '{"state":"running"}'), ("done", "")]),
        ):
            ...
    """

    def _factory(
        events: list[tuple[str, str]],
        *,
        status_code: int = 200,
    ):
        sse_events = [FakeServerSentEvent(event=e, data=d) for e, d in events]
        source = FakeEventSource(sse_events, status_code=status_code)

        def _aconnect_sse(*_args: Any, **_kwargs: Any) -> FakeSseContextManager:
            return FakeSseContextManager(source)

        return _aconnect_sse

    return _factory


SAMPLE_MANIFEST_DICT: dict[str, Any] = {
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


@pytest.fixture
def make_part():
    """Factory to build a :class:`DatasetPart` with sensible defaults."""

    def _factory(**overrides: Any) -> DatasetPart:
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

    return _factory


@pytest.fixture
def make_manifest():
    """Factory to build a :class:`Manifest` from the sample dict."""

    def _factory(**overrides: Any) -> Manifest:
        data = {**SAMPLE_MANIFEST_DICT, **overrides}
        return Manifest.model_validate(data)

    return _factory


@pytest.fixture
def make_repo_response():
    """Factory to build a :class:`RepoDatasetResponse` with given job statuses."""

    def _factory(
        *,
        dataset_id: str = "ds-1",
        dataset_name: str = "Test Repo Dataset",
        source: RepoSource | str = RepoSource.PRIDE,
        accession: str = "PXD000001",
        job_statuses: list[RepoImportStatus] | None = None,
        job_file_names: list[str] | None = None,
        error_messages: list[str | None] | None = None,
    ) -> RepoDatasetResponse:
        statuses = job_statuses or [RepoImportStatus.PENDING]
        file_names = job_file_names or [f"file_{i}.raw" for i in range(len(statuses))]
        errors = error_messages or [None] * len(statuses)
        jobs = [
            RepoImportJob(
                status=status,
                source=RepoSource(source) if isinstance(source, str) else source,
                file_name=file_names[i],
                job_id=f"job-{i}",
                dataset_id=dataset_id,
                error_message=errors[i],
            )
            for i, status in enumerate(statuses)
        ]
        return RepoDatasetResponse(
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            source=RepoSource(source) if isinstance(source, str) else source,
            accession=accession,
            total_files=len(jobs),
            jobs=jobs,
        )

    return _factory
