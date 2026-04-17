"""Tests for msdatasets.client (async HTTP client + SSE streams)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from msdatasets.client import (
    _iter_sse_events,
    _stream_repo_import,
    ensure_all_extracted,
    stream_task,
    trigger_repo_import,
)
from msdatasets.exceptions import DatasetNotFoundError, DownloadError, ExtractionError
from msdatasets.models import (
    RepoImportEvent,
    RepoImportStatus,
    RepoSource,
    TaskEvent,
)


class TestIterSseEvents:
    """Tests for the generic SSE iterator."""

    @pytest.mark.asyncio
    async def test_non_200_raises(self, fake_sse_stream):
        patcher = fake_sse_stream([], status_code=500)
        client = AsyncMock()
        with patch("msdatasets.client.aconnect_sse", patcher):
            with pytest.raises(DownloadError, match="HTTP 500"):
                async for _ in _iter_sse_events(
                    client, "GET", "https://x/y", TaskEvent
                ):
                    pass

    @pytest.mark.asyncio
    async def test_done_sentinel_yields_none(self, fake_sse_stream):
        patcher = fake_sse_stream([("done", "")])
        client = AsyncMock()
        with patch("msdatasets.client.aconnect_sse", patcher):
            collected = [
                item
                async for item in _iter_sse_events(
                    client, "GET", "https://x/y", TaskEvent
                )
            ]
        assert collected == [("done", None)]

    @pytest.mark.asyncio
    async def test_normal_event_is_validated(self, fake_sse_stream):
        patcher = fake_sse_stream([("update", '{"state":"running"}')])
        client = AsyncMock()
        with patch("msdatasets.client.aconnect_sse", patcher):
            collected = [
                item
                async for item in _iter_sse_events(
                    client, "GET", "https://x/y", TaskEvent
                )
            ]
        assert len(collected) == 1
        event_type, event = collected[0]
        assert event_type == "update"
        assert isinstance(event, TaskEvent)
        assert event.state == "running"

    @pytest.mark.asyncio
    async def test_malformed_payload_raises(self, fake_sse_stream):
        patcher = fake_sse_stream([("update", '{"state":')])
        client = AsyncMock()
        with patch("msdatasets.client.aconnect_sse", patcher):
            with pytest.raises(ValidationError):
                async for _ in _iter_sse_events(
                    client, "GET", "https://x/y", TaskEvent
                ):
                    pass


class TestStreamTask:
    """Tests for stream_task."""

    @pytest.mark.asyncio
    @patch("msdatasets.client.get_api_url", return_value="https://api.example.com")
    async def test_complete_returns(self, _api, fake_sse_stream):
        patcher = fake_sse_stream([("update", '{"state":"complete"}')])
        client = AsyncMock()
        with patch("msdatasets.client.aconnect_sse", patcher):
            await stream_task(client, "task-1")

    @pytest.mark.asyncio
    @patch("msdatasets.client.get_api_url", return_value="https://api.example.com")
    async def test_failed_raises_extraction_error(self, _api, fake_sse_stream):
        patcher = fake_sse_stream([("update", '{"state":"failed","error":"boom"}')])
        client = AsyncMock()
        with patch("msdatasets.client.aconnect_sse", patcher):
            with pytest.raises(ExtractionError, match="boom"):
                await stream_task(client, "task-1")

    @pytest.mark.asyncio
    @patch("msdatasets.client.get_api_url", return_value="https://api.example.com")
    async def test_failed_without_error_message(self, _api, fake_sse_stream):
        patcher = fake_sse_stream([("update", '{"state":"failed"}')])
        client = AsyncMock()
        with patch("msdatasets.client.aconnect_sse", patcher):
            with pytest.raises(ExtractionError, match="unknown error"):
                await stream_task(client, "task-1")

    @pytest.mark.asyncio
    @patch("msdatasets.client.get_api_url", return_value="https://api.example.com")
    async def test_done_sentinel_returns(self, _api, fake_sse_stream):
        patcher = fake_sse_stream([("done", "")])
        client = AsyncMock()
        with patch("msdatasets.client.aconnect_sse", patcher):
            await stream_task(client, "task-1")

    @pytest.mark.asyncio
    @patch("msdatasets.client.get_api_url", return_value="https://api.example.com")
    async def test_intermediate_state_continues(self, _api, fake_sse_stream):
        patcher = fake_sse_stream(
            [
                ("update", '{"state":"queued"}'),
                ("update", '{"state":"running"}'),
                ("update", '{"state":"complete"}'),
            ]
        )
        client = AsyncMock()
        with patch("msdatasets.client.aconnect_sse", patcher):
            await stream_task(client, "task-1")


class TestEnsureAllExtracted:
    """Tests for ensure_all_extracted (concurrency wrapper)."""

    @pytest.mark.asyncio
    @patch("msdatasets.client.ensure_extracted", new_callable=AsyncMock)
    async def test_calls_each_part(self, mock_ensure, make_part):
        parts = [make_part(item_id=f"p{i}") for i in range(3)]
        client = AsyncMock()
        await ensure_all_extracted(client, parts)
        assert mock_ensure.call_count == 3

    @pytest.mark.asyncio
    @patch("msdatasets.client.ensure_extracted", new_callable=AsyncMock)
    async def test_failure_propagates(self, mock_ensure, make_part):
        mock_ensure.side_effect = [None, DownloadError("boom"), None]
        parts = [make_part(item_id=f"p{i}") for i in range(3)]
        client = AsyncMock()
        with pytest.raises(DownloadError, match="boom"):
            await ensure_all_extracted(client, parts)


class TestStreamRepoImport:
    """Tests for _stream_repo_import."""

    @pytest.mark.asyncio
    @patch("msdatasets.client.get_api_url", return_value="https://api.example.com")
    async def test_failed_status_raises(
        self, _api, fake_sse_stream, make_repo_response
    ):
        result = make_repo_response()
        payload = (
            '{"status":"failed","job_id":"job-0","file_name":"a.raw",'
            '"error_message":"disk full"}'
        )
        patcher = fake_sse_stream([("status", payload)])
        client = AsyncMock()
        with patch("msdatasets.client.aconnect_sse", patcher):
            with pytest.raises(DownloadError, match="disk full"):
                await _stream_repo_import(client, result)

    @pytest.mark.asyncio
    @patch("msdatasets.client.get_api_url", return_value="https://api.example.com")
    async def test_on_status_callback_invoked(
        self, _api, fake_sse_stream, make_repo_response
    ):
        result = make_repo_response()
        patcher = fake_sse_stream(
            [
                (
                    "status",
                    '{"status":"downloading","job_id":"job-0","file_name":"a.raw"}',
                ),
                (
                    "status",
                    '{"status":"complete","job_id":"job-0","file_name":"a.raw"}',
                ),
            ]
        )
        callback = MagicMock()
        client = AsyncMock()
        with patch("msdatasets.client.aconnect_sse", patcher):
            await _stream_repo_import(client, result, on_status=callback)
        assert callback.call_count == 2
        assert callback.call_args_list[0].args == (
            "a.raw",
            RepoImportStatus.DOWNLOADING,
        )
        assert callback.call_args_list[1].args == (
            "a.raw",
            RepoImportStatus.COMPLETE,
        )

    @pytest.mark.asyncio
    @patch("msdatasets.client.get_api_url", return_value="https://api.example.com")
    async def test_exits_when_all_jobs_complete(
        self, _api, fake_sse_stream, make_repo_response
    ):
        result = make_repo_response(
            job_statuses=[RepoImportStatus.PENDING, RepoImportStatus.PENDING],
            job_file_names=["a", "b"],
        )
        patcher = fake_sse_stream(
            [
                ("status", '{"status":"complete","job_id":"job-0","file_name":"a"}'),
                ("status", '{"status":"complete","job_id":"job-1","file_name":"b"}'),
                # Extra event that should never be consumed because loop exits.
                ("status", '{"status":"pending","job_id":"job-2","file_name":"c"}'),
            ]
        )
        client = AsyncMock()
        with patch("msdatasets.client.aconnect_sse", patcher):
            await _stream_repo_import(client, result)

    @pytest.mark.asyncio
    @patch("msdatasets.client.get_api_url", return_value="https://api.example.com")
    async def test_done_sentinel_returns(
        self, _api, fake_sse_stream, make_repo_response
    ):
        result = make_repo_response()
        patcher = fake_sse_stream([("done", "")])
        client = AsyncMock()
        with patch("msdatasets.client.aconnect_sse", patcher):
            await _stream_repo_import(client, result)

    @pytest.mark.asyncio
    @patch("msdatasets.client.get_api_url", return_value="https://api.example.com")
    async def test_missing_file_name_uses_unknown(
        self, _api, fake_sse_stream, make_repo_response
    ):
        result = make_repo_response()
        payload = '{"status":"failed","job_id":"j1","error_message":"oops"}'
        patcher = fake_sse_stream([("status", payload)])
        client = AsyncMock()
        with patch("msdatasets.client.aconnect_sse", patcher):
            with pytest.raises(DownloadError, match="unknown"):
                await _stream_repo_import(client, result)

    @pytest.mark.asyncio
    @patch("msdatasets.client.get_api_url", return_value="https://api.example.com")
    async def test_on_progress_receives_download_ticks(
        self, _api, fake_sse_stream, make_repo_response
    ):
        """on_progress fires on every event, including repeated DOWNLOADING ticks."""
        result = make_repo_response()
        patcher = fake_sse_stream(
            [
                (
                    "status",
                    '{"status":"downloading","job_id":"job-0","file_name":"a.raw"}',
                ),
                (
                    "status",
                    '{"status":"downloading","job_id":"job-0","file_name":"a.raw",'
                    '"bytes_downloaded":1000,"total_bytes":4000,"speed_bps":500.0}',
                ),
                (
                    "status",
                    '{"status":"downloading","job_id":"job-0","file_name":"a.raw",'
                    '"bytes_downloaded":3000,"total_bytes":4000,"speed_bps":750.0}',
                ),
                (
                    "status",
                    '{"status":"complete","job_id":"job-0","file_name":"a.raw"}',
                ),
            ]
        )
        progress_events: list[RepoImportEvent] = []
        client = AsyncMock()
        with patch("msdatasets.client.aconnect_sse", patcher):
            await _stream_repo_import(
                client, result, on_progress=progress_events.append
            )
        # Every event — including progress ticks and terminal — is delivered.
        assert len(progress_events) == 4
        assert progress_events[1].bytes_downloaded == 1000
        assert progress_events[1].total_bytes == 4000
        assert progress_events[1].speed_bps == 500.0
        assert progress_events[2].bytes_downloaded == 3000

    @pytest.mark.asyncio
    @patch("msdatasets.client.get_api_url", return_value="https://api.example.com")
    async def test_on_status_fires_only_on_transitions(
        self, _api, fake_sse_stream, make_repo_response
    ):
        """Repeated DOWNLOADING events must not re-fire on_status."""
        result = make_repo_response()
        patcher = fake_sse_stream(
            [
                (
                    "status",
                    '{"status":"downloading","job_id":"job-0","file_name":"a.raw"}',
                ),
                (
                    "status",
                    '{"status":"downloading","job_id":"job-0","file_name":"a.raw",'
                    '"bytes_downloaded":100,"total_bytes":200,"speed_bps":50.0}',
                ),
                (
                    "status",
                    '{"status":"downloading","job_id":"job-0","file_name":"a.raw",'
                    '"bytes_downloaded":200,"total_bytes":200,"speed_bps":75.0}',
                ),
                (
                    "status",
                    '{"status":"converting","job_id":"job-0","file_name":"a.raw"}',
                ),
                (
                    "status",
                    '{"status":"complete","job_id":"job-0","file_name":"a.raw"}',
                ),
            ]
        )
        callback = MagicMock()
        client = AsyncMock()
        with patch("msdatasets.client.aconnect_sse", patcher):
            await _stream_repo_import(client, result, on_status=callback)
        # Transitions: PENDING→DOWNLOADING, DOWNLOADING→CONVERTING, CONVERTING→COMPLETE.
        # The two DOWNLOADING ticks after the initial one must not re-fire.
        assert callback.call_count == 3
        statuses = [c.args[1] for c in callback.call_args_list]
        assert statuses == [
            RepoImportStatus.DOWNLOADING,
            RepoImportStatus.CONVERTING,
            RepoImportStatus.COMPLETE,
        ]

    @pytest.mark.asyncio
    @patch("msdatasets.client.get_api_url", return_value="https://api.example.com")
    async def test_silent_stream_close_raises(
        self, _api, fake_sse_stream, make_repo_response
    ):
        """Stream ends without ``done`` and an expected job never completed."""
        result = make_repo_response()  # one PENDING job
        # Only DOWNLOADING ticks, no COMPLETE — simulates connection drop.
        patcher = fake_sse_stream(
            [
                (
                    "status",
                    '{"status":"downloading","job_id":"job-0","file_name":"a.raw",'
                    '"bytes_downloaded":100,"total_bytes":200}',
                ),
            ]
        )
        client = AsyncMock()
        with patch("msdatasets.client.aconnect_sse", patcher):
            with pytest.raises(DownloadError, match="stream closed"):
                await _stream_repo_import(client, result)

    @pytest.mark.asyncio
    @patch("msdatasets.client.get_api_url", return_value="https://api.example.com")
    async def test_done_event_suppresses_guard(
        self, _api, fake_sse_stream, make_repo_response
    ):
        """Explicit ``done`` is authoritative even if local state is behind."""
        result = make_repo_response()  # one PENDING job
        patcher = fake_sse_stream([("done", "")])
        client = AsyncMock()
        with patch("msdatasets.client.aconnect_sse", patcher):
            # No DownloadError raised even though job-0 never saw COMPLETE.
            await _stream_repo_import(client, result)


class TestRepoImportEventModel:
    """RepoImportEvent parses with and without progress fields."""

    def test_parses_without_progress_fields(self):
        """Payload from an older server (no progress fields) must still parse."""
        event = RepoImportEvent.model_validate_json(
            '{"status":"downloading","job_id":"j","file_name":"a.raw"}'
        )
        assert event.bytes_downloaded is None
        assert event.total_bytes is None
        assert event.speed_bps is None

    def test_parses_with_progress_fields(self):
        """New-server payload with progress fields populates them."""
        event = RepoImportEvent.model_validate_json(
            '{"status":"downloading","job_id":"j","file_name":"a.raw",'
            '"bytes_downloaded":1024,"total_bytes":4096,"speed_bps":512.5}'
        )
        assert event.bytes_downloaded == 1024
        assert event.total_bytes == 4096
        assert event.speed_bps == 512.5

    def test_total_bytes_none_for_chunked(self):
        """Chunked transfer: bytes + speed present, total_bytes absent."""
        event = RepoImportEvent.model_validate_json(
            '{"status":"downloading","job_id":"j","file_name":"a.raw",'
            '"bytes_downloaded":2048,"speed_bps":1024.0}'
        )
        assert event.bytes_downloaded == 2048
        assert event.total_bytes is None
        assert event.speed_bps == 1024.0


class TestTriggerRepoImport:
    """Tests for trigger_repo_import."""

    @pytest.mark.asyncio
    @patch("msdatasets.client.get_api_url", return_value="https://api.example.com")
    async def test_404_raises_not_found(self, _api):
        mock_response = MagicMock()
        mock_response.status_code = 404
        client = AsyncMock()
        client.post.return_value = mock_response
        with pytest.raises(DatasetNotFoundError, match="PXD000001"):
            await trigger_repo_import("pride", "PXD000001", client=client)

    @pytest.mark.asyncio
    @patch("msdatasets.client.get_api_url", return_value="https://api.example.com")
    async def test_500_raises_download_error(self, _api):
        mock_response = MagicMock()
        mock_response.status_code = 500
        client = AsyncMock()
        client.post.return_value = mock_response
        with pytest.raises(DownloadError, match="500"):
            await trigger_repo_import("pride", "PXD000001", client=client)

    @pytest.mark.asyncio
    @patch("msdatasets.client.get_api_url", return_value="https://api.example.com")
    async def test_all_complete_short_circuits(self, _api, make_repo_response):
        response_body = make_repo_response(
            job_statuses=[RepoImportStatus.COMPLETE, RepoImportStatus.COMPLETE]
        )
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = response_body.model_dump()
        client = AsyncMock()
        client.post.return_value = mock_response

        with patch(
            "msdatasets.client._stream_repo_import", new_callable=AsyncMock
        ) as mock_stream:
            result = await trigger_repo_import("pride", "PXD000001", client=client)

        assert result.dataset_id == response_body.dataset_id
        mock_stream.assert_not_called()

    @pytest.mark.asyncio
    @patch("msdatasets.client.get_api_url", return_value="https://api.example.com")
    async def test_pre_failed_jobs_raise_with_details(self, _api, make_repo_response):
        response_body = make_repo_response(
            job_statuses=[RepoImportStatus.COMPLETE, RepoImportStatus.FAILED],
            job_file_names=["a.raw", "b.raw"],
            error_messages=[None, "network"],
        )
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = response_body.model_dump()
        client = AsyncMock()
        client.post.return_value = mock_response

        with pytest.raises(DownloadError, match="b.raw: network"):
            await trigger_repo_import("pride", "PXD000001", client=client)

    @pytest.mark.asyncio
    @patch("msdatasets.client.get_api_url", return_value="https://api.example.com")
    async def test_streams_when_jobs_pending(self, _api, make_repo_response):
        response_body = make_repo_response(
            job_statuses=[RepoImportStatus.PENDING, RepoImportStatus.DOWNLOADING]
        )
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = response_body.model_dump()
        client = AsyncMock()
        client.post.return_value = mock_response

        with patch(
            "msdatasets.client._stream_repo_import", new_callable=AsyncMock
        ) as mock_stream:
            result = await trigger_repo_import("pride", "PXD000001", client=client)

        assert result.dataset_id == response_body.dataset_id
        mock_stream.assert_called_once()
        # on_status + on_progress kwargs forwarded
        assert "on_status" in mock_stream.call_args.kwargs
        assert "on_progress" in mock_stream.call_args.kwargs

    @pytest.mark.asyncio
    @patch("msdatasets.client.get_api_url", return_value="https://api.example.com")
    async def test_source_enum_coercion(self, _api, make_repo_response):
        response_body = make_repo_response(
            source=RepoSource.MASSIVE,
            accession="MSV000078787",
            job_statuses=[RepoImportStatus.COMPLETE],
        )
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = response_body.model_dump()
        client = AsyncMock()
        client.post.return_value = mock_response

        result = await trigger_repo_import(
            RepoSource.MASSIVE, "MSV000078787", client=client
        )
        assert result.source == RepoSource.MASSIVE
        # URL should use the enum value
        called_url = client.post.call_args.args[0]
        assert "/repositories/massive/projects/MSV000078787/dataset" in called_url

    @pytest.mark.asyncio
    @patch("msdatasets.client.get_api_url", return_value="https://api.example.com")
    async def test_filenames_forwarded_in_body(self, _api, make_repo_response):
        response_body = make_repo_response(job_statuses=[RepoImportStatus.COMPLETE])
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = response_body.model_dump()
        client = AsyncMock()
        client.post.return_value = mock_response

        await trigger_repo_import(
            "pride",
            "PXD000001",
            filenames=["a.raw", "b.mzML"],
            client=client,
        )
        body = client.post.call_args.kwargs["json"]
        assert body == {"filenames": ["a.raw", "b.mzML"]}

    @pytest.mark.asyncio
    @patch("msdatasets.client.get_api_url", return_value="https://api.example.com")
    async def test_no_filenames_excluded_from_body(self, _api, make_repo_response):
        response_body = make_repo_response(job_statuses=[RepoImportStatus.COMPLETE])
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = response_body.model_dump()
        client = AsyncMock()
        client.post.return_value = mock_response

        await trigger_repo_import("pride", "PXD000001", client=client)
        body = client.post.call_args.kwargs["json"]
        assert body == {}
