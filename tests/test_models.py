"""Tests for msdatasets.models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from msdatasets.models import (
    DatasetPart,
    NotificationEvent,
    RepoDatasetRequest,
    RepoImportEvent,
    RepoImportStatus,
    RepoSource,
    TaskEvent,
)


class TestEnums:
    """Tests for the string enums."""

    def test_repo_source_values(self):
        assert RepoSource("pride") == RepoSource.PRIDE
        assert RepoSource("massive") == RepoSource.MASSIVE

    def test_repo_source_string_behavior(self):
        # Enum should be usable as a plain string (inherits from str).
        assert RepoSource.PRIDE.value == "pride"
        assert f"{RepoSource.MASSIVE.value}" == "massive"

    def test_repo_import_status_values(self):
        expected = {
            "pending",
            "downloading",
            "converting",
            "indexing",
            "complete",
            "failed",
        }
        assert {s.value for s in RepoImportStatus} == expected


class TestIsTerminal:
    """Tests for the is_terminal() methods on notification events."""

    def test_base_notification_event_is_not_terminal(self):
        assert NotificationEvent().is_terminal() is False

    @pytest.mark.parametrize("state", ["complete", "failed"])
    def test_task_event_terminal_states(self, state):
        assert TaskEvent(state=state).is_terminal() is True

    @pytest.mark.parametrize("state", ["queued", "running", "pending"])
    def test_task_event_non_terminal_states(self, state):
        assert TaskEvent(state=state).is_terminal() is False

    @pytest.mark.parametrize(
        "status", [RepoImportStatus.COMPLETE, RepoImportStatus.FAILED]
    )
    def test_repo_import_event_terminal(self, status):
        ev = RepoImportEvent(status=status, job_id="j1")
        assert ev.is_terminal() is True

    @pytest.mark.parametrize(
        "status",
        [
            RepoImportStatus.PENDING,
            RepoImportStatus.DOWNLOADING,
            RepoImportStatus.CONVERTING,
            RepoImportStatus.INDEXING,
        ],
    )
    def test_repo_import_event_non_terminal(self, status):
        ev = RepoImportEvent(status=status, job_id="j1")
        assert ev.is_terminal() is False


class TestRepoDatasetRequest:
    """Tests for RepoDatasetRequest serialization."""

    def test_default_filenames_is_none(self):
        req = RepoDatasetRequest()
        assert req.filenames is None

    def test_exclude_none_drops_filenames(self):
        req = RepoDatasetRequest()
        assert req.model_dump(exclude_none=True) == {}

    def test_with_filenames_serializes(self):
        req = RepoDatasetRequest(filenames=["a.raw", "b.mzML"])
        assert req.model_dump(exclude_none=True) == {"filenames": ["a.raw", "b.mzML"]}


class TestDatasetPartFrozen:
    """DatasetPart uses ``frozen=True`` — mutation should fail."""

    def test_cannot_mutate(self):
        part = DatasetPart(
            part_index=0,
            item_id="id",
            filename="f.mszx",
            num_indices=1,
            extract_url="/x",
            download_url="/y",
        )
        with pytest.raises(ValidationError):
            part.filename = "other.mszx"  # type: ignore[misc]
