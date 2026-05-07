"""Data models for the msdatasets package."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from pydantic import BaseModel, ConfigDict


class RepoSource(str, enum.Enum):
    """Supported repository sources for dataset imports."""

    PRIDE = "pride"
    MASSIVE = "massive"


class RepoImportStatus(str, enum.Enum):
    """Status of a repository file import job."""

    PENDING = "pending"
    DOWNLOADING = "downloading"
    CONVERTING = "converting"
    INDEXING = "indexing"
    COMPLETE = "complete"
    FAILED = "failed"


class DatasetPart(BaseModel, frozen=True):
    """A single downloadable part of a dataset."""

    part_index: int
    item_id: str
    filename: str
    num_indices: int
    extract_url: str
    download_url: str


class RepoDatasetRequest(BaseModel):
    """Request body for creating a dataset from a repository project."""

    filenames: list[str] | None = None


class RepoImportJob(BaseModel):
    """Status of a single repository file import job."""

    status: RepoImportStatus
    source: RepoSource | None = None
    file_name: str | None = None
    job_id: str | None = None
    dataset_id: str | None = None
    error_message: str | None = None


class RepoDatasetResponse(BaseModel):
    """Response from the repository dataset creation endpoint."""

    dataset_id: str
    dataset_name: str
    source: RepoSource
    accession: str
    total_files: int
    jobs: list[RepoImportJob]


class NotificationEvent(BaseModel):
    """Base schema for a payload received over an SSE stream."""

    model_config = ConfigDict(extra="ignore")

    def is_terminal(self) -> bool:
        """Return ``True`` when this event represents a terminal state."""
        return False


class TaskEvent(NotificationEvent):
    """SSE payload from ``/tasks/{task_id}/stream``."""

    state: str
    error: str | None = None

    def is_terminal(self) -> bool:
        return self.state in ("complete", "failed")


class RepoImportEvent(NotificationEvent):
    """SSE payload from ``/repositories/datasets/{dataset_id}/stream``."""

    status: RepoImportStatus
    job_id: str
    source: RepoSource | None = None
    file_name: str | None = None
    dataset_id: str | None = None
    error_message: str | None = None

    # Download progress — populated on DOWNLOADING ticks, None elsewhere.
    # total_bytes is None when the server omits Content-Length (chunked transfer).
    # Fields are optional for forward/backward compat with older servers.
    bytes_downloaded: int | None = None
    total_bytes: int | None = None
    speed_bps: float | None = None

    def is_terminal(self) -> bool:
        return self.status in (RepoImportStatus.COMPLETE, RepoImportStatus.FAILED)


class Manifest(BaseModel):
    """Parsed server manifest describing available dataset parts."""

    dataset_id: str
    dataset_name: str | None = None
    total_parts: int
    parts: list[DatasetPart]


@dataclass
class Dataset:
    """Result object returned by `load_dataset`.

    Supports ``len()``, indexing, and iteration over downloaded file paths.
    """

    dataset_id: str
    dataset_name: str | None
    cache_dir: Path
    files: list[Path] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Path:
        return self.files[index]

    def __iter__(self) -> Iterator[Path]:
        return iter(self.files)
