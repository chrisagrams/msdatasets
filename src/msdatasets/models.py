"""Data models for the msdatasets package."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from pydantic import BaseModel


class PrideImportStatus(str, enum.Enum):
    """Status of a PRIDE file import job."""

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


class PrideDatasetRequest(BaseModel):
    """Request body for creating a dataset from a PRIDE project."""

    filenames: list[str] | None = None


class PrideImportJob(BaseModel):
    """Status of a single PRIDE file import job."""

    status: PrideImportStatus
    pride_file_name: str | None = None
    job_id: str | None = None
    error_message: str | None = None


class PrideDatasetResponse(BaseModel):
    """Response from the PRIDE dataset creation endpoint."""

    dataset_id: str
    dataset_name: str
    accession: str
    total_files: int
    jobs: list[PrideImportJob]


class Manifest(BaseModel):
    """Parsed server manifest describing available dataset parts."""

    dataset_id: str
    dataset_name: str | None = None
    total_parts: int
    parts: list[DatasetPart]


@dataclass
class Dataset:
    """Result object returned by :func:`load_dataset`.

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
