"""Data models for the msdatasets package."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class DatasetPart:
    """A single downloadable part of a dataset."""

    part_index: int
    item_id: str
    filename: str
    num_indices: int
    download_url: str


@dataclass
class Manifest:
    """Parsed server manifest describing available dataset parts."""

    dataset_id: str
    dataset_name: str | None
    total_parts: int
    parts: list[DatasetPart]

    @classmethod
    def from_dict(cls, data: dict) -> Manifest:
        parts = [
            DatasetPart(
                part_index=p["part_index"],
                item_id=p["item_id"],
                filename=p["filename"],
                num_indices=p["num_indices"],
                download_url=p["download_url"],
            )
            for p in data["parts"]
        ]
        return cls(
            dataset_id=data["dataset_id"],
            dataset_name=data.get("dataset_name"),
            total_parts=data["total_parts"],
            parts=parts,
        )


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
