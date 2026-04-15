"""Configuration: paths, URLs, and environment variables."""

from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger("msdatasets")

_DEFAULT_API_URL = "https://datasets.lab.gy"


def get_api_url() -> str:
    """Return the base API URL.

    Resolution: ``MS_API_URL`` env var, or the default production URL.
    """
    url = os.environ.get("MS_API_URL", _DEFAULT_API_URL)
    log.debug("API URL: %s", url)
    return url


def get_cache_dir() -> Path:
    """Return the root cache directory for downloaded datasets.

    Resolution order:
    1. ``MS_DATASETS_CACHE`` env var
    2. ``MS_HOME`` env var + ``/datasets``
    3. ``~/.ms/datasets``
    """
    if env := os.environ.get("MS_DATASETS_CACHE"):
        log.debug("Cache dir from MS_DATASETS_CACHE: %s", env)
        return Path(env)
    if env := os.environ.get("MS_HOME"):
        path = Path(env) / "datasets"
        log.debug("Cache dir from MS_HOME: %s", path)
        return path
    path = Path.home() / ".ms" / "datasets"
    log.debug("Cache dir (default): %s", path)
    return path


def get_dataset_dir(dataset_id: str) -> Path:
    """Return the cache directory for a specific dataset."""
    return get_cache_dir() / dataset_id
