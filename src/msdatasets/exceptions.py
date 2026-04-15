"""Exceptions for the msdatasets package."""


class DatasetNotFoundError(Exception):
    """Raised when the server returns 404 for a dataset."""


class DownloadError(Exception):
    """Raised on network or server failures during download."""


class ExtractionError(DownloadError):
    """Raised when a server-side extraction task fails."""
