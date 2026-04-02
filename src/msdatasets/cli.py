"""Command-line interface for msdatasets."""

from __future__ import annotations

import argparse
import logging
import sys


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``msdatasets`` CLI."""
    parser = argparse.ArgumentParser(
        prog="msdatasets",
        description="A unified dataset framework for mass spectrometry",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (-v for info, -vv for debug)",
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- download subcommand ---
    dl_parser = subparsers.add_parser("download", help="Download a dataset by ID")
    dl_parser.add_argument("dataset_id", help="The dataset identifier (UUID)")
    dl_parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Re-download files even if they already exist locally",
    )
    dl_parser.add_argument(
        "--no-progress",
        action="store_true",
        default=False,
        help="Disable the progress bar",
    )
    dl_parser.add_argument(
        "--workers",
        type=int,
        default=4,
        metavar="N",
        help="Number of parallel downloads (default: 4)",
    )

    args = parser.parse_args(argv)

    _configure_logging(args.verbose)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "download":
        return _cmd_download(args)

    return 0


def _configure_logging(verbosity: int) -> None:
    """Set up the ``msdatasets`` logger based on CLI verbosity."""
    if verbosity == 0:
        level = logging.WARNING
    elif verbosity == 1:
        level = logging.INFO
    else:
        level = logging.DEBUG

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    logger = logging.getLogger("msdatasets")
    logger.setLevel(level)
    logger.addHandler(handler)


def _cmd_download(args: argparse.Namespace) -> int:
    """Handle the ``download`` subcommand."""
    from rich.console import Console

    from msdatasets.download import download_dataset
    from msdatasets.exceptions import DatasetNotFoundError, DownloadError

    console = Console(stderr=True)

    try:
        ds = download_dataset(
            args.dataset_id,
            force_download=args.force,
            show_progress=not args.no_progress,
            max_workers=args.workers,
        )
    except DatasetNotFoundError:
        console.print(f"[bold red]Error:[/] Dataset not found: {args.dataset_id}")
        return 1
    except DownloadError as exc:
        console.print(f"[bold red]Error:[/] {exc}")
        return 1

    name = ds.dataset_name or ds.dataset_id
    console.print(f"[bold green]Done![/] {name}: {len(ds)} file(s) in {ds.cache_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
