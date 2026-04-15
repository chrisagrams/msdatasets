# Usage

## CLI

The `msdatasets` command is installed as a console script by the package.

### `msdatasets download`

```
msdatasets [-v | -vv] download <dataset_id>
                               [--force]
                               [--no-progress]
                               [--workers N]
                               [--store-as {mszx,msz,mzml}]
                               [-o DIR]
```

| Flag               | Description                                                               |
|--------------------|---------------------------------------------------------------------------|
| `dataset_id`       | UUID or repository spec (see [Dataset identifiers](#dataset-identifiers)) |
| `--force`          | Re-download even if files are already cached                              |
| `--no-progress`    | Suppress the progress bar                                                 |
| `--workers N`      | Parallel downloads (default: `4`)                                         |
| `--store-as FMT`   | On-disk format: `mszx` (default), `msz`, or `mzml`                        |
| `-o, --output DIR` | Write files directly to `DIR`; bypasses the cache, no `{id}` subdir       |
| `-v`, `-vv`        | Increase verbosity to `INFO` / `DEBUG` (applies to the whole command)     |

Exit codes: `0` on success, `1` on dataset-not-found or download error,
`2` on argparse errors.

## Dataset identifiers

The `dataset_id` argument accepts three shapes:

| Shape                      | Example                                       |
|----------------------------|-----------------------------------------------|
| UUID                       | `550e8400-e29b-41d4-a716-446655440000`        |
| Repository accession       | `pride/PXD075509` or `massive/MSV000101460`   |
| Accession with file subset | `pride/PXD075509[19HCD_3.mzML,other.mzML]`    |

When a repository spec is supplied, the server imports the project on
demand (if not already imported) and streams progress over SSE until all
files are ready. The call is idempotent — re-running for the same project
returns the existing dataset.

## Storage formats

`--store-as` controls the on-disk extension and the client-side conversion
performed by `mstransfer`:

| Format | Extension | Description                                          |
|--------|-----------|------------------------------------------------------|
| `mszx` | `.mszx`   | Raw archive shipped by the server (no conversion)    |
| `msz`  | `.msz`    | Inner MSZ extracted from the MSZX archive            |
| `mzml` | `.mzML`   | Fully decompressed mzML                              |

The cache is keyed by the target filename. Switching `--store-as` for a
dataset you've already downloaded triggers a re-download in the new format
rather than reusing a stale artifact in a different format.

## Caching

Downloads are cached at `<cache_dir>/<dataset_id>/`, where `cache_dir` is
resolved in this order:

1. `$MS_DATASETS_CACHE`
2. `$MS_HOME/datasets`
3. `~/.ms/datasets`

Each dataset directory also contains a `manifest.json` written after the
manifest is fetched, for offline inspection.

To write somewhere else entirely, pass `-o/--output DIR` on the CLI or
`output_dir=Path(...)` to the Python functions. The directory is used
as-is — no `{dataset_id}` subdirectory is added.

## Environment variables

| Variable            | Purpose                                      | Default                   |
|---------------------|----------------------------------------------|---------------------------|
| `MS_API_URL`        | Server base URL                              | `https://datasets.lab.gy` |
| `MS_DATASETS_CACHE` | Explicit cache directory                     | —                         |
| `MS_HOME`           | Alternative cache root (`$MS_HOME/datasets`) | `~/.ms`                   |

## Python API

### `download_dataset`

Download a dataset **by UUID** and return a `Dataset`:

```python
from msdatasets import download_dataset

ds = download_dataset(
    "550e8400-e29b-41d4-a716-446655440000",
    filenames=["file1.mzML"],     # optional subset
    store_as="mzml",              # mszx (default) | msz | mzml
    max_workers=8,
    force_download=False,
)

print(ds.dataset_name, ds.cache_dir)
for path in ds:
    ...
```

`Dataset` supports `len()`, indexing, and iteration over the downloaded
`Path` objects.

### `download_repo_dataset`

Import a PRIDE or MassIVE project and download the resulting dataset. Use
this for repository accessions — `download_dataset` takes UUIDs only:

```python
from msdatasets import download_repo_dataset, RepoSource

ds = download_repo_dataset(
    RepoSource.PRIDE,          # or "pride" / "massive"
    "PXD075509",
    filenames=["19HCD_3.mzML"],
    store_as="mszx",
)
```

### `load_dataset` / `load_repo_dataset`

Convenience wrappers that return an
`mscompress.datasets.torch.MSCompressDataset`. They require
`pip install 'msdatasets[torch]'`:

```python
from msdatasets import load_dataset

# UUIDs and repository specs both work; filenames come from the [...] syntax
dataset = load_dataset("pride/PXD075509[19HCD_3.mzML]")
```

`load_dataset` and `load_repo_dataset` do not expose `store_as` or
`output_dir`; they download to the default cache as `.mszx` and hand the
cache directory to `MSCompressDataset`.

### Exceptions

- `DatasetNotFoundError` — server returned 404 for the dataset or project.
- `DownloadError` — network or server failure during download.
- `ExtractionError` — server-side extraction task failed
  (subclass of `DownloadError`).
