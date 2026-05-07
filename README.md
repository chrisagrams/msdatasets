# msdatasets

[![CI](https://github.com/chrisagrams/msdatasets/actions/workflows/ci.yml/badge.svg)](https://github.com/chrisagrams/msdatasets/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/chrisagrams/msdatasets/graph/badge.svg)](https://codecov.io/gh/chrisagrams/msdatasets)
[![PyPI version](https://badge.fury.io/py/msdatasets.svg)](https://pypi.org/project/msdatasets/)

A unified dataset framework for mass spectrometry.

`msdatasets` is a Python client and CLI for downloading mass spectrometry
datasets from the msdatasets server. Datasets are fetched by server UUID or
by repository accession (PRIDE, MassIVE), cached on disk, and optionally
loaded as a PyTorch `Dataset` for training pipelines.

## Features

- Download by server UUID or by PRIDE / MassIVE accession — the server
  imports and converts remote projects on demand
- Choose the on-disk format per download: `mszx` (raw archive), `msz`
  (inner compressed MS data), or `mzml` (fully decompressed)
- Parallel downloads with a live progress bar
- Filename subsets via `accession[file1.raw,file2.mzML]` syntax
- Server-side extraction is tracked over SSE until files are ready
- Optional PyTorch integration via the `torch` extra

## Installation

```bash
pip install msdatasets              # base install
pip install 'msdatasets[torch]'     # with PyTorch integration
pip install 'msdatasets[hf]'        # with HuggingFace Hub support
```

## Quick start

### CLI

```bash
# By server UUID
msdatasets download 550e8400-e29b-41d4-a716-446655440000

# From a PRIDE project
msdatasets download pride/PXD075509

# Subset of files, stored as mzML
msdatasets download pride/PXD075509[19HCD_3.mzML] --store-as mzml

# Write directly to a directory instead of the shared cache
msdatasets download massive/MSV000101460 -o ./my-data

# From a HuggingFace dataset repo of .mszx files
msdatasets download hf/myorg/proteomics-bench
msdatasets download 'hf/myorg/proteomics-bench[run_01.mszx,run_02.mszx]'
```

### Python

```python
from msdatasets import download_dataset, download_repo_dataset

# By UUID
ds = download_dataset("550e8400-e29b-41d4-a716-446655440000")
print(ds.dataset_name, len(ds), "files")
for path in ds:
    ...

# By PRIDE accession (filename subset, stored as mzML)
ds = download_repo_dataset(
    "pride",
    "PXD075509",
    filenames=["19HCD_3.mzML"],
    store_as="mzml",
)
```

### PyTorch

```python
from msdatasets import load_dataset

# Returns an mscompress.datasets.torch.MSCompressDataset.
# Accepts UUIDs, repository specs, and HuggingFace specs.
dataset = load_dataset("pride/PXD075509[19HCD_3.mzML]")
dataset = load_dataset("hf/myorg/proteomics-bench")
```

### HuggingFace Hub

```python
from msdatasets import download_hf_dataset

ds = download_hf_dataset(
    "myorg/proteomics-bench",
    filenames=["run_01.mszx", "run_02.mszx"],   # optional subset
    revision="v1.0",                              # optional branch/tag/commit
    token=None,                                   # falls back to HF_TOKEN
)
print(ds.cache_dir, len(ds), "files")
```

HF downloads land at `~/.ms/datasets/hf/<owner>/<repo>/` by default. Files are
stored as-is — `--store-as` conversion is not supported for HF specs in this
version.

## Configuration

| Environment variable | Purpose                                      | Default                   |
|----------------------|----------------------------------------------|---------------------------|
| `MS_API_URL`         | Server base URL                              | `https://datasets.lab.gy` |
| `MS_DATASETS_CACHE`  | Explicit cache directory                     | —                         |
| `MS_HOME`            | Alternative cache root (`$MS_HOME/datasets`) | `~/.ms`                   |

Full CLI reference, storage-format details, and Python API are in the
[documentation](https://chrisagrams.github.io/msdatasets/).

## Development

```bash
git clone https://github.com/chrisagrams/msdatasets.git
cd msdatasets
uv sync --extra dev --extra docs
uv run pre-commit install
uv run pytest
```

Pre-commit runs `ruff`, `mypy`, and `pytest` (90% coverage gate). CI runs on
Python 3.10, 3.11, and 3.12.

## License

MIT
