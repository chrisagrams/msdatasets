# Getting Started

## Installation

Install the base package from PyPI:

```bash
pip install msdatasets
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add msdatasets
```

`load_dataset` and `load_repo_dataset` return a PyTorch-compatible
`MSCompressDataset`. To use them, install the `torch` extra:

```bash
pip install 'msdatasets[torch]'
```

## Your first download

### By server UUID

```bash
msdatasets download 550e8400-e29b-41d4-a716-446655440000
```

### From a PRIDE project

```bash
msdatasets download pride/PXD075509
```

### A specific file, stored as mzML

```bash
msdatasets download pride/PXD075509[19HCD_3.mzML] --store-as mzml
```

By default files land in `~/.ms/datasets/{dataset_id}/`. See
[Usage › Caching](usage.md#caching) to change the behavior.

## From Python

```python
from msdatasets import download_dataset, download_repo_dataset

# By UUID
ds = download_dataset("550e8400-e29b-41d4-a716-446655440000")

# From a repository
ds = download_repo_dataset(
    "pride",
    "PXD075509",
    filenames=["19HCD_3.mzML"],
)

print(ds.dataset_name, len(ds))
for path in ds:
    print(path)
```

To load a MSDataset as a PyTorch Dataset (requires `pip install 'msdatasets[torch]'`):

```python
from msdatasets import load_dataset

# load_dataset accepts UUIDs and repository specs
dataset = load_dataset("pride/PXD075509[19HCD_3.mzML]")
```

## Pointing at a different server

The default server is `https://datasets.lab.gy`. Override it with the
`MS_API_URL` environment variable:

```bash
export MS_API_URL=http://localhost:8000
msdatasets download pride/PXD075509
```

Or in Python, before importing the download functions:

```python
import os
os.environ["MS_API_URL"] = "http://localhost:8000"

from msdatasets import download_dataset
...
```

See [Usage › Environment variables](usage.md#environment-variables) for the
full list.

## Development setup

Clone the repository and install in development mode:

```bash
git clone https://github.com/chrisagrams/msdatasets.git
cd msdatasets
uv sync --extra dev --extra docs
uv run pre-commit install
```

Run the test suite (90% coverage gate):

```bash
uv run pytest
```

Serve the docs locally:

```bash
uv run mkdocs serve
```
