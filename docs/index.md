# msdatasets

A unified dataset framework for mass spectrometry.

`msdatasets` is a Python client and CLI for downloading mass spectrometry
datasets from the msdatasets server. Datasets are fetched by server UUID or
by repository accession (PRIDE, MassIVE), cached on disk, and optionally
loaded as a PyTorch `Dataset` for training pipelines.

## Features

- **Flexible sources.** Download by UUID, or by PRIDE / MassIVE accession —
  the server imports and converts remote projects on demand.
- **Format-aware storage.** Store each download as `.mszx` (raw archive),
  `.msz` (compressed MS data), or `.mzML` (fully decompressed).
- **Parallel downloads** with a live `rich` progress bar.
- **On-demand extraction.** The server queues extraction tasks and streams
  their state back over SSE; the client follows until files are ready.
- **Filename filtering** with `accession[file1.raw,file2.mzML]` syntax.
- **PyTorch integration** via the optional `torch` extra, returning an
  `mscompress.datasets.torch.MSCompressDataset`.

## Installation

```bash
pip install msdatasets              # base install
pip install 'msdatasets[torch]'     # with PyTorch integration
```

## Quick start

From the command line:

```bash
msdatasets download pride/PXD075509[19HCD_3.mzML] --store-as mzml
```

From Python:

```python
from msdatasets import download_repo_dataset

ds = download_repo_dataset(
    "pride",
    "PXD075509",
    filenames=["19HCD_3.mzML"],
    store_as="mzml",
)
for path in ds:
    ...
```

## Next steps

- [Getting Started](getting-started.md) — install, configure, and run your
  first download.
- [Usage](usage.md) — CLI reference, storage formats, repository specs,
  caching, and PyTorch integration.
- [API Reference](api.md) — auto-generated reference for the public Python
  API.

## License

MIT.
