# Getting Started

## Installation

Install from PyPI:

```bash
pip install msdatasets
```

Or with uv:

```bash
uv add msdatasets
```

## Development Installation

Clone the repository and install in development mode:

```bash
git clone https://github.com/chrisagrams/msdatasets.git
cd msdatasets
uv sync --extra dev --extra docs
```

## Running Tests

```bash
uv run pytest
```

## Building Documentation

Serve docs locally:

```bash
uv run mkdocs serve
```

Build static site:

```bash
uv run mkdocs build
```
