# msdatasets

[![CI](https://github.com/chrisagrams/msdatasets/actions/workflows/ci.yml/badge.svg)](https://github.com/chrisagrams/msdatasets/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/chrisagrams/msdatasets/graph/badge.svg)](https://codecov.io/gh/chrisagrams/msdatasets)
[![PyPI version](https://badge.fury.io/py/msdatasets.svg)](https://pypi.org/project/msdatasets/)

A unified dataset framework for mass spectrometry.

## Installation

```bash
pip install msdatasets
```

## Development

Install in development mode with test dependencies:

```bash
uv sync --extra dev
```

Set up pre-commit hooks:

```bash
uv run pre-commit install
```

Run tests:

```bash
uv run pytest
```

## License

MIT
