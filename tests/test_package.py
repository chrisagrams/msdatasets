"""Basic package tests."""

from importlib.metadata import version

import msdatasets


def test_version():
    """Test that version is defined."""
    assert hasattr(msdatasets, "__version__")
    assert isinstance(msdatasets.__version__, str)
    assert msdatasets.__version__ == version("msdatasets")


def test_import():
    """Test that package can be imported."""
    import msdatasets

    assert msdatasets is not None
