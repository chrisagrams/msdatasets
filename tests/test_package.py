"""Basic package tests."""

import msdatasets


def test_version():
    """Test that version is defined."""
    assert hasattr(msdatasets, "__version__")
    assert isinstance(msdatasets.__version__, str)
    assert msdatasets.__version__ == "0.1.0"


def test_import():
    """Test that package can be imported."""
    import msdatasets

    assert msdatasets is not None
