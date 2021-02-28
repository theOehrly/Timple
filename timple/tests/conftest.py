import pytest
import importlib


@pytest.fixture
def pd():
    """Fixture to import and configure pandas."""
    pd = pytest.importorskip('pandas')
    try:
        from pandas.plotting import (
            deregister_matplotlib_converters as deregister)
        deregister()
    except ImportError:
        pass
    return pd


@pytest.fixture
def mpl():
    """Fixture to import matplotlib"""
    import matplotlib as mpl
    from matplotlib import pyplot, dates, units
    importlib.reload(mpl)  # ensure clean import for each test
    importlib.reload(pyplot)
    importlib.reload(dates)
    importlib.reload(units)
    return mpl
