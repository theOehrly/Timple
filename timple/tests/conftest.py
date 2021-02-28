import pytest


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


# clean up imports before EACH test automatically
# timple/matplotlib need to be impolrted on a per
# test basis
@pytest.fixture(autouse=True)
def clean_state():
    import sys
    modules = list(sys.modules.keys())
    for m in modules:
        if 'matplotlib' in m or 'timple' in m:
            del sys.modules[m]
