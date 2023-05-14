import pytest
import numpy as np
import datetime
import sys
import subprocess


# NOTE:
# DO NOT make relative imports here
# WRONG: import core
# RIGHT: from timple import core


def test_enable_disable():
    import timple
    import matplotlib.units as munits

    # test enabling and disabing Timple to ensure it works as expected and
    # can be removed again
    tmpl = timple.Timple()

    td = np.timedelta64(1, 'D')
    # not yet enabled
    assert munits._is_natively_supported(td)
    assert np.timedelta64 not in munits.registry
    assert datetime.timedelta not in munits.registry

    tmpl.enable()
    assert not munits._is_natively_supported(td)
    assert np.timedelta64 in munits.registry
    assert datetime.timedelta in munits.registry

    tmpl.disable()
    assert munits._is_natively_supported(td)
    assert np.timedelta64 not in munits.registry
    assert datetime.timedelta not in munits.registry


def test_mpl_default_functionality():
    # run this test in a subprocess to ensure a clean state
    # NOTE: the matplotlib tests require a local development/editable
    # matplotlib installation form sources!
    ret = subprocess.call(
        'python -c '
        '"'
        'from timple.tests import test_patch; '
        'test_patch._subprocess_matplotlib_subtests()'
        '"', shell=True)
    assert ret == 0


def _subprocess_matplotlib_subtests():
    import os
    import matplotlib as mpl
    import timple
    tmpl = timple.Timple()
    tmpl.enable(pd_nat_dates_support=True)

    mpl_path = os.path.dirname(mpl.__file__)
    # to be run in a subprocess
    ret = 0
    ret += pytest.main(['-x', mpl_path, 'matplotlib.tests.test_dates'])
    ret += pytest.main(['-x', mpl_path, 'matplotlib.tests.test_units'])
    if ret != 0:
        sys.exit(1)


def test_natively_supported_pandas_to_numpy(pd):
    import timple
    tmpl = timple.Timple()
    tmpl.enable()

    from matplotlib.units import _is_natively_supported

    series = pd.Series((pd.Timedelta(days=1), pd.NaT))
    assert not _is_natively_supported(series)

    dataframe = pd.DataFrame(
        {'data1': (1, 2),
         'data2': (pd.Timedelta(days=1), pd.Timedelta(days=2))}
    )
    assert not _is_natively_supported(dataframe['data2'].to_numpy())


def test_td2num_pandas_nat(pd):
    import timple
    tmpl = timple.Timple()
    import matplotlib.dates as mdates

    test_case = [pd.Timestamp('1970-01-03'), pd.NaT]
    expected = [2.0, np.nan]

    with pytest.raises((ValueError, TypeError)):
        # raises ValueError or TypeError depending on the version of mpl
        np.testing.assert_equal(mdates.date2num(test_case), expected)

    tmpl.enable(pd_nat_dates_support=True)
    np.testing.assert_equal(mdates.date2num(test_case), expected)
