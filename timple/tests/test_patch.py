import timple

import pytest

import datetime
import numpy as np


def test_enable_disable(mpl):
    # test enabling and disabing Timple to ensure it works as expected and
    # can be removed again
    tmpl = timple.Timple(mpl)

    td = np.timedelta64(1, 'D')
    # not yet enabled
    assert mpl.units._is_natively_supported(td)
    assert np.timedelta64 not in mpl.units.registry
    assert datetime.timedelta not in mpl.units.registry

    tmpl.enable()
    assert not mpl.units._is_natively_supported(td)
    assert np.timedelta64 in mpl.units.registry
    assert datetime.timedelta in mpl.units.registry

    tmpl.disable()
    assert mpl.units._is_natively_supported(td)
    assert np.timedelta64 not in mpl.units.registry
    assert datetime.timedelta not in mpl.units.registry


def test_date2num_pandas_nat(pd, mpl):
    tmpl = timple.Timple(mpl)

    test_case = [pd.Timestamp('1970-01-03'), pd.NaT]
    expected = [2.0, np.nan]

    with pytest.raises(ValueError):
        np.testing.assert_equal(mpl.dates.date2num(test_case), expected)

    tmpl.enable(pd_nat_dates_support=True)
    np.testing.assert_equal(mpl.dates.date2num(test_case), expected)


def test_mpl_default_functionality(mpl):
    tmpl = timple.Timple(mpl)
    tmpl.enable()

    ret = 0

    print("\n\n#####\nRunning subtests for matplotlib")
    print("Testing 'matplotlib.tests.test_dates'")
    ret += mpl.test(verbosity=1, argv=['matplotlib.tests.test_dates']).value

    print("\nTesting 'matplotlib.tests.test_units'")
    ret += mpl.test(verbosity=1, argv=['matplotlib.tests.test_units']).value

    if ret != 0:
        raise Exception("Subtests for matplotlib failed!"
                        "Have you installed matplotlib from sources?")
