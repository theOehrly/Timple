import pytest
import numpy as np
import datetime

# NOTE:
# imports for timple and matplotlib are automatically cleaned up before
# EACH individual test. Therefore, imports for timple and matplotlib
# need to be done inside a test and on a per test basis


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
    import matplotlib as mpl
    import timple
    tmpl = timple.Timple()
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


def test_date2num_pandas_nat(pd):
    import timple
    import matplotlib.dates as mdates
    tmpl = timple.Timple()
    test_case = [pd.Timestamp('1970-01-03'), pd.NaT]
    expected = [2.0, np.nan]

    with pytest.raises(ValueError):
        np.testing.assert_equal(mdates.date2num(test_case), expected)

    tmpl.enable(pd_nat_dates_support=True)
    np.testing.assert_equal(mdates.date2num(test_case), expected)
