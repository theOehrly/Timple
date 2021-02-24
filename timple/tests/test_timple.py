import timple
from timple import timedelta as tmpldelta

import datetime

import numpy as np
import pytest

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt


tmpl = timple.Timple(mpl)
tmpl.enable()


def test_strftimedelta():
    cases = [
        (datetime.timedelta(days=1), "%d %day, %h:%m", "1 day, 00:00"),
        (datetime.timedelta(days=2.25), "%d %day, %h:%m", "2 days, 06:00"),
        (datetime.timedelta(seconds=362), "%h:%m:%s.%ms", "00:06:02.000"),
        (datetime.timedelta(microseconds=1250), "%s.%ms%us", "00.001250"),
        (datetime.timedelta(days=-0.25), "%h:%m", "-06:00"),
        (datetime.timedelta(days=-1.5), "%d %day, %h:%m", "-1 day, 12:00"),
        (datetime.timedelta(days=2), "%H hours", "48 hours"),
        (datetime.timedelta(days=0.25), "%M min", "360 min"),
        (datetime.timedelta(seconds=362.13), "%S.%ms", "362.130")
    ]

    for td, fmt, expected in cases:
        assert tmpldelta.strftimedelta(td, fmt) == expected


def test_timdelta_formatter():
    def _create_timedelta_locator(td1, td2, fmt, kwargs):
        fig, ax = plt.subplots()

        locator = tmpldelta.AutoTimedeltaLocator()
        formatter = tmpldelta.TimedeltaFormatter(fmt, **kwargs)
        ax.yaxis.set_major_locator(locator)
        ax.yaxis.set_major_formatter(formatter)
        ax.set_ylim(td1, td2)
        fig.canvas.draw()
        sts = [st.get_text() for st in ax.get_yticklabels()]
        offset_text = ax.yaxis.get_offset_text().get_text()
        return sts, offset_text

    td1 = datetime.timedelta(days=100, hours=3, minutes=40)
    results = ([datetime.timedelta(days=141), "%d %day", {},
                ['100 days', '120 days', '140 days', '160 days', '180 days',
                 '200 days', '220 days', '240 days', '260 days'], ""
                ],
               [datetime.timedelta(hours=40), "%H:%m",
                {'offset_fmt': '%d %day', 'offset_on': 'days'},
                ['0:00', '4:00', '8:00', '12:00', '16:00', '20:00',
                 '24:00', '28:00', '32:00', '36:00', '40:00', '44:00'],
                "100 days"
                ],
               [datetime.timedelta(minutes=30), "%M:%s.0",
                {'offset_fmt': '%d %day, %h:%m', 'offset_on': 'hours'},
                ['39:00.0', '42:00.0', '45:00.0', '48:00.0', '51:00.0',
                 '54:00.0', '57:00.0', '60:00.0', '63:00.0',
                 '66:00.0', '69:00.0', '72:00.0'],
                "100 days, 03:00"
                ],
               [datetime.timedelta(seconds=30), "%S.%ms",
                {'offset_fmt': '%d %day, %h:%m', 'offset_on': 'minutes'},
                ['57.000', '60.000', '63.000', '66.000', '69.000',
                 '72.000', '75.000', '78.000', '81.000', '84.000',
                 '87.000', '90.000', '93.000'],
                "100 days, 03:39"
                ],
               [datetime.timedelta(microseconds=600), "%S.%ms%us",
                {'offset_fmt': '%d %day, %h:%m:%s', 'offset_on': 'seconds'},
                ['0.999900', '1.000000', '1.000100', '1.000200', '1.000300',
                 '1.000400', '1.000500', '1.000600', '1.000700'],
                "100 days, 03:39:59"
                ],
               )
    for t_delta, fmt, kwargs, expected, expected_offset in results:
        td2 = td1 + t_delta
        strings, offset_string = _create_timedelta_locator(
            td1, td2, fmt, kwargs
        )
        assert strings == expected
        assert offset_string == expected_offset


def test_timedelta_formatter_usetex():
    formatter = tmpldelta.TimedeltaFormatter("%h:%m", offset_on='days',
                                          offset_fmt="%d %day", usetex=True)
    values = [datetime.timedelta(days=0, hours=12),
              datetime.timedelta(days=1, hours=0),
              datetime.timedelta(days=1, hours=12),
              datetime.timedelta(days=2, hours=0)]

    labels = formatter.format_ticks(tmpldelta.timedelta2num(values))

    start = '$\\mathdefault{'
    i_start = len(start)
    end = '}$'
    i_end = -len(end)

    def verify(string):
        assert string[:i_start] == start
        assert string[i_end:] == end

    # assert ticks are tex formatted
    for lbl in labels:
        verify(lbl)

    # assert offset is tex formatted
    verify(formatter.get_offset())


def test_concise_timedelta_formatter():
    def _create_concise_timedelta_locator(td1, td2):
        fig, ax = plt.subplots()

        locator = tmpldelta.AutoTimedeltaLocator()
        formatter = tmpldelta.ConciseTimedeltaFormatter(locator)
        ax.yaxis.set_major_locator(locator)
        ax.yaxis.set_major_formatter(formatter)
        ax.set_ylim(td1, td2)
        fig.canvas.draw()
        sts = [st.get_text() for st in ax.get_yticklabels()]
        offset_text = ax.yaxis.get_offset_text().get_text()
        return sts, offset_text

    td1 = datetime.timedelta(days=100, hours=3, minutes=40)
    results = ([datetime.timedelta(days=141),
                ['100 days', '120 days', '140 days', '160 days', '180 days',
                 '200 days', '220 days', '240 days', '260 days'],
                ""
                ],
               [datetime.timedelta(hours=40),
                ['0:00', '4:00', '8:00', '12:00', '16:00', '20:00',
                 '24:00', '28:00', '32:00', '36:00', '40:00', '44:00'],
                "100 days"
                ],
               [datetime.timedelta(minutes=30),
                ['3:39', '3:42', '3:45', '3:48', '3:51', '3:54',
                 '3:57', '4:00', '4:03', '4:06', '4:09', '4:12'],
                "100 days"
                ],
               [datetime.timedelta(seconds=30),
                ['39:57.0', '40:00.0', '40:03.0', '40:06.0', '40:09.0',
                 '40:12.0', '40:15.0', '40:18.0', '40:21.0', '40:24.0',
                 '40:27.0', '40:30.0', '40:33.0'],
                "100 days, 03:00"
                ],
               [datetime.timedelta(microseconds=600),
                ['59.999900', '60.000000', '60.000100', '60.000200',
                 '60.000300', '60.000400', '60.000500', '60.000600',
                 '60.000700'], "100 days, 03:39"
                ],
               )
    for t_delta, expected, expected_offset in results:
        td2 = td1 + t_delta
        strings, offset_string = _create_concise_timedelta_locator(td1, td2)
        assert strings == expected
        assert offset_string == expected_offset


def test_auto_timedelta_formatter():
    def _create_auto_timedelta_locator(td1, td2):
        fig, ax = plt.subplots()

        locator = tmpldelta.AutoTimedeltaLocator()
        formatter = tmpldelta.AutoTimedeltaFormatter(locator)
        ax.yaxis.set_major_locator(locator)
        ax.yaxis.set_major_formatter(formatter)
        ax.set_ylim(td1, td2)
        fig.canvas.draw()
        sts = [st.get_text() for st in ax.get_yticklabels()]
        offset_text = ax.yaxis.get_offset_text().get_text()
        return sts, offset_text

    td1 = datetime.timedelta(days=100, hours=3, minutes=40)
    results = ([datetime.timedelta(days=141),
                ['100 days', '120 days', '140 days', '160 days', '180 days',
                 '200 days', '220 days', '240 days', '260 days']
                ],
               [datetime.timedelta(hours=40),
                ['100 days, 00:00', '100 days, 04:00', '100 days, 08:00',
                 '100 days, 12:00', '100 days, 16:00', '100 days, 20:00',
                 '101 days, 00:00', '101 days, 04:00', '101 days, 08:00',
                 '101 days, 12:00', '101 days, 16:00', '101 days, 20:00']
                ],
               [datetime.timedelta(minutes=30),
                ['100 days, 03:39', '100 days, 03:42', '100 days, 03:45',
                 '100 days, 03:48', '100 days, 03:51', '100 days, 03:54',
                 '100 days, 03:57', '100 days, 04:00', '100 days, 04:03',
                 '100 days, 04:06', '100 days, 04:09', '100 days, 04:12'],
                ],
               [datetime.timedelta(seconds=30),
                ['100 days, 03:39:57',
                 '100 days, 03:40:00', '100 days, 03:40:03',
                 '100 days, 03:40:06', '100 days, 03:40:09',
                 '100 days, 03:40:12', '100 days, 03:40:15',
                 '100 days, 03:40:18', '100 days, 03:40:21',
                 '100 days, 03:40:24', '100 days, 03:40:27',
                 '100 days, 03:40:30', '100 days, 03:40:33']
                ],
               [datetime.timedelta(microseconds=600),
                ['100 days, 03:39:59.999900', '100 days, 03:40:00.000000',
                 '100 days, 03:40:00.000100', '100 days, 03:40:00.000200',
                 '100 days, 03:40:00.000300', '100 days, 03:40:00.000400',
                 '100 days, 03:40:00.000500', '100 days, 03:40:00.000600',
                 '100 days, 03:40:00.000700']
                ])
    for t_delta, expected in results:
        td2 = td1 + t_delta
        strings, offset_string = _create_auto_timedelta_locator(td1, td2)
        assert strings == expected


# TODO remove
# def test_pd_timedelta2np(pd):
#     pd_td = [pd.Timedelta(seconds=1), pd.NaT]
#     td = np.array([np.timedelta64(1, 's'),
#                    np.timedelta64('nat')])
#     converted = tmpldelta.pd_timedelta2np(pd_td)
#     np.testing.assert_array_equal(td, converted)
#
#
# def test_timedelta2num():
#     results = [(1, datetime.timedelta(days=1)),
#                (0.25, datetime.timedelta(hours=6)),
#                (3 / 86400 / 1000, datetime.timedelta(milliseconds=3)),
#                (np.nan, np.timedelta64('nat')),
#                ([1, 1.5], [datetime.timedelta(days=1),
#                            datetime.timedelta(days=1.5)])]
#     for x, tdelta in results:
#         np.testing.assert_equal(x, tmpldelta.timedelta2num(tdelta))
#
#
# def test_timedelta2num_pandas(pd):
#     results = [(2, {'days': 2}),
#                (0.25, {'hours': 6}),
#                (3 / 86400 / 1000, {'milliseconds': 3})]
#     for x, td_kwargs in results:
#         pd_td = pd.Timedelta(**td_kwargs)
#         np.testing.assert_equal(x, tmpldelta.timedelta2num(pd_td))
#
#
# def test_timedelta2num_pandas_nat(pd):
#     assert np.isnan(tmpldelta.timedelta2num(pd.NaT))


def test_date2num_timedelta(pd):
    cases = ((1, datetime.timedelta(days=1)),
             (0.25, datetime.timedelta(hours=6)),
             (3 / 86400 / 1000, datetime.timedelta(milliseconds=3)),
             ([1, 1.5], [datetime.timedelta(days=1),
                         datetime.timedelta(days=1.5)]),
             (np.nan, np.timedelta64('nat')),
             (2, np.timedelta64(2, 'D')),
             (0.25, np.timedelta64(6, 'h')),
             (3 / 86400 / 1000, np.timedelta64(3, 'ms')),
             ([1, 2], [np.timedelta64(1, 'D'),
                       np.timedelta64(2, 'D')]),
             (2, pd.Timedelta(days=2)),
             (0.25, pd.Timedelta(hours=6)),
             (3 / 86400 / 1000, pd.Timedelta(milliseconds=3)),
             ([1, 1.5], [pd.Timedelta(days=1),
                         pd.Timedelta(days=1.5)]),
             ([], [])  # test
             )

    for expected, tdelta in cases:
        np.testing.assert_equal(tmpldelta.timedelta2num(tdelta), expected)


def test_date2num_pandas_nat(pd):
    cases = (
        (pd.NaT, np.nan),
        ([pd.NaT, pd.Timedelta(days=1)], [np.nan, 1.0])
    )
    for x, expected in cases:
        np.testing.assert_equal(tmpldelta.timedelta2num(x), expected)


def test_auto_timedelta_locator():
    def _create_auto_timedelta_locator(delta1, delta2):
        locator = tmpldelta.AutoTimedeltaLocator()
        locator.create_dummy_axis()
        locator.set_view_interval(tmpldelta.timedelta2num(delta1),
                                  tmpldelta.timedelta2num(delta2))
        return locator

    dt1 = datetime.timedelta(days=100)
    results = ([datetime.timedelta(days=141),
               ['80 days, 0:00:00', '100 days, 0:00:00', '120 days, 0:00:00',
                '140 days, 0:00:00', '160 days, 0:00:00', '180 days, 0:00:00',
                '200 days, 0:00:00', '220 days, 0:00:00', '240 days, 0:00:00',
                '260 days, 0:00:00']
                ],
               [datetime.timedelta(hours=40),
                ['99 days, 20:00:00',
                 '100 days, 0:00:00', '100 days, 4:00:00',
                 '100 days, 8:00:00', '100 days, 12:00:00',
                 '100 days, 16:00:00', '100 days, 20:00:00',
                 '101 days, 0:00:00', '101 days, 4:00:00',
                 '101 days, 8:00:00', '101 days, 12:00:00',
                 '101 days, 16:00:00', '101 days, 20:00:00']
                ],
               [datetime.timedelta(minutes=20),
                ['99 days, 23:58:00',
                 '100 days, 0:00:00', '100 days, 0:02:00', '100 days, 0:04:00',
                 '100 days, 0:06:00', '100 days, 0:08:00', '100 days, 0:10:00',
                 '100 days, 0:12:00', '100 days, 0:14:00', '100 days, 0:16:00',
                 '100 days, 0:18:00', '100 days, 0:20:00', '100 days, 0:22:00']
                ],
               [datetime.timedelta(seconds=40),
                ['99 days, 23:59:55',
                 '100 days, 0:00:00', '100 days, 0:00:05', '100 days, 0:00:10',
                 '100 days, 0:00:15', '100 days, 0:00:20', '100 days, 0:00:25',
                 '100 days, 0:00:30', '100 days, 0:00:35', '100 days, 0:00:40',
                 '100 days, 0:00:45']
                ],
               [datetime.timedelta(microseconds=1500),
                ['100 days, 0:00:00', '100 days, 0:00:00.000200',
                 '100 days, 0:00:00.000400', '100 days, 0:00:00.000600',
                 '100 days, 0:00:00.000800', '100 days, 0:00:00.001000',
                 '100 days, 0:00:00.001200', '100 days, 0:00:00.001400',
                 '100 days, 0:00:00.001600']]
               )

    for t_delta, expected in results:
        dt2 = dt1 + t_delta
        locator = _create_auto_timedelta_locator(dt1, dt2)
        assert list(map(str, mdates.num2timedelta(locator()))) == expected


def test_fixed_timedelta_locator_allowed_base():
    for base in tmpldelta.TimedeltaLocator().base_units:
        # should not raise
        tmpldelta.FixedTimedeltaLocator(base, 1)

    with pytest.raises(ValueError):
        tmpldelta.FixedTimedeltaLocator('lightyear', 1)


def test_fixed_timedelta_locator():
    results = [
        ('days', 0.5, 0.5, ['12:00:00', '1 day, 0:00:00',
                            '1 day, 12:00:00', '2 days, 0:00:00']),
        ('minutes', 20, 1 / mdates.HOURS_PER_DAY,
         ['23:40:00', '1 day, 0:00:00',
          '1 day, 0:20:00', '1 day, 0:40:00',
          '1 day, 1:00:00', '1 day, 1:20:00'])
    ]
    for base, interval, tdelta, expected in results:
        dt0 = datetime.timedelta(days=1)
        dt1 = dt0 + datetime.timedelta(days=tdelta)
        locator = tmpldelta.FixedTimedeltaLocator(base, interval)
        locator.create_dummy_axis()
        locator.set_view_interval(*tmpldelta.timedelta2num([dt0, dt1]))
        assert list(map(str, mdates.num2timedelta(locator()))) == expected


def test_auto_modified_intervald():
    locator = tmpldelta.AutoTimedeltaLocator()
    locator.intervald['hours'] = [3]
    locator.create_dummy_axis()
    dt1 = datetime.timedelta(days=1)
    dt2 = datetime.timedelta(days=3)
    locator.set_view_interval(tmpldelta.timedelta2num(dt1),
                              tmpldelta.timedelta2num(dt2))
    expected = ['21:00:00', '1 day, 0:00:00', '1 day, 3:00:00',
                '1 day, 6:00:00', '1 day, 9:00:00', '1 day, 12:00:00',
                '1 day, 15:00:00', '1 day, 18:00:00', '1 day, 21:00:00',
                '2 days, 0:00:00', '2 days, 3:00:00', '2 days, 6:00:00',
                '2 days, 9:00:00', '2 days, 12:00:00', '2 days, 15:00:00',
                '2 days, 18:00:00', '2 days, 21:00:00', '3 days, 0:00:00',
                '3 days, 3:00:00']
    # auto would usually be using longer intervals for 2 days
    assert list(map(str, mdates.num2timedelta(locator()))) == expected


# new
#
# def test_concise_timedelta_formatter():
#     def _create_concise_timedelta_locator(td1, td2):
#         fig, ax = plt.subplots()
#
#         locator = tmpldelta.AutoTimedeltaLocator()
#         formatter = tmpldelta.ConciseTimedeltaFormatter(locator)
#         ax.yaxis.set_major_locator(locator)
#         ax.yaxis.set_major_formatter(formatter)
#         ax.set_ylim(td1, td2)
#         fig.canvas.draw()
#         sts = [st.get_text() for st in ax.get_yticklabels()]
#         offset_text = ax.yaxis.get_offset_text().get_text()
#         return sts, offset_text
#
#     td1 = datetime.timedelta(days=100, hours=3, minutes=40)
#     results = ([datetime.timedelta(days=141),
#                 ['100 days', '120 days', '140 days', '160 days', '180 days',
#                  '200 days', '220 days', '240 days', '260 days'],
#                 ""
#                 ],
#                [datetime.timedelta(hours=40),
#                 ['4:00', '8:00', '12:00', '16:00', '20:00',
#                  '24:00', '28:00', '32:00', '36:00', '40:00'],
#                 "100 days"
#                 ],
#                [datetime.timedelta(minutes=30),
#                 ['3:42', '3:45', '3:48', '3:51', '3:54',
#                  '3:57', '4:00', '4:03', '4:06', '4:09'],
#                 "100 days"
#                 ],
#                [datetime.timedelta(seconds=30),
#                 ['40:00.0', '40:03.0', '40:06.0', '40:09.0',
#                  '40:12.0', '40:15.0', '40:18.0', '40:21.0',
#                  '40:24.0', '40:27.0', '40:30.0'],
#                 "100 days, 03:00"
#                 ],
#                [datetime.timedelta(microseconds=600),
#                 ['59.999900', '60.000000', '60.000100', '60.000200',
#                  '60.000300', '60.000400', '60.000500', '60.000600',
#                  '60.000700'], "100 days, 03:39"
#                 ],
#                )
#     for t_delta, expected, expected_offset in results:
#         td2 = td1 + t_delta
#         strings, offset_string = _create_concise_timedelta_locator(td1, td2)
#         assert strings == expected
#         assert offset_string == expected_offset
#
#
# def test_date2num_timedelta(pd):
#     cases = ((1, datetime.timedelta(days=1)),
#              (0.25, datetime.timedelta(hours=6)),
#              (3 / 86400 / 1000, datetime.timedelta(milliseconds=3)),
#              ([1, 1.5], [datetime.timedelta(days=1),
#                          datetime.timedelta(days=1.5)]),
#              (np.nan, np.timedelta64('nat')),
#              (2, np.timedelta64(2, 'D')),
#              (0.25, np.timedelta64(6, 'h')),
#              (3 / 86400 / 1000, np.timedelta64(3, 'ms')),
#              ([1, 2], [np.timedelta64(1, 'D'),
#                        np.timedelta64(2, 'D')]),
#              (2, pd.Timedelta(days=2)),
#              (0.25, pd.Timedelta(hours=6)),
#              (3 / 86400 / 1000, pd.Timedelta(milliseconds=3)),
#              ([1, 1.5], [pd.Timedelta(days=1),
#                          pd.Timedelta(days=1.5)]),
#              ([], [])  # test
#              )
#
#     for expected, tdelta in cases:
#         np.testing.assert_equal(mdates.date2num(tdelta), expected)
#
#
# def test_date2num_pandas_nat(pd):
#     cases = (
#         (pd.NaT, np.nan),
#         ([pd.NaT, pd.Timedelta(days=1)], [np.nan, 1.0]),
#         ([pd.Timestamp('1970-01-03'), pd.NaT], [2.0, np.nan])
#     )
#     for x, expected in cases:
#         np.testing.assert_equal(mdates.date2num(x), expected)
#
#
# def test_auto_timedelta_locator():
#     def _create_auto_timedelta_locator(delta1, delta2):
#         locator = tmpldelta.AutoTimedeltaLocator()
#         locator.create_dummy_axis()
#         locator.set_view_interval(mdates.date2num(delta1),
#                                   mdates.date2num(delta2))
#         return locator
#
#     dt1 = datetime.timedelta(days=100)
#     results = ([datetime.timedelta(days=141),
#                 ['100 days, 0:00:00', '120 days, 0:00:00', '140 days, 0:00:00',
#                  '160 days, 0:00:00', '180 days, 0:00:00', '200 days, 0:00:00',
#                  '220 days, 0:00:00', '240 days, 0:00:00', '260 days, 0:00:00']
#                 ],
#                [datetime.timedelta(hours=40),
#                 ['100 days, 0:00:00', '100 days, 4:00:00',
#                  '100 days, 8:00:00', '100 days, 12:00:00',
#                  '100 days, 16:00:00', '100 days, 20:00:00',
#                  '101 days, 0:00:00', '101 days, 4:00:00',
#                  '101 days, 8:00:00', '101 days, 12:00:00',
#                  '101 days, 16:00:00']
#                 ],
#                [datetime.timedelta(minutes=20),
#                 ['100 days, 0:00:00', '100 days, 0:02:00', '100 days, 0:04:00',
#                  '100 days, 0:06:00', '100 days, 0:08:00', '100 days, 0:10:00',
#                  '100 days, 0:12:00', '100 days, 0:14:00', '100 days, 0:16:00',
#                  '100 days, 0:18:00', '100 days, 0:20:00']
#                 ],
#                [datetime.timedelta(seconds=40),
#                 ['100 days, 0:00:00', '100 days, 0:00:05', '100 days, 0:00:10',
#                  '100 days, 0:00:15', '100 days, 0:00:20', '100 days, 0:00:25',
#                  '100 days, 0:00:30', '100 days, 0:00:35', '100 days, 0:00:40']
#                 ],
#                [datetime.timedelta(microseconds=1500),
#                 ['99 days, 23:59:59.999500', '100 days, 0:00:00',
#                  '100 days, 0:00:00.000500', '100 days, 0:00:00.001000',
#                  '100 days, 0:00:00.001500', '100 days, 0:00:00.002000']
#                 ])
#
#     for t_delta, expected in results:
#         dt2 = dt1 + t_delta
#         locator = _create_auto_timedelta_locator(dt1, dt2)
#         assert list(map(str, mdates.num2timedelta(locator()))) == expected
#
#
# def test_timedelta_locators_fixed():
#     dt0 = datetime.timedelta(days=0)
#     results = [
#         [tmpldelta.DayLocatorTimedelta,
#          {'interval': 25},
#          datetime.timedelta(days=137),
#          ['0:00:00', '25 days, 0:00:00', '50 days, 0:00:00',
#           '75 days, 0:00:00', '100 days, 0:00:00',
#           '125 days, 0:00:00', '150 days, 0:00:00']],
#         [mdates.HourLocator,
#          {'byhour': [2, 14]},
#          datetime.timedelta(days=3),
#          ['2:00:00', '14:00:00',
#           '1 day, 2:00:00', '1 day, 14:00:00',
#           '2 days, 2:00:00', '2 days, 14:00:00']],
#         [mdates.MinuteLocator,
#          {'byminute': [0, 20, 40]},
#          datetime.timedelta(hours=2),
#          ['0:00:00', '0:20:00', '0:40:00',
#           '1:00:00', '1:20:00', '1:40:00', '2:00:00']],
#         [mdates.SecondLocator,
#          {'bysecond': [0, 15, 30, 45]},
#          datetime.timedelta(minutes=1),
#          ['0:00:00', '0:00:15', '0:00:30',
#           '0:00:45', '0:01:00']],
#         [mdates.MicrosecondLocator,
#          {'interval': 300000},
#          datetime.timedelta(seconds=2),
#          ['-1 day, 23:59:59.700000', '0:00:00', '0:00:00.300000',
#           '0:00:00.600000', '0:00:00.900000', '0:00:01.200000',
#           '0:00:01.500000', '0:00:01.800000', '0:00:02.100000']]
#     ]
#
#     for loc_cls, kwargs, dt1, expected in results:
#         locator = loc_cls(**kwargs)
#         locator.create_dummy_axis()
#         locator.set_axis(locator.axis)  # only because of MicrosecondLocator
#         locator.set_view_interval(*mdates.date2num([dt0, dt1]))
#         assert list(map(str, mdates.num2timedelta(locator()))) == expected
#
#
# def test_timedelta_epoch_independent():
#     # timedelta is handled as datetime internally
#     # make sure this is independent of the epoch
#     mdates._reset_epoch_test_example()
#     mdates.set_epoch("2000-01-01")
#     test_auto_timedelta_locator()
#     mdates._reset_epoch_test_example()
#
#
# def test_auto_modified_intervald():
#     locator = tmpldelta.AutoTimedeltaLocator()
#     locator.intervald[mdates.HOURLY] = [3]
#     locator.create_dummy_axis()
#     dt1 = datetime.timedelta(days=1)
#     dt2 = datetime.timedelta(days=3)
#     locator.set_view_interval(mdates.date2num(dt1),
#                               mdates.date2num(dt2))
#     expected = ['1 day, 0:00:00', '1 day, 3:00:00', '1 day, 6:00:00',
#                 '1 day, 9:00:00', '1 day, 12:00:00', '1 day, 15:00:00',
#                 '1 day, 18:00:00', '1 day, 21:00:00', '2 days, 0:00:00',
#                 '2 days, 3:00:00', '2 days, 6:00:00', '2 days, 9:00:00',
#                 '2 days, 12:00:00', '2 days, 15:00:00', '2 days, 18:00:00',
#                 '2 days, 21:00:00', '3 days, 0:00:00']
#     # auto would usually be using longer intervals for 2 days
#     with pytest.warns(UserWarning, match="AutoDateLocator was unable"):
#         res = list(map(str, mdates.num2timedelta(locator())))
#     assert res == expected
