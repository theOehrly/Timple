"""
Matplotlib provides sophisticated date plotting capabilities, standing on the
shoulders of python :mod:`datetime` and the add-on module :mod:`dateutil`.

.. _date-format:

Matplotlib date format
----------------------

Matplotlib represents dates using floating point numbers specifying the number
of days since a default epoch of 1970-01-01 UTC; for example,
1970-01-01, 06:00 is the floating point number 0.25. The formatters and
locators require the use of `datetime.datetime` objects, so only dates between
year 0001 and 9999 can be represented.  Microsecond precision
is achievable for (approximately) 70 years on either side of the epoch, and
20 microseconds for the rest of the allowable range of dates (year 0001 to
9999). The epoch can be changed at import time via `.dates.set_epoch` or
:rc:`dates.epoch` to other dates if necessary; see
:doc:`/gallery/ticks_and_spines/date_precision_and_epochs` for a discussion.

.. note::

   Before Matplotlib 3.3, the epoch was 0000-12-31 which lost modern
   microsecond precision and also made the default axis limit of 0 an invalid
   datetime.  In 3.3 the epoch was changed as above.  To convert old
   ordinal floats to the new epoch, users can do::

     new_ordinal = old_ordinal + mdates.date2num(np.datetime64('0000-12-31'))


There are a number of helper functions to convert between :mod:`datetime`
objects and Matplotlib dates:

.. currentmodule:: matplotlib.dates

.. autosummary::
   :nosignatures:

   datestr2num
   date2num
   num2date
   num2timedelta
   drange
   set_epoch
   get_epoch

.. note::

   Like Python's `datetime.datetime`, Matplotlib uses the Gregorian calendar
   for all conversions between dates and floating point numbers. This practice
   is not universal, and calendar differences can cause confusing
   differences between what Python and Matplotlib give as the number of days
   since 0001-01-01 and what other software and databases yield.  For
   example, the US Naval Observatory uses a calendar that switches
   from Julian to Gregorian in October, 1582.  Hence, using their
   calculator, the number of days between 0001-01-01 and 2006-04-01 is
   732403, whereas using the Gregorian calendar via the datetime
   module we find::

     In [1]: date(2006, 4, 1).toordinal() - date(1, 1, 1).toordinal()
     Out[1]: 732401

All the Matplotlib date converters, tickers and formatters are timezone aware.
If no explicit timezone is provided, :rc:`timezone` is assumed.  If you want to
use a custom time zone, pass a `datetime.tzinfo` instance with the tz keyword
argument to `num2date`, `~.Axes.plot_date`, and any custom date tickers or
locators you create.

A wide range of specific and general purpose date tick locators and
formatters are provided in this module.  See
:mod:`matplotlib.ticker` for general information on tick locators
and formatters.  These are described below.

The dateutil_ module provides additional code to handle date ticking, making it
easy to place ticks on any kinds of dates.  See examples below.

.. _dateutil: https://dateutil.readthedocs.io

Date tickers
------------

Most of the date tickers can locate single or multiple values.  For example::

    # import constants for the days of the week
    from matplotlib.dates import MO, TU, WE, TH, FR, SA, SU

    # tick on mondays every week
    loc = WeekdayLocator(byweekday=MO, tz=tz)

    # tick on mondays and saturdays
    loc = WeekdayLocator(byweekday=(MO, SA))

In addition, most of the constructors take an interval argument::

    # tick on mondays every second week
    loc = WeekdayLocator(byweekday=MO, interval=2)

The rrule locator allows completely general date ticking::

    # tick every 5th easter
    rule = rrulewrapper(YEARLY, byeaster=1, interval=5)
    loc = RRuleLocator(rule)

The available date tickers are:

* `MicrosecondLocator`: Locate microseconds.

* `SecondLocator`: Locate seconds.

* `MinuteLocator`: Locate minutes.

* `HourLocator`: Locate hours.

* `DayLocator`: Locate specified days of the month.

* `WeekdayLocator`: Locate days of the week, e.g., MO, TU.

* `MonthLocator`: Locate months, e.g., 7 for July.

* `YearLocator`: Locate years that are multiples of base.

* `RRuleLocator`: Locate using a `matplotlib.dates.rrulewrapper`.
  `.rrulewrapper` is a simple wrapper around dateutil_'s `dateutil.rrule` which
  allow almost arbitrary date tick specifications.  See :doc:`rrule example
  </gallery/ticks_and_spines/date_demo_rrule>`.

* `AutoDateLocator`: On autoscale, this class picks the best `DateLocator`
  (e.g., `RRuleLocator`) to set the view limits and the tick locations.  If
  called with ``interval_multiples=True`` it will make ticks line up with
  sensible multiples of the tick intervals.  E.g. if the interval is 4 hours,
  it will pick hours 0, 4, 8, etc as ticks.  This behaviour is not guaranteed
  by default.

Date formatters
---------------

The available date formatters are:

* `AutoDateFormatter`: attempts to figure out the best format to use.  This is
  most useful when used with the `AutoDateLocator`.

* `ConciseDateFormatter`: also attempts to figure out the best format to use,
  and to make the format as compact as possible while still having complete
  date information.  This is most useful when used with the `AutoDateLocator`.

* `DateFormatter`: use `~datetime.datetime.strftime` format strings.

* `IndexDateFormatter`: date plots with implicit *x* indexing.
"""
import datetime
import string
import math

import numpy as np

import matplotlib as mpl
from matplotlib import ticker, units

try:
    # only available for matplotlib version >= 3.4.0
    from matplotlib.dates import _wrap_in_tex
except ImportError:
    def _wrap_in_tex(text):
        # Braces ensure dashes are not spaced like binary operators.
        return '$\\mathdefault{' + text.replace('-', '{-}') + '}$'

__all__ = ('num2timedelta', 'timedelta2num',
           'TimedeltaFormatter', 'ConciseTimedeltaFormatter',
           'AutoTimedeltaFormatter',
           'TimedeltaLocator', 'AutoTimedeltaLocator', 'FixedTimedeltaLocator',
           'TimedeltaConverter', 'ConciseTimedeltaConverter')


"""
Time-related constants.
"""
HOURS_PER_DAY = 24.
MIN_PER_HOUR = 60.
SEC_PER_MIN = 60.

MINUTES_PER_DAY = MIN_PER_HOUR * HOURS_PER_DAY

SEC_PER_HOUR = SEC_PER_MIN * MIN_PER_HOUR
SEC_PER_DAY = SEC_PER_HOUR * HOURS_PER_DAY

MUSECONDS_PER_DAY = 1e6 * SEC_PER_DAY


def _td64_to_ordinalf(d):
    """
    Convert `numpy.timedelta64` or an ndarray of those types to a number of
    days as float. Roundoff is float64 precision. Practically: microseconds
    for up to 292271 years, milliseconds for larger time spans.
    (see `numpy.timedelta64`).
    """

    # the "extra" ensures that we at least allow the dynamic range out to
    # seconds.  That should get out to +/-2e11 years.
    dseconds = d.astype('timedelta64[s]')
    extra = (d - dseconds).astype('timedelta64[ns]')
    dt = dseconds.astype(np.float64)
    dt += extra.astype(np.float64) / 1.0e9
    dt = dt / SEC_PER_DAY

    NaT_int = np.timedelta64('NaT').astype(np.int64)
    d_int = d.astype(np.int64)
    try:
        dt[d_int == NaT_int] = np.nan
    except TypeError:
        if d_int == NaT_int:
            dt = np.nan
    return dt


def timedelta2num(t):
    """
    Convert timedelta objects to Matplotlib timedeltas.

    Parameters
    ----------
    t : `datetime.timedelta`, `numpy.timedelta64` or `pandas.Timedelta`
        or sequences of these

    Returns
    -------
    float or sequence of floats
        Number of days
    """
    if hasattr(t, "values"):
        # this unpacks pandas series or dataframes...
        t = t.values

    # make an iterable, but save state to unpack later:
    iterable = np.iterable(t)
    if not iterable:
        t = [t]

    t = np.asarray(t)
    if not t.size:
        # deals with an empty array...
        return t.astype('float64')

    if hasattr(t.take(0), 'value'):
        # elements are pandas objects; temporarily convert data to numbers
        # pandas nat is defined as the minimum value of int64,
        # replace all 'min int' values with the string 'nat' and convert the
        # array to the dtype of the first non-nat value
        values = np.asarray([x.value for x in t], dtype='object')
        nat_mask = (np.iinfo('int64').min == values)
        if not all(nat_mask):
            _ttype = t[~nat_mask].take(0).to_numpy().dtype
        else:
            _ttype = 'timedelta64[us]'  # default in case of all NaT
        t = np.where(nat_mask, 'nat', values).astype(_ttype)

    # convert to datetime64 or timedelta64 arrays, if not already:
    if not np.issubdtype(t.dtype, np.timedelta64):
        t = t.astype('timedelta64[us]')

    t = _td64_to_ordinalf(t)

    return t if iterable else t[0]


_ordinalf_to_timedelta_np_vectorized = np.vectorize(
    lambda x: datetime.timedelta(days=x), otypes="O")


def num2timedelta(x):
    """
    Convert number of days to a `~datetime.timedelta` object.

    If *x* is a sequence, a sequence of `~datetime.timedelta` objects will
    be returned.

    Parameters
    ----------
    x : float, sequence of floats
        Number of days. The fraction part represents hours, minutes, seconds.

    Returns
    -------
    `datetime.timedelta` or list[`datetime.timedelta`]
    """
    return _ordinalf_to_timedelta_np_vectorized(x).tolist()


# def pd_timedelta2np(t):
#     """
#     Convert `pandas.Timedelta` objects to `numpy.timedelta64`
#
#     Parameters
#     ----------
#     t: `pandas.Timedelta` or `pandas.NaT` or sequences of these
#
#     Returns
#     -------
#     `numpy.timedelta64` or sequence of `numpy.timedelta64`
#     """
#     iterable = np.iterable(t)
#     if not iterable:
#         t = [t]
#
#     conv = list()
#     for val in t:
#         c = val.to_numpy()
#         if np.isnat(c):
#             # pandas.NaT.to_numpy() returns datetime64('nat')
#             # but here we need timedelta64('nat')
#             conv.append(np.timedelta64('nat'))
#         else:
#             conv.append(c)
#
#     t = np.asarray(conv)
#     return t if iterable else t[0]


class _TimedeltaFormatTemplate(string.Template):
    # formatting template for datetime-like formatter strings
    delimiter = '%'


def strftimedelta(td, fmt_str):
    """
    Return a string representing a timedelta, controlled by an explicit
    format string.

    Arguments
    ---------
    td : datetime.timedelta
    fmt_str : str
        format string
    """
    # *_t values are not partially consumed by there next larger unit
    # e.g. for timedelta(days=1.5): d=1, h=12, H=36
    s_t = td.total_seconds()
    sign = '-' if s_t < 0 else ''
    s_t = abs(s_t)

    d, s = divmod(s_t, SEC_PER_DAY)
    m_t, s = divmod(s, SEC_PER_MIN)
    h, m = divmod(m_t, MIN_PER_HOUR)
    h_t, _ = divmod(s_t, SEC_PER_HOUR)

    us = td.microseconds
    ms, us = divmod(us, 1e3)

    # create correctly zero padded string for substitution
    # last one is a special for correct day(s) plural
    values = {'d': int(d),
              'H': int(h_t),
              'M': int(m_t),
              'S': int(s_t),
              'h': '{:02d}'.format(int(h)),
              'm': '{:02d}'.format(int(m)),
              's': '{:02d}'.format(int(s)),
              'ms': '{:03d}'.format(int(ms)),
              'us': '{:03d}'.format(int(us)),
              'day': 'day' if d == 1 else 'days'}

    try:
        result = _TimedeltaFormatTemplate(fmt_str).substitute(**values)
    except KeyError:
        raise ValueError(f"Invalid format string '{fmt_str}' for timedelta")
    return sign + result


def strftdnum(td_num, fmt_str):
    """
    Return a string representing a matplotlib internal float based timedelta,
    controlled by an explicit format string.

    Arguments
    ---------
    td_num : float
        timedelta in matplotlib float representation
    fmt_str : str
        format string
    """
    td = num2timedelta(td_num)
    return strftimedelta(td, fmt_str)


class TimedeltaFormatter(ticker.Formatter):
    """
    Format a tick (in days) with a format string or using as custom
    `.FuncFormatter`.

    This `.Formatter` formats ticks according to a fixed specification.
    Ticks can optionally be offset to generate shorter tick labels.

    .. note:: The format string for timedeltas works similar to a
        `~datetime.datetime.strftime` format string but they are NOT the
        same and NOT compatible.

    Examples
    --------
    .. plot::

        import datetime
        import matplotlib.dates as mdates

        base = datetime.timedelta(days=100)
        timedeltas = np.array([base + datetime.timedelta(minutes=(4 * i))
                              for i in range(720)])
        N = len(timedeltas)
        np.random.seed(19680801)
        y = np.cumsum(np.random.randn(N))

        fig, ax = plt.subplots(constrained_layout=True)
        locator = mdates.AutoTimedeltaLocator()
        formatter = mdates.TimedeltaFormatter("%H:%m", offset_on='days',
                                              offset_fmt="%d %day")
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

        ax.plot(timedeltas, y)
        ax.set_title('Timedelta Formatter with Offset on Days')
    """
    # TODO explain format strings somewhere
    def __init__(self, fmt, *, offset_on=None, offset_fmt=None, usetex=None):
        """
        Parameters
        ----------
        fmt : str or callable
            a format string or a callable for formatting the tick values

        offset_on : str, optional
            One of ``('days', 'hours', 'minutes', 'seconds')``

            Specifies how to offset large values; default is no offset.
            If ``offset_on`` is set but ``offset_fmt`` is not, the offset will
            be applied but not shown.

        offset_fmt : str or callable
            A format string or a callable for formatting the offset string.
            This also requires ``offset_on`` to be specified.

        usetex : bool, default: :rc:`text.usetex`
            To enable/disable the use of TeX's math mode for rendering the
            results of the formatter.
        """
        super().__init__()
        if (offset_on is None) and (offset_fmt is not None):
            raise ValueError("'offset_fmt' requires 'offset_on to be "
                             "specified.'")
        self.fmt = fmt
        self.offset_fmt = offset_fmt
        self.offset_on = offset_on
        self.offset_string = ''
        self._usetex = (usetex if usetex is not None else
                        mpl.rcParams['text.usetex'])

    def __call__(self, x, pos=None):
        return self._format_tick(x, pos)

    def _format_tick(self, x, pos=None):
        # format a single tick value
        if isinstance(self.fmt, str):
            return strftdnum(x, self.fmt)
        elif callable(self.fmt):
            return self.fmt(x, pos)
        else:
            raise TypeError('Unexpected type passed to {0!r} as string '
                            'formatter.'.format(self))

    def get_offset(self):
        return self.offset_string

    def _offset_values(self, values):
        # offset the values based on data or view limits
        ref = min(values)
        # evaluate data interval if available
        # the leftmost (i.e. smallest) data value inside the view window
        # is used as a reference value
        # the offset is calculated so that the value of the reference is zero
        # on the offset level
        # Example: 1 day 12:00, 2 days 00:00, 2 days 12:00; offset on day
        # Resulting offset: 1 day
        # Resulting values: 0 days 12:00, 1 day 00:00, 1 day 12:00
        if self.axis is not None:
            data_ref = min(self.axis.get_data_interval())
            ref = max(data_ref, ref)
            # ref based on data if fully zoomed out else based on view

        # prevent floating point errors; 13 digits > musecond precision
        ref = round(ref, 13)

        # calculate offset based on the reference value and the level
        # specified by self.offset_on
        if self.offset_on == 'days':
            offset = math.floor(ref)
        elif self.offset_on == 'hours':
            offset = math.floor(ref*HOURS_PER_DAY)/HOURS_PER_DAY
        elif self.offset_on == 'minutes':
            offset = math.floor(ref*MINUTES_PER_DAY)/MINUTES_PER_DAY
        elif self.offset_on == 'seconds':
            offset = math.floor(ref*SEC_PER_DAY)/SEC_PER_DAY
        else:
            raise ValueError("Invalid value passed to {0!r} for "
                             "'offset_on'".format(self))

        # return the values with the offset applied and the offset itself
        return [val - offset for val in values], offset

    def format_ticks(self, values):
        offset = None
        if self.offset_on is not None:
            # apply an offset to all values
            values, offset = self._offset_values(values)
        # create labels based on the values after the offset was applied
        result = [self._format_tick(val) for val in values]

        if self._usetex:
            result = [_wrap_in_tex(label) for label in result]

        if self.offset_fmt is not None:
            # format the applied offset itself so it can be displayed
            # as axis offset
            if isinstance(self.offset_fmt, str):
                offset_str = strftdnum(offset, self.offset_fmt)
            elif callable(self.offset_fmt):
                offset_str = self.offset_fmt(offset)
            else:
                raise TypeError('Unexpected type passed to {0!r} as offset '
                                'string formatter.'.format(self))

            if self._usetex:
                offset_str = _wrap_in_tex(offset_str)

            self.offset_string = offset_str

        return result


class ConciseTimedeltaFormatter(ticker.Formatter):
    """
    A `.Formatter` which attempts to figure out the best format to use for the
    timedelta, and to make it as compact as possible, but still be complete.
    This is most useful when used with the `AutoTimedeltaLocator`::

    >>> locator = AutoTimedeltaLocator()
    >>> formatter = ConciseTimedeltaFormatter(locator)

    The formatter will make use of the axis offset. Depending on the tick
    frequency of the locator, the axis offset as well as the format for ticks
    and offset will be determined.

    There are 5 tick levels. These are the same as the base units of the
    locator. The levels are ``('days', 'hours', 'minutes', 'seconds',
    'microseconds')``.
    For each tick level a format string, an offset format string and the
    offset position can be specified. Else, the defaults will be used.


    Parameters
    ----------
    locator : `.Locator`
        Locator that the axis is using.

    formats : list of 5 strings, optional
        Format strings for tick labels.
        TODO: ref explanation of codes
        The default is::

            ["%d %day",
             "%H:00",
             "%H:%m",
             "%M:%s.0",
             "%S.%ms%us"]

    offset_formats : list of 5 tuples, optional
        A combination of ``(offset format, offset position)`` where the offset
        format is a format string similar to the tick format string.
        Offset position specifies on which level the offset should be applied.
        See the ``offset_fmt=`` and ``offset_on=`` arguments of
        `TimedeltaFormatter`.
        The default is::

            [(None, None),
             ("%d %day", "days"),
             ("%d %day", "days"),
             ("%d %day, %h:00", "hours"),
             ("%d %day, %h:%m", "minutes")]

        For no offset, set both values of a level to None. To apply an offset
        but don't show it, set only the format string to None.

    show_offset : bool, default: True
        Whether to show the offset or not.

    usetex : bool, default: :rc:`text.usetex`
        To enable/disable the use of TeX's math mode for rendering the results
        of the formatter.
    """
    def __init__(self, locator, formats=None, offset_formats=None,
                 show_offset=True, *, usetex=None):
        self._locator = locator
        self.defaultfmt = "%d %days"  # TODO
        self.show_offset = show_offset

        # 5 formatting levels
        self._levels = (1,
                        1/HOURS_PER_DAY,
                        1/MINUTES_PER_DAY,
                        1/SEC_PER_DAY,
                        1/MUSECONDS_PER_DAY)
        if formats:
            if len(formats) != 6:
                raise ValueError('formats argument must be a list of '
                                 '5 format strings (or None)')
            self.formats = formats
        else:
            self.formats = [
                "%d %day",
                "%H:00",
                "%H:%m",
                "%M:%s.0",
                "%S.%ms%us"
            ]

        if offset_formats:
            if len(offset_formats) != 5:
                raise ValueError('offset_formats argument must be a list of '
                                 '5 format strings (or None)')
            self.offset_formats = offset_formats
        else:
            self.offset_formats = [
                (None, None),
                ("%d %day", "days"),
                ("%d %day", "days"),
                ("%d %day, %h:00", "hours"),
                ("%d %day, %h:%m", "minutes")
            ]
        self.offset_str = ''
        self._usetex = (usetex if usetex is not None else
                        mpl.rcParams['text.usetex'])

    def __call__(self, x, pos=None):
        formatter = TimedeltaFormatter(self.defaultfmt, usetex=self._usetex)
        return formatter(x, pos=pos)

    def format_ticks(self, values):
        try:
            locator_unit_scale = float(self._locator._get_unit())
        except AttributeError:
            locator_unit_scale = 1
        # get the level index corresponding to the locator unit scale and
        # select the appropriate format strings and offset position
        i = self._levels.index(locator_unit_scale)
        fmt = self.formats[i]
        offset_fmt, offset_on = self.offset_formats[i]
        formatter = TimedeltaFormatter(fmt, offset_fmt=offset_fmt,
                                       offset_on=offset_on,
                                       usetex=self._usetex)
        formatter.set_axis(self.axis)
        labels = formatter.format_ticks(values)
        if self.show_offset:
            self.offset_str = formatter.get_offset()

        return labels

    def get_offset(self):
        return self.offset_str


class AutoTimedeltaFormatter(ticker.Formatter):
    """
    A `.Formatter` which attempts to figure out the best format to use. This
    is most useful when used with the `AutoTimedeltaLocator`.

    The AutoTimedeltaFormatter has a scale dictionary that maps the scale
    of the tick (the distance in days between one major tick) and a
    format string.  The default looks like this::

        self.scaled = {
            1: "%d %day",
            1 / HOURS_PER_DAY: '%d %day, %h:%m',
            1 / MINUTES_PER_DAY: '%d %day, %h:%m',
            1 / SEC_PER_DAY: '%d %day, %h:%m:%s',
            1e3 / MUSECONDS_PER_DAY: '%d %day, %h:%m:%s.%ms',
            1 / MUSECONDS_PER_DAY: '%d %day, %h:%m:%s.%ms%us',
        }

    The algorithm picks the key in the dictionary that is >= the
    current scale and uses that format string.  You can customize this
    dictionary by doing::

    >>> locator = AutoTimedeltaLocator()
    >>> formatter = AutoTimedeltaFormatter(locator)
    >>> formatter.scaled[1/(24.*60.)] = '%M:%S' # only show min and sec

    A custom `.FuncFormatter` can also be used. See `AutoDateLocator` for an
    example of this.

    Parameters
    ----------
    locator : `.Locator`
        Locator that this axis is using

    defaultfmt : str
        The default format to use if none of the values in ``self.scaled``
        are greater than the unit returned by ``locator._get_unit()``.

    usetex : bool, default: :rc:`text.usetex`
        To enable/disable the use of TeX's math mode for rendering the
        results of the formatter. If any entries in ``self.scaled`` are set
        as functions, then it is up to the customized function to enable or
        disable TeX's math mode itself.
    """
    def __init__(self, locator, defaultfmt='%d %day, %h:%m', *, usetex=None):
        """
        Autoformat the timedelta labels.
        """
        self._locator = locator
        self.defaultfmt = defaultfmt
        self._usetex = (usetex if usetex is not None else
                        mpl.rcParams['text.usetex'])

        self.scaled = {
            1: "%d %day",
            1 / HOURS_PER_DAY: '%d %day, %h:%m',
            1 / MINUTES_PER_DAY: '%d %day, %h:%m',
            1 / SEC_PER_DAY: '%d %day, %h:%m:%s',
            1e3 / MUSECONDS_PER_DAY: '%d %day, %h:%m:%s.%ms',
            1 / MUSECONDS_PER_DAY: '%d %day, %h:%m:%s.%ms%us',
        }

    def _set_locator(self, locator):
        self._locator = locator

    def __call__(self, x, pos=0):
        try:
            locator_unit_scale = float(self._locator._get_unit())
        except AttributeError:
            locator_unit_scale = 1
        # Pick the first scale which is greater than the locator unit.
        fmt = next((fmt for scale, fmt in sorted(self.scaled.items())
                    if scale >= locator_unit_scale),
                   self.defaultfmt)

        if isinstance(fmt, str):
            self._formatter = TimedeltaFormatter(fmt, usetex=self._usetex)
            result = self._formatter(x, pos)
        elif callable(fmt):
            result = fmt(x, pos)
        else:
            raise TypeError('Unexpected type passed to {0!r}.'.format(self))

        return result


class TimedeltaLocator(ticker.MultipleLocator):
    """
    Determines the tick locations when plotting timedeltas.

    This class is subclassed by other Locators and
    is not meant to be used on its own.

    Attributes
    ----------
    base_units : list

        list of all supported base units

        By default those are::

            self.base_units = ['days',
                               'hours',
                               'minutes',
                               'seconds',
                               'microseconds']

    base_factors : dict

        mapping of base units to conversion factors to convert from the
        default day representation to hours, seconds, ...
    """
    def __init__(self):
        super().__init__()
        self.base_factors = {'days': 1,
                             'hours': HOURS_PER_DAY,
                             'minutes': MINUTES_PER_DAY,
                             'seconds': SEC_PER_DAY,
                             'microseconds': MUSECONDS_PER_DAY}
        # don't rely on order of dict
        self.base_units = ['days',
                           'hours',
                           'minutes',
                           'seconds',
                           'microseconds']  # mind docstring for fixed locator

    def datalim_to_td(self):
        """Convert axis data interval to timedelta objects."""
        tmin, tmax = self.axis.get_data_interval()
        if tmin > tmax:
            tmin, tmax = tmax, tmin

        return num2timedelta(tmin), num2timedelta(tmax)

    def viewlim_to_td(self):
        """Convert the view interval to timedelta objects."""
        tmin, tmax = self.axis.get_view_interval()
        if tmin > tmax:
            tmin, tmax = tmax, tmin
        return num2timedelta(tmin), num2timedelta(tmax)

    def _create_locator(self, base, interval):
        """
        Create an instance of :class:`ticker.MultipleLocator` using base unit
        and interval

        Parameters
        ----------
        base : {'days', 'hours', 'minutes',  'seconds', 'microseconds'}
        interval : int or float

        Returns
        -------
        instance of :class:`matplotlib.ticker.MultipleLocator`
        """
        factor = self.base_factors[base]

        locator = ticker.MultipleLocator(base=interval/factor)
        locator.set_axis(self.axis)

        if self.axis is not None:
            locator.set_view_interval(*self.axis.get_view_interval())
            locator.set_data_interval(*self.axis.get_data_interval())

        return locator

    def _get_unit(self):
        """
        Return how many days a unit of the locator is; used for
        intelligent autoscaling.
        """
        return 1

    def _get_interval(self):
        """
        Return the number of units for each tick.
        """
        return 1

    def nonsingular(self, vmin, vmax):
        """
        Given the proposed upper and lower extent, adjust the range
        if it is too close to being singular (i.e. a range of ~0).
        """
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            # Except if there is no data, then use 1 day - 2 days as default.
            return (timedelta2num(datetime.timedelta(days=1)),
                    timedelta2num(datetime.timedelta(days=2)))
        if vmax < vmin:
            vmin, vmax = vmax, vmin
        unit = self._get_unit()
        interval = self._get_interval()
        if abs(vmax - vmin) < 1e-6:
            vmin -= 2 * unit * interval
            vmax += 2 * unit * interval
        return vmin, vmax


class FixedTimedeltaLocator(TimedeltaLocator):
    """
    Make ticks in an interval of the base unit.

    Examples::

      # Ticks every 2 days
      locator = TimedeltaLocatorManual('days', 2)

      # Ticks every 20 seconds
      locator = TimedeltaLocatorManual('seconds', 20)
    """
    def __init__(self, base_unit, interval):
        """
        Parameters
        ----------
        base_unit: {'days', 'hours', 'minutes', 'seconds', 'microseconds'}
        interval: `int` or `float`
        """
        super().__init__()
        if base_unit not in self.base_units:
            raise ValueError(f"base must be one of {self.base_units}")
        self.base = base_unit
        self.interval = interval
        self._freq = 1 / self.base_factors[base_unit]

    def __call__(self):
        # docstring inherited
        locator = self._create_locator(self.base, self.interval)
        return locator()

    def tick_values(self, vmin, vmax):
        return self._create_locator(self.base, self.interval)\
            .tick_values(vmin, vmax)

    def _get_unit(self):
        return self._freq

    def nonsingular(self, vmin, vmax):
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            # Except if there is no data, then use 1 day - 2 days as default.
            return (timedelta2num(datetime.timedelta(days=1)),
                    timedelta2num(datetime.timedelta(days=2)))
        if vmax < vmin:
            vmin, vmax = vmax, vmin
        unit = self._get_unit()
        interval = self._get_interval()
        # factor adjusts unit from days to hours, seconds, ... if necessary
        factor = self.base_factors[self.base]
        if abs(vmax - vmin) < 1e-6 / factor:
            vmin -= 2 * unit * interval / factor
            vmax += 2 * unit * interval / factor
        return vmin, vmax


class AutoTimedeltaLocator(TimedeltaLocator):
    """
    This class automatically finds the best base unit and interval for setting
    view limits and tick locations.

    Attributes
    ----------
    intervald : dict

        Mapping of tick frequencies to multiples allowed for that ticking.
        The default is ::

            self.intervald = {
                'days': [1, 2, 5, 10, 20, 25, 50, 100, 200, 500, 1000, 2000,
                         5000, 10000, 20000, 50000, 100000, 200000, 500000,
                         1000000],
                'hours': [1, 2, 3, 4, 6, 8, 12],
                'minutes': [1, 2, 3, 5, 10, 15, 20, 30],
                'seconds': [1, 2, 3, 5, 10, 15, 20, 30],
                'microseconds': [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000,
                                 2000, 5000, 10000, 20000, 50000, 100000,
                                 200000, 500000, 1000000],
            }

        The interval is used to specify multiples that are appropriate for
        the frequency of ticking. For instance, every 12 hours is sensible
        for hourly ticks, but for minutes/seconds, 15 or 30 make sense.

        When customizing, you should only modify the values for the existing
        keys. You should not add or delete entries.

        Example for forcing ticks every 3 hours::

            locator = AutoTimedeltaLocator()
            locator.intervald['hours'] = [3]  # only show every 3 hours

        For forcing ticks in one specific interval only,
        :class:`FixedTimedeltaLocator` might be preferred.
    """
    def __init__(self, minticks=5, maxticks=None):
        """
        Parameters
        ----------
        minticks : int
            The minimum number of ticks desired; controls whether ticks occur
            daily, hourly, etc.
        maxticks : int
            The maximum number of ticks desired; controls the interval between
            ticks (ticking every other, every 3, etc.).  For fine-grained
            control, this can be a dictionary mapping individual base units
            ('days', 'hours', etc.) to their own maximum
            number of ticks.  This can be used to keep the number of ticks
            appropriate to the format chosen in `AutoDateFormatter`. Any
            frequency not specified in this dictionary is given a default
            value.
        """
        super().__init__()
        self.intervald = {
            'days': [1, 2, 5, 10, 20, 25, 50, 100, 200, 500, 1000, 2000,
                     5000, 10000, 20000, 50000, 100000, 200000, 500000,
                     1000000],
            'hours': [1, 2, 3, 4, 6, 8, 12],
            'minutes': [1, 2, 3, 5, 10, 15, 20, 30],
            'seconds': [1, 2, 3, 5, 10, 15, 20, 30],
            'microseconds': [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000,
                             5000, 10000, 20000, 50000, 100000, 200000, 500000,
                             1000000],
        }  # mind the default in the docstring
        self.minticks = minticks
        self.maxticks = {'days': 11, 'hours': 12,
                         'minutes': 11, 'seconds': 11, 'microseconds': 8}
        if maxticks is not None:
            try:
                self.maxticks.update(maxticks)
            except TypeError:
                # Assume we were given an integer. Use this as the maximum
                # number of ticks for every frequency and create a
                # dictionary for this
                self.maxticks = dict.fromkeys(self.base_units, maxticks)
        self._freq = 1.0  # default is daily

    def __call__(self):
        # docstring inherited
        tmin, tmax = self.viewlim_to_td()
        locator = self.get_locator(tmin, tmax)
        return locator()

    def tick_values(self, vmin, vmax):
        locator = self.get_locator(vmin, vmax)
        return locator.tick_values(vmin, vmax)

    def nonsingular(self, vmin, vmax):
        # whatever is thrown at us, we can scale the unit.
        # But default nonsingular date plots at an ~4 day period.
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            # Except if there is no data, then use 1 day - 2 days as default.
            return (timedelta2num(datetime.timedelta(days=1)),
                    timedelta2num(datetime.timedelta(days=2)))
        if vmax < vmin:
            vmin, vmax = vmax, vmin
        if vmin == vmax:
            vmin -= 2
            vmax += 2
        return vmin, vmax

    def _get_unit(self):
        return self._freq

    def get_locator(self, vmin, vmax):
        """
        Create the best locator based on the given limits.

        This will choose the settings for a
        :class:`matplotlib.ticker.MultipleLocator`
        based on the available base units and associated intervals.
        The locator is created so that there are as few ticks as possible
        but more ticks than specified with min_ticks in init.

        Returns
        -------
        instance of :class:`matplotlib.ticker.MultipleLocator`
        """
        tdelta = vmax - vmin

        # take absolute difference
        if vmin > vmax:
            tdelta = -tdelta

        tdelta = timedelta2num(tdelta)

        # find an appropriate base unit and interval for it
        base = self._get_base(tdelta)
        factor = self.base_factors[base]
        norm_delta = tdelta * factor
        self._freq = 1/factor
        interval = self._get_interval_for_base(norm_delta, base)

        return self._create_locator(base, interval)

    def _get_base(self, tdelta):
        # find appropriate base unit for given time delta
        base = 'days'  # fallback
        for base in self.base_units:
            try:
                factor = self.base_factors[base]
                if tdelta * factor >= self.minticks:
                    break
            except KeyError:
                continue  # intervald was modified
        return base

    def _get_interval_for_base(self, norm_delta, base):
        # find appropriate interval for given delta and min ticks
        # norm_delta = tdelta * base_factor
        base_intervals = self.intervald[base]
        interval = 1  # fallback (and for static analysis)
        # for interval in reversed(base_intervals):
        #     if norm_delta // interval >= self.minticks:
        for interval in base_intervals:
            if norm_delta // interval <= self.maxticks[base]:
                break

        return interval


class TimedeltaConverter(units.ConversionInterface):
    """
    Converter for `datetime.timedelta` and `numpy.timedelta64` data.

    The 'unit' tag for such data is None.
    """

    def __init__(self):
        super().__init__()

    def axisinfo(self, unit, axis):
        """
        Return the `~matplotlib.units.AxisInfo`.

        The *unit* and *axis* arguments are required but not used.
        """
        majloc = AutoTimedeltaLocator()
        majfmt = AutoTimedeltaFormatter(majloc)
        datemin = datetime.timedelta(days=1)
        datemax = datetime.timedelta(days=2)

        return units.AxisInfo(majloc=majloc, majfmt=majfmt, label='',
                              default_limits=(datemin, datemax))

    @staticmethod
    def convert(value, unit, axis):
        """
        If *value* is not already a number or sequence of numbers, convert it
        with `timedelta2num`.

        The *unit* and *axis* arguments are not used.
        """
        return timedelta2num(value)


class ConciseTimedeltaConverter(TimedeltaConverter):
    # docstring inherited
    def __init__(self, formats=None, offset_formats=None, show_offset=True):
        super().__init__()
        self._formats = formats
        self._offset_formats = offset_formats
        self._show_offset = show_offset

    def axisinfo(self, unit, axis):
        # docstring inherited
        majloc = AutoTimedeltaLocator()
        majfmt = ConciseTimedeltaFormatter(majloc, formats=self._formats,
                                           offset_formats=self._offset_formats,
                                           show_offset=self._show_offset)
        datemin = datetime.timedelta(days=1)
        datemax = datetime.timedelta(days=2)

        return units.AxisInfo(majloc=majloc, majfmt=majfmt, label='',
                              default_limits=(datemin, datemax))
