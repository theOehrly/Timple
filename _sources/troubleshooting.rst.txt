Troubleshooting steps
=====================

This is a collection of errors that may be encountered.


.. _Changing the Matplotlib style:

Changing the matplotlib style causes the view limits to be set incorrectly
--------------------------------------------------------------------------

Styles modify Matplotlib's rcParams. Some styles may set
``'axes.autolimit_mode': 'round_numbers'``.
Timple represents timedelta as floating point numbers where one day
is represented as ``1.0``.
This means, that if this option is set, the smallest difference between the upper
and lower bound for the view limit can be 1 day. For small timedelta values this
may look like autoscaling is no longer working properly.

One example of a style that sets this option is matplotlib's 'classic' style.

Workaround:
  Make sure to not use a style that sets this option or make sure
  to explicitly unset it after enabling that style.



pytest-mpl produces weird output images where view limits are set incorrectly
-----------------------------------------------------------------------------

pytest-mpl by default uses the matplotlib 'classic' style.
See `Changing the Matplotlib style`_

Workaround:
  Make sure to always explicitly set a working style for pytest-mpl.
  Recommended style: ``'default'``
  See the pytest-mpl documentation for that.