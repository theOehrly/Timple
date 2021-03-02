Introduction to Timple
======================

Timple is a module that provides additional functionality for plotting
timedelta-like values with matplotlib.

Timple consists of:

- A main timple object which is used for enabling Timple's
  functionality. (see: :mod:`timple.core`)
  It registers Timple's timedelta converter with matplotlib
  and patches some internal functions of matplotlib.

- A collection of locator and formatter classes for customizing
  plot ticks. (see :mod:`timple.timedelta`)


It supports the following data types:

- ``datetime.timedelta``
- ``numpy.timedelta64``
- ``pandas.Timedelta`` (including ``pandas.NaT``)


While Timple's converters support `pandas.NaT` values, matplotlib does not
support these values natively.
Timple can optionally patch Matplotlib's date functionality to enable support
for this data type when plotting date-like values. This is disabled by default.
See :func:`timple.core.Timple.enable` on how to enable this.



.. toctree::
   :maxdepth: 2
   :caption: Contents:

   timple
   timedelta


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
