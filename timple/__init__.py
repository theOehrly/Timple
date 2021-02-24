import numpy as np
import datetime

from .timedelta import TimedeltaConverter, ConciseTimedeltaConverter
from . import patches


class Timple:
    def __init__(self, mpl, converter='default'):
        """
        Parameters
        ----------
        mpl: Imported instance of matplotlib
        converter (str): one of 'auto', 'concise', 'default'
            Default will use the same value as set in matplotlib's rcParams
            for 'date.converter'. If this value does not exist it will fall
            back to 'auto'.
        """
        if converter not in ('default', 'auto', 'concise'):
            raise ValueError("Invalid value for keyword argument 'converter'")
        self._mpl = mpl
        self._revert_funcs = list()
        self._converter = converter

    def enable(self, pd_nat_dates_support=False):
        """
        Enables Timple by patching matplotlib and registering either
        `timedelta.TimedeltaConverter` or
        `timedelta.ConciseTimedeltaConverter`.

        After this, you can plot timedelta values and matplotlib
        will automatically choose appropriate locators and formatter which
        Timple provides.

        If you want more control over the result, you can always specify the
        formatter and locator for a plot manually. See `timedelta` for more
        information.

        Parameters
        ----------
        pd_nat_dates_support: (Optional) Patch matplotlib internal functionality to support
            pandas NaT values when plotting dates too.
        """
        revert_units_patch = self._patch_supported_units()
        self._revert_funcs.append(revert_units_patch)

        revert_converters = self._add_converters()
        self._revert_funcs.append(revert_converters)

        if pd_nat_dates_support:
            revert_date2num = self._patch_date2num()
            self._revert_funcs.append(revert_date2num)

    def disable(self):
        """
        Disables Timple. Reverts the applied patch and unregisters the
        timedelta converter.
        """
        for revert in self._revert_funcs:
            revert()

    def _add_converters(self):
        # register the appropriate matplotlib converter
        # return a function that reverts this change
        if self._converter == 'default':
            try:
                # option only exists in matplotlib >= 3.4.0
                conv_type = self._mpl.rcParams['date.converter']
                if conv_type not in ('auto', 'concise'):
                    raise ValueError
            except (KeyError, ValueError):
                conv_type = 'auto'
        else:
            conv_type = self._converter

        if conv_type == 'concise':
            timedelta_converter = ConciseTimedeltaConverter
        else:
            timedelta_converter = TimedeltaConverter

        conv_inst = timedelta_converter()
        self._mpl.units.registry[np.timedelta64] = conv_inst
        self._mpl.units.registry[datetime.timedelta] = conv_inst

        def revert():
            del self._mpl.units.registry[np.timedelta64]
            del self._mpl.units.registry[datetime.timedelta]

        return revert

    def _patch_supported_units(self):
        # patch matplotlibs ._is_natively_supported function
        # timedelta is a Number and therefore 'supported' by default
        # make it 'unsupported' so matplotlib will use the converter
        # return a function that reverts this change
        orig_func = self._mpl.units._is_natively_supported

        patched = patches.get_patched_is_natively_supported(self._mpl)

        self._mpl.units._is_natively_supported = patched

        def revert():
            self._mpl.units._is_natively_supported = orig_func

        return revert

    def _patch_date2num(self):
        # patch matplotlibs dates.date2num function to add support for
        # pandas nat
        # return a function that reverts this change
        orig_func = self._mpl.dates.date2num

        patched = patches.get_patched_date2num(self._mpl)
        self._mpl.dates.date2num = patched

        def revert():
            self._mpl.dates.date2num = orig_func

        return revert
