"""Code for handling CDF function methods for NSRDB TMY."""

import logging
from copy import deepcopy

import numpy as np
import pandas as pd

from .utilities import drop_leap, make_time_masks

logger = logging.getLogger(__name__)


class Cdf:
    """Class for handling cumulative distribution function methods for TMY."""

    def __init__(self, my_arr, my_time_index, time_masks=None):
        """
        Parameters
        ----------
        my_arr : np.ndarray
            Multi-year (my) daily timeseries 2D array (time, sites) for one
            variable (daily total GHI, daily mean temp, etc...).
        my_time_index : pd.datetimeindex
            Daily datetime index corresponding to the rows in my_arr
        time_masks : dict
            Lookup of boolean masks for a daily timeseries keyed by month and
            year integer.
        """

        if len(my_arr.shape) < 2:
            raise TypeError('Input array is 1D, needs to be 2D (time, sites).')

        self._my_arr = deepcopy(my_arr)
        self._my_time_index = deepcopy(my_time_index)
        self._years = sorted(self._my_time_index.year.unique())

        self._my_arr, self._my_time_index = drop_leap(
            self._my_arr, self._my_time_index
        )

        if time_masks is None:
            self._time_masks = make_time_masks(self._my_time_index)
        else:
            masks = deepcopy(time_masks)
            ti_temp = deepcopy(my_time_index)
            for k, v in masks.items():
                masks[k], _ = drop_leap(v, ti_temp)
            self._time_masks = masks

        self._cdf = self._sort_monthly(self._my_arr)

        self._lt_cdf = self._make_lt_cdf(self._cdf)

        self._cdf_frac, self._lt_frac, self._interp_frac = self._cdf_fracs()

        self._fs_all = self._fs_stat()

    @property
    def time_index(self):
        """Get the multi-year daily non-leap time index.

        Returns
        -------
        time_index : pd.datetimeindex
            Daily datetime index corresponding to the rows in cdf.
        """
        return self._my_time_index

    @property
    def cdf(self):
        """Get the multi-year cdf array with daily values sorted within each
        year and month.

        Returns
        -------
        cdf : np.ndarray
            Multi-year daily timeseries array with sorted values
            resetting each month.
        """
        return self._cdf

    @property
    def lt_cdf(self):
        """Get the multi-year long term sorted array (values are sorted within
        monthly masks across all years).

        Returns
        -------
        lt_cdf : np.ndarray
            Array with the same shape as arr but with monthly values sorted
            across all years.
        """
        return self._lt_cdf

    @property
    def years(self):
        """Get the list of years included in this multi-year cdf.

        Returns
        -------
        years : list
            List of years included in this cdf analysis.
        """
        return self._years

    @property
    def fs_all(self):
        """Get all FS statistics for all months/years/sites.

        Returns
        -------
        fs : dict
            Dictionary with month keys. Each dict value is a (y, n) array where
            y is years and n is sites. Each array entry is the FS metric.
        """
        return self._fs_all

    @staticmethod
    def _cumulative_sum_monthly(arr, time_index):
        """Calculate the montly CDF of the data array.

        Parameters
        ----------
        arr : np.ndarray
            Timeseries array (time, sites) for one variable
            (instantaneous data values).
        time_index : pd.datetimeindex
            Datetime index corresponding to the rows in arr.

        Returns
        -------
        arr : np.ndarray
            Timeseries array with cumulative values resetting each month.
        """
        years = time_index.year.unique()
        for y in years:
            year_mask = time_index.year == y
            for m in range(1, 13):
                mask = year_mask & (time_index.month == m)
                arr[mask] = np.cumsum(arr[mask], axis=0)

        return arr

    def _sort_monthly(self, arr):
        """Sort daily values within each month within each year.

        Parameters
        ----------
        arr : np.ndarray
            Timeseries array (time, sites) for one variable
            (instantaneous or daily values).

        Returns
        -------
        arr : np.ndarray
            Timeseries array with sorted values resetting each month.
        """

        for y in self._years:
            year_mask = self._time_masks[y]
            for m in range(1, 13):
                mask = year_mask & self._time_masks[m]
                arr[mask] = np.sort(arr[mask], axis=0)

        return arr

    @staticmethod
    def _resample_daily_max(arr, time_index):
        """Convert a timeseries array to daily maximum data points.

        Parameters
        ----------
        arr : np.ndarray
            Timeseries array (time, sites) for one variable.
        time_index : pd.datetimeindex
            Datetime index corresponding to the rows in arr.

        Returns
        -------
        arr : np.ndarray
            Daily timeseries array where each value is the maximum value in
            the given day.
        time_index : pd.datetimeindex
            Daily datetime index corresponding to the rows in arr.
        """
        df = pd.DataFrame(arr, index=time_index)
        df = df.resample('1D').max()

        return df.values, df.index

    def _make_lt_cdf(self, arr):
        """Make a long term value sorted array where values within month masks
        are sorted across multi-years.

        Parameters
        ----------
        arr : np.ndarray
            Timeseries array with cumulative values.

        Returns
        -------
        lt_cdf : np.ndarray
            Array with the same shape as arr but with monthly values sorted
            across all years.
        """
        lt_cdf = np.zeros_like(arr)
        for m in range(1, 13):
            mask = self._time_masks[m]
            lt_cdf[mask, :] = np.sort(arr[mask], axis=0)
        return lt_cdf

    def _cdf_fracs(self):
        """Make the fractional arrays for the y-axis of a CDF.

        Returns
        -------
        cdf_frac : np.ndarray
            (t, ) array of the cumulative summation fraction (0 to 1)
            corresponding to the individual month and year for CDFs.
        lt_frac : np.ndarray
            (t, ) array of the cumulative summation fraction (0 to 1)
            corresponding to months over multiple years in the long term cdf.
        interp_frac : np.ndarray
            (t, n) array of the cumulative summation fraction (0 to 1)
            corresponding to the long term multi-year CDF. This is
            the cdf fraction y-projected onto the long term CDF so that a
            broadcasted subtraction can be performed.
        """

        cdf_frac = np.zeros((len(self._cdf), self._cdf.shape[1]))
        lt_frac = np.zeros((len(self._lt_cdf), self._lt_cdf.shape[1]))
        for m in range(1, 13):
            mask = self._time_masks[m]
            lt_frac[mask, :] = np.expand_dims(
                np.linspace(0, 1, mask.sum()), axis=1
            )
            for y in self._years:
                mask = self._time_masks[y] & self._time_masks[m]
                cdf_frac[mask, :] = np.expand_dims(
                    np.linspace(0, 1, mask.sum()), axis=1
                )

        interp_frac = np.zeros(self._cdf.shape)
        for n in range(self._cdf.shape[1]):
            for m in range(1, 13):
                for y in self._years:
                    lt_mask = self._time_masks[m]
                    mask = self._time_masks[y] & self._time_masks[m]

                    interp_frac[mask, n] = np.interp(
                        self._cdf[mask, n],
                        self._lt_cdf[lt_mask, n],
                        lt_frac[lt_mask, n],
                    )

        return cdf_frac, lt_frac, interp_frac

    def _fs_stat(self):
        """Finkelstein-Schafer metric comparing the test cdf to a baseline

        Returns
        -------
        fs : dict
            Dictionary with month keys. Each dict value is a (y, n) array where
            y is years and n is sites. Each array entry is the FS metric.
        """

        diff_cdf = np.abs(self._cdf_frac - self._interp_frac)

        fs_arr = np.zeros((len(self._years), self._cdf.shape[1]))
        fs = {m: deepcopy(fs_arr) for m in range(1, 13)}
        for i, y in enumerate(self._years):
            for m in range(1, 13):
                mask = self._time_masks[y] & self._time_masks[m]
                fs[m][i, :] = diff_cdf[mask, :].mean(axis=0)

        return fs

    def _best_fs_year(self):
        """Select single best TMY year for each month based on the FS statistic

        Returns
        -------
        years : np.ndarray
            Array of best TMY years for every month for every site.
            Shape is (months, sites).
        fs : np.ndarray
            Array of min FS statistic for each month for every site
            corresponding to the year selected. Shape is (months, sites).
        """

        years = None
        fs = None

        for m, ranks in self._fs_all.items():
            if years is None:
                years = np.zeros((12, ranks.shape[1]), dtype=np.uint16)
                fs = np.zeros((12, ranks.shape[1]))

            i = m - 1
            years[i, :] = int(self.years[0]) + np.argmin(ranks, axis=0)
            fs[i, :] = np.min(ranks, axis=0)

        return years, fs

    def plot_tmy_selection(
        self,
        month=1,
        site=0,
        fig_size=(12, 9),
        fout=None,
        xlabel='Cumulative Value',
        ylabel='CDF',
        plot_years=None,
    ):
        """Plot a single site's TMY CDFs for one month.

        Parameters
        ----------
        month : int
            Month number to plot (1-indexed)
        site : int
            Site index to plot (0-indexed)
        fig_size : tuple
            Figure size (height, width)
        fout : str | None
            Filepath to save image to.
        xlabel : str
            Xlabel string (cumulative data value).
        ylabel : str
            Ylabel string (CDF fraction).
        plot_years : list | None
            Optional set of years to plot (makes the plot less busy)
        """

        tmy_years, _ = self._best_fs_year()

        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(111)

        if plot_years is None:
            plot_years = self.years
        for y in plot_years:
            mask = self._time_masks[y] & self._time_masks[month]
            ax.plot(self.cdf[mask, site], self._cdf_frac[mask, site], '--')

        lt_mask = self._time_masks[month]
        ax.plot(self.lt_cdf[lt_mask, site], self._lt_frac[lt_mask, site], 'b-')

        tmy_year = tmy_years[(month - 1), site]
        mask = self._time_masks[tmy_year] & self._time_masks[month]
        ax.plot(self.cdf[mask, site], self._interp_frac[mask, site], 'rx')
        ax.plot(self.cdf[mask, site], self._cdf_frac[mask, site], 'r-x')

        legend = [
            *plot_years,
            'Long Term CDF',
            'Interpolated',
            'Best FS ({})'.format(tmy_year),
        ]
        plt.legend(legend)
        ax.set_title('TMY CDFs for Month {} and Site {}'.format(month, site))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if fout is not None:
            fig.savefig(fout, bbox_inches='tight')
        else:
            plt.show()

        plt.close()
