# -*- coding: utf-8 -*-
"""NSRDB Typical Meteorological Year (TMY) code.

Created on Wed Oct 23 10:55:23 2019

@author: gbuster
"""
from copy import deepcopy
import h5py
import os
import pandas as pd
import numpy as np
import datetime
from nsrdb.file_handlers.resource import Resource


class Cdf:
    """Class for handling cumulative distribution function methods for TMY."""

    def __init__(self, my_arr, my_time_index):
        """
        Parameters
        ----------
        my_arr : np.ndarray
            Multi-year (my) timeseries array (time, sites) for one variable.
        my_time_index : pd.datetimeindex
            Datetime index corresponding to the rows in my_arr
        """

        self._my_arr = my_arr
        self._my_time_index = my_time_index
        self._years = sorted(self._my_time_index.year.unique())

        self._my_cdf = self._cumulative_sum_monthly(
            self._my_arr, self._my_time_index)

        self._my_cdf, self._my_time_index = self._resample_daily_max(
            self._my_cdf, self._my_time_index)

        self._my_cdf, self._my_time_index = Tmy.drop_leap(
            self._my_cdf, self._my_time_index)

        self._time_masks = self._make_time_masks(self._my_time_index)

        self._mean_cdf = self._make_my_mean_cdf(
            self._my_cdf, self._my_time_index, self._time_masks)

        self._lt_frac, self._annual_frac = self._make_cdf_frac()

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
        """Get the multi-year cumulative array.

        Returns
        -------
        my_cdf : np.ndarray
            Multi-year daily timeseries array with cumulative values
            resetting each month.
        """
        return self._my_cdf

    @property
    def mean_cdf(self):
        """Get the multi-year mean cumulative array.

        Returns
        -------
        mean_cdf : np.ndarray
            Timeseries array with the same shape as cdf, but the annual series
            is a multi-year mean for all years in the input time_index. The
            multi-year mean is repeated for the number of years to get the
            same shape as the input CDF.
        """
        return self._mean_cdf

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
    def _make_time_masks(time_index):
        """Make a time index mask lookup dict.

        Parameters
        ----------
        time_index : pd.datetimeindex
            Time index to mask.

        Returns
        -------
        masks : dict
            Lookup of boolean masks keyed by month and year integer.
        """
        masks = {}
        years = time_index.year.unique()
        months = time_index.month.unique()
        for y in years:
            masks[y] = (time_index.year == y)
        for m in months:
            masks[m] = (time_index.month == m)
        return masks

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
            year_mask = (time_index.year == y)
            for m in range(1, 13):
                mask = (year_mask & (time_index.month == m))
                arr[mask] = np.cumsum(arr[mask], axis=0)

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

    @staticmethod
    def _make_my_mean_cdf(cdf, time_index, masks):
        """Make a multi-year (my) mean CDF corresponding to the my CDF input.

        Parameters
        ----------
        cdf : np.ndarray
            Timeseries array with cumulative values.
        time_index : pd.datetimeindex
            Datetime index corresponding to the rows in cdf.
        masks : dict
            Lookup of boolean masks keyed by month and year integer.

        Returns
        -------
        mean_cdf : np.ndarray
            Timeseries array with the same shape as cdf, but the annual series
            is a multi-year mean for all years in the input time_index. The
            multi-year mean is repeated for the number of years to get the
            same shape as the input CDF.
        """
        mean_cdf = None
        years = time_index.year.unique()
        for y in years:
            mask = masks[y]
            if mean_cdf is None:
                shape = (mask.sum(), cdf.shape[1])
                mean_cdf = np.zeros(shape, dtype=np.float64)
            mean_cdf += cdf[mask]
        mean_cdf /= len(years)
        mean_cdf = np.tile(mean_cdf, (len(years), 1))
        return mean_cdf

    def _make_cdf_frac(self):
        """Make the fractional arrays for the y-axis of a CDF.

        Returns
        -------
        long_term_frac : np.ndarray
            (t, n) array of the cumulative summation fraction (0 to 1)
            corresponding to the long term multi-year mean CDF. This is
            basically the annual fraction y-projected onto the multi-year
            mean CDF so that a broadcasted subtraction can be performed.
        annual_frac : np.ndarray
            (t, ) array of the cumulative summation fraction (0 to 1)
            corresponding to the individual year CDFs.
        """

        annual_frac = np.zeros((len(self._my_cdf), self._my_cdf.shape[1]))
        for m in range(1, 13):
            for y in self._years:
                mask = self._time_masks[y] & self._time_masks[m]
                annual_frac[mask, :] = np.expand_dims(
                    np.linspace(0, 1, mask.sum()), axis=1)

        long_term_frac = np.zeros(self._my_cdf.shape)
        for n in range(self._my_cdf.shape[1]):
            for m in range(1, 13):
                for y in self._years:
                    mask = self._time_masks[y] & self._time_masks[m]
                    long_term_frac[mask, n] = np.interp(
                        self._my_cdf[mask, n], self._mean_cdf[mask, n],
                        annual_frac[mask, n])

        return long_term_frac, annual_frac

    def _fs_stat(self):
        """Finkelstein-Schafer metric comparing the test cdf to a baseline

        Returns
        -------
        fs : dict
            Dictionary with month keys. Each dict value is a (y, n) array where
            y is years and n is sites. Each array entry is the FS metric.
        """

        diff_cdf = np.abs(self._lt_frac - self._annual_frac)

        fs_arr = np.zeros((len(self._years), self._my_cdf.shape[1]))
        fs = {m: deepcopy(fs_arr) for m in range(1, 13)}
        for i, y in enumerate(self._years):
            for m in range(1, 13):
                mask = self._time_masks[y] & self._time_masks[m]
                fs[m][i, :] = diff_cdf[mask, :].mean(axis=0)

        return fs

    @staticmethod
    def _best_fs_year(fs_all, year0):
        """Select single best TMY year for each month based on the FS statistic

        Parameters
        ----------
        fs_all : dict
            Dictionary with month keys. Each dict value is a (y, n) array where
            y is years and n is sites. Each array entry is the FS metric.
        year0 : int | str
            Initial year of the TMY.

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

        for m, ranks in fs_all.items():
            if years is None:
                years = np.zeros((12, ranks.shape[1]), dtype=np.uint16)
                fs = np.zeros((12, ranks.shape[1]))

            i = m - 1
            years[i, :] = int(year0) + np.argmin(ranks, axis=0)
            fs[i, :] = np.min(ranks, axis=0)

        return years, fs

    def plot_tmy_selection(self, month=1, site=0, fig_size=(12, 9), fout=None,
                           xlabel='Cumulative Value', ylabel='CDF',
                           plot_years=None):
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

        tmy_years, _ = self._best_fs_year(
            self._fs_all, self._years[0])

        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(111)

        if plot_years is None:
            plot_years = self.years
        for y in plot_years:
            mask = self._time_masks[y] & self._time_masks[month]
            ax.plot(self.cdf[mask, site], self._annual_frac[mask, site], '--')

        my_mask = ((self.time_index.year == self.years[0])
                   & self._time_masks[month])
        ax.plot(self.mean_cdf[my_mask, site], self._annual_frac[my_mask, site],
                'b-o')

        tmy_year = tmy_years[(month - 1), site]
        mask = self._time_masks[tmy_year] & self._time_masks[month]
        ax.plot(self.cdf[mask, site], self._lt_frac[mask, site], 'rx')
        ax.plot(self.cdf[mask, site], self._annual_frac[mask, site], 'r-x')

        legend = plot_years + ['Mean', 'Mean Interp',
                               'Best FS ({})'.format(tmy_year)]
        plt.legend(legend)
        ax.set_title('TMY CDFs for Month {} and Site {}'.format(month, site))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if fout is not None:
            fig.savefig(fout, bbox_inches='tight')
        else:
            plt.show()

        plt.close()


class Tmy:
    """NSRDB Typical Meteorological Year (TMY) calculation framework."""

    def __init__(self, nsrdb_dir, years, weights, site_slice=slice(None)):
        """
        Parameters
        ----------
        nsrdb_dir : str
            Directory containing annual NSRDB files. All .h5 files with year
            strings in their names will be used.
        years : iterable
            Iterable of years to include in the TMY calculation.
        weights : dict
            Lookup of {dset: weight} where dset is a variable h5 dset name
            and weight is a fractional TMY weighting. All weights must
            sum to 1.0
        site_slice : slice
            Sites to consider in this TMY.
        """

        self._dir = nsrdb_dir
        self._years = sorted([int(y) for y in years])
        self._weights = weights
        self._site_slice = site_slice
        self._my_time_index = None
        self._time_index = None
        self._meta = None

        self._fpaths = self._parse_dir(self._dir, self._years)
        self._check_weights(self._fpaths, self._weights)

    @staticmethod
    def _parse_dir(nsrdb_dir, years):
        """Get the included nsrdb file paths from a directory.

        Parameters
        ----------
        nsrdb_dir : str
            Directory containing annual NSRDB files. All .h5 files with year
            strings in their names will be used.
        years : iterable
            Iterable of years to include in the TMY calculation.

        Returns
        -------
        fpaths : list
            List of full filepaths to nsrdb .h5 files to include in the tmy.
        """
        fpaths = [os.path.join(nsrdb_dir, fn) for fn in os.listdir(nsrdb_dir)
                  if any([str(y) in fn for y in years])
                  and fn.endswith('.h5') and 'nsrdb' in fn]
        return fpaths

    @staticmethod
    def _check_weights(fpaths, weights):
        """Check the weights.

        Parameters
        ----------
        fpaths : list
            List of nsrdb .h5 file paths.
        weights : dict
            Lookup of {dset: weight} where dset is a variable h5 dset name
            and weight is a fractional TMY weighting. All weights must
            sum to 1.0
        """
        if sum(list(weights.values())) != 1.0:
            raise ValueError('Weights do not sum to 1.0!')
        for dset in weights.keys():
            for fpath in fpaths:
                with h5py.File(fpath, 'r') as f:
                    if dset not in list(f):
                        e = ('Weight dset "{}" not found in file: "{}"'
                             .format(dset, os.path.basename(fpath)))
                        raise KeyError(e)

    def _get_my_arr(self, dset, unscale=False):
        """Get a multi-year 2D numpy array for a given dataset

        Parameters
        ----------
        dset : str
            Dataset / variable name.
        unscale : bool
            Flag to unscale data from h5 disk storage precision to float.
            Unscaling shouldn't be necessary for TMY calculation.

        Returns
        -------
        arr : np.ndarray
            Multi-year multi-site array of dset data.
        """

        arr = None
        if unscale:
            dtype = np.float32
        else:
            dtype = np.int32

        for fpath in self._fpaths:
            with Resource(fpath, unscale=unscale) as res:
                ti = res.time_index
                temp = res[dset, :, self._site_slice]

                if arr is None:
                    shape = (len(self.my_time_index), temp.shape[1])
                    arr = np.zeros(shape, dtype=dtype)

                iloc = self.my_time_index.isin(ti)
                arr[iloc, :] = temp

        return arr

    @property
    def my_time_index(self):
        """Full multi-year time index.

        Returns
        -------
        my_time_index : pd.Datetimeindex
            Multi-year datetime index corresponding to multi-year data arrays.
        """
        if self._my_time_index is None:
            start = '1-1-{}'.format(self.years[0])
            end = '1-1-{}'.format(self.years[-1] + 1)
            self._my_time_index = pd.date_range(start=start, end=end,
                                                freq=self.source_freq,
                                                closed='left')
        return self._my_time_index

    @property
    def time_index(self):
        """Time index for first TMY year without leap day.

        Returns
        -------
        time_index : pd.Datetimeindex
            Single-year datetime index corresponding to TMY output.
        """
        if self._time_index is None:
            start = '1-1-{}'.format(self.years[0])
            end = '1-1-{}'.format(self.years[0] + 1)
            self._time_index = pd.date_range(start=start, end=end,
                                             freq='1h', closed='left')
            self._time_index += datetime.timedelta(minutes=30)
            if len(self._time_index) != 8760:
                _, self._time_index = self.drop_leap(
                    np.zeros((len(self._time_index), 1)), self._time_index)
        return self._time_index

    @property
    def meta(self):
        """Get the meta data from the current site slice.

        Returns
        -------
        meta : pd.DataFrame
            Meta data from the current site slice.
        """
        if self._meta is None:
            with Resource(self._fpaths[0]) as res:
                meta = res.meta
            self._meta = meta.iloc[self._site_slice, :]
        return self._meta

    @staticmethod
    def drop_leap(arr, time_index):
        """Make 365-day timeseries (TMY does not have leap days).

        Parameters
        ----------
        arr : np.ndarray
            Timeseries array (time, sites) for one variable.
        time_index : pd.datetimeindex
            Datetime index corresponding to the rows in arr. May have leap days

        Returns
        -------
        arr : np.ndarray
            Timeseries array (time, sites) for one variable, n_rows is a
            multiple of 8760.
        time_index : pd.datetimeindex
            Datetime index corresponding to the rows in arr, without leap days.
        """
        leap_day = (time_index.month == 2) & (time_index.day == 29)
        if any(leap_day):
            arr = arr[~leap_day]
            time_index = time_index[~leap_day]
        return arr, time_index

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
    def source_freq(self):
        """Get the nsrdb source temporal frequency.

        Returns
        -------
        freq : str
            Pandas datetimeindex frequency string ('30min', '1h').
        """
        with h5py.File(self._fpaths[0], 'r') as f:
            ti_len = f['time_index'].shape[0]

        if (ti_len == 17520) | (ti_len == 17568):
            freq = '30min'
        elif (ti_len == 8760) | (ti_len == 8784):
            freq = '1h'
        else:
            raise ValueError('Could not parse source temporal frequency '
                             'from time index length {}'.format(ti_len))
        return freq

    @staticmethod
    def select_fs_years(fs_all, year0, n=5):
        """Select 5 best TMY years for each month based on the FS statistic

        Parameters
        ----------
        fs_all : dict
            Dictionary with month keys. Each dict value is a (y, n) array where
            y is years and n is sites. Each array entry is the FS metric.
        year0 : int | str
            Initial year of the TMY.
        n : int
            Number of years to select.

        Returns
        -------
        years : dict
            Month-keyed dictionary of arrays of best TMY years for every month
            for every site. Shape is (n, sites).
        fs : dict
            Month-keyed dictionary of arrays of min FS statistic for each month
            for every site corresponding to the 5 years selected.
            Shape is (n, sites).
        """

        years = None
        fs = None

        for m, ranks in fs_all.items():
            if years is None:
                years = {m: np.zeros((n, ranks.shape[1]), dtype=np.uint16)
                         for m in range(1, 13)}
                fs = {m: np.zeros((n, ranks.shape[1])) for m in range(1, 13)}

            years[m] = int(year0) + np.argsort(ranks, axis=0)[:n, :]
            fs[m] = np.sort(ranks, axis=0)[:n, :]

        return years, fs

    def sort_years_mm(self, tmy_years_5):
        """Sort candidate TMY months/years based on deviation from the
        multi-year mean and median GHI.

        Parameters
        ----------
        tmy_years_5 : dict
            Month-keyed dictionary of arrays of best 5 TMY years for every
            month for every site. Shape is (5, sites).

        Returns
        -------
        tmy_years_5 : dict
            Month-keyed dictionary of arrays of best 5 TMY years for every
            month for every site SORTED BY deviation from the multi-year
            monthly mean and median GHI values.
        """

        ghi = self._get_my_arr('ghi')
        lt_mean = np.zeros((12, ghi.shape[1]), dtype=np.float32)
        lt_median = np.zeros((12, ghi.shape[1]), dtype=np.float32)

        for m in range(1, 13):
            mask = (self.my_time_index.month == m)
            lt_mean[(m - 1), :] = np.mean(ghi[mask, :], axis=0)
            lt_median[(m - 1), :] = np.median(ghi[mask, :], axis=0)

        diffs = {m: np.zeros((len(self.years), ghi.shape[1]), dtype=np.float32)
                 for m in range(1, 13)}
        sorted_years = {m: np.zeros((len(self.years), ghi.shape[1]),
                                    dtype=np.float32)
                        for m in range(1, 13)}

        for i, y in enumerate(self.years):
            for m in range(1, 13):
                mask = ((self.my_time_index.month == m)
                        & (self.my_time_index.year == y))
                this_mean = np.mean(ghi[mask, :], axis=0)
                this_median = np.median(ghi[mask, :], axis=0)

                im = m - 1
                diffs[m][i, :] = (np.abs(lt_mean[im, :] - this_mean)
                                  + np.abs(lt_median[im, :] - this_median))

        for m in range(1, 13):
            sorted_years[m] = np.argsort(diffs[m], axis=0) + self.years[0]

        for site in range(ghi.shape[1]):
            for m in range(1, 13):
                temp = [y for y in sorted_years[m][:, site]
                        if y in tmy_years_5[m][:, site]]
                tmy_years_5[m][:, site] = temp

        return tmy_years_5

    def calculate_tmy(self):
        """Calculate the TMY based on the multiple-dataset weights.

        Returns
        -------
        tmy_years : np.ndarray
            Array of best TMY years for every month for every site.
            Shape is (months, sites).
        """

        ws = {}
        for dset, weight in self._weights.items():
            cdf = Cdf(self._get_my_arr(dset), self.my_time_index)
            if not ws:
                for m, fs in cdf.fs_all.items():
                    ws[m] = weight * fs
            else:
                for m, fs in cdf.fs_all.items():
                    ws[m] += weight * fs

        tmy_years_5, _ = self.select_fs_years(ws, self.years[0], n=5)
        tmy_years_5 = self.sort_years_mm(tmy_years_5)

        tmy_years = np.zeros((12, tmy_years_5[1].shape[1]), dtype=np.uint16)
        for m in range(1, 13):
            tmy_years[(m - 1), :] = tmy_years_5[m][0, :]

        return tmy_years

    def _make_tmy_timeseries(self, dset, tmy_years, unscale=False):
        """Make the TMY 8760 timeseries from the selected TMY years.

        Parameters
        ----------
        dset : str
            Dataset name to make timeseries for.
        tmy_years : np.ndarray
            Array of best TMY years for every month for every site.
            Shape is (months, sites).
        unscale : bool
            Flag to unscale data from dset storage dtype to physical dtype.

        Returns
        -------
        data : np.ndarray
            8760 x n_sites timeseries array of dset with data in each month
            taken from the selected tmy_year.
        """

        year_set = sorted(list(set(list(tmy_years.flatten()))))
        data = None
        masks = None

        for year in year_set:
            fpath = [f for f in self._fpaths if str(year) in f][0]
            with Resource(fpath, unscale=unscale) as res:
                ti = res.time_index
                temp = res[dset, :, self._site_slice]
                temp, ti = self.drop_leap(temp, ti)

            if masks is None:
                masks = {m: (ti.month == m) for m in range(1, 13)}
            if data is None:
                data = np.zeros(temp.shape, dtype=temp.dtype)

            mask = (tmy_years == year)
            locs = np.where(mask)
            months = locs[0] + 1
            sites = locs[1]

            for month, site in zip(months, sites):
                data[masks[month], site] = temp[masks[month], site]

        if len(data) > 8760:
            data = data[1::2, :]
        if len(data) != 8760:
            raise ValueError('TMY timeseries was not evaluated as an 8760! '
                             'Instead had final length {}.'.format(len(data)))

        return data
