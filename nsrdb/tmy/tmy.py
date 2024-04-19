# -*- coding: utf-8 -*-
"""NSRDB Typical Meteorological Year (TMY) code.

Created on Wed Oct 23 10:55:23 2019

@author: gbuster
"""
import datetime
import json
import logging
import os
import shutil
from concurrent.futures import as_completed
from copy import deepcopy
from inspect import signature
from itertools import groupby

import h5py
import numpy as np
import pandas as pd
from cloud_fs import FileSystem as FS
from rex import init_logger
from rex.utilities.execution import SpawnProcessPool
from rex.utilities.hpc import SLURM

from nsrdb.data_model.variable_factory import VarFactory
from nsrdb.file_handlers.outputs import Outputs
from nsrdb.file_handlers.resource import MultiFileResource, Resource
from nsrdb.utilities.file_utils import pd_date_range

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

        self._my_arr, self._my_time_index = Tmy.drop_leap(self._my_arr,
                                                          self._my_time_index)

        if time_masks is None:
            self._time_masks = Tmy._make_time_masks(self._my_time_index)
        else:
            masks = deepcopy(time_masks)
            ti_temp = deepcopy(my_time_index)
            for k, v in masks.items():
                masks[k], _ = Tmy.drop_leap(v, ti_temp)
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
            year_mask = (time_index.year == y)
            for m in range(1, 13):
                mask = (year_mask & (time_index.month == m))
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
                mask = (year_mask & self._time_masks[m])
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
                np.linspace(0, 1, mask.sum()), axis=1)
            for y in self._years:
                mask = self._time_masks[y] & self._time_masks[m]
                cdf_frac[mask, :] = np.expand_dims(
                    np.linspace(0, 1, mask.sum()), axis=1)

        interp_frac = np.zeros(self._cdf.shape)
        for n in range(self._cdf.shape[1]):
            for m in range(1, 13):
                for y in self._years:
                    lt_mask = self._time_masks[m]
                    mask = self._time_masks[y] & self._time_masks[m]

                    interp_frac[mask, n] = np.interp(self._cdf[mask, n],
                                                     self._lt_cdf[lt_mask, n],
                                                     lt_frac[lt_mask, n])

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

        legend = [*plot_years, 'Long Term CDF', 'Interpolated', 'Best FS ({})'
                  .format(tmy_year)]
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

    def __init__(self, nsrdb_base_fp, years, weights, site_slice=None,
                 supplemental_fp=None):
        """
        Parameters
        ----------
        nsrdb_base_fp : str
            Base nsrdb filepath to retrieve annual files from. Must include
            a single {} format option for the year. Can include * for an
            NSRDB multi file source.
        years : iterable
            Iterable of years to include in the TMY calculation.
        weights : dict
            Lookup of {dset: weight} where dset is a variable h5 dset name
            and weight is a fractional TMY weighting. All weights must
            sum to 1.0
        site_slice : slice
            Sites to consider in this TMY.
        supplemental_fp : None | dict
            Supplemental data base filepaths including {} for year for
            uncommon dataset inputs to the TMY calculation. For example:
            {'poa': '/projects/pxs/poa_h5_dir/poa_out_{}.h5'}
        """

        logger.debug('Initializing TMY algorithm for sites {}...'
                     .format(site_slice))

        self._nsrdb_base_fp = nsrdb_base_fp
        self._years = sorted([int(y) for y in years])
        self._weights = weights
        self._site_slice = slice(None)
        if site_slice is not None:
            self._site_slice = site_slice
        self._my_time_index = None
        self._time_index = None
        self._my_daily_time_index = None
        self._meta = None
        self._d_total_ghi = None
        self._d_mean_temp = None
        self._time_masks = None
        self._daily_time_masks = None

        self._raw_data_cache = {}

        self._tmy_years_short = None
        self._tmy_years_long = None

        self._supplemental = {} if supplemental_fp is None else supplemental_fp

        self._check_weights()
        self._init_daily_ghi_temp()

    def _check_weights(self):
        """Check that the weights are valid."""

        if np.abs(sum(list(self._weights.values())) - 1) > 1e5:
            raise ValueError('Weights do not sum to 1.0!')

        for year in self._years:
            for dset in self._weights.keys():
                fpath, Handler = self._get_fpath(dset, year)
                dset = self._strip_dset_fun(dset)[0]
                with Handler(fpath) as f:
                    if dset not in f.dsets:
                        e = ('Weight dset "{}" not found in file: "{}"'
                             .format(dset, os.path.basename(fpath)))
                        raise KeyError(e)

    @staticmethod
    def _strip_dset_fun(dset):
        """Retrieve a resampling method from dset name prefix.

        Parameters
        ----------
        dset : str
            Dataset / variable name with optional min_ max_ mean_ sum_ prefix.

        Returns
        -------
        dset : str
            Dataset / variable name with the optional function prefix
            stripped off.
        fun : str | NoneType
            Function string from the prefix of dset
        """
        fun = None
        if dset.startswith('min_'):
            fun = 'min_'
        elif dset.startswith('max_'):
            fun = 'max_'
        elif dset.startswith('mean_'):
            fun = 'mean_'
        elif dset.startswith('sum_'):
            fun = 'sum_'
        if fun is not None:
            dset = dset.replace(fun, '')
        return dset, fun

    @staticmethod
    def _resample_arr_daily(arr, time_index, fun):
        """Resample an array to daily using a specified function.

        Parameters
        ----------
        arr : np.ndarray
            Array of timeseries data corresponding to time_index.
        time_index : pd.Datetimeindex
            Datetimeindex corresponding to arr.
        fun : str | None
            Resampling method.

        Returns
        -------
        arr : np.ndarray
            Array of daily timeseries data if fun is not None.
        """
        if fun is not None:
            df = pd.DataFrame(deepcopy(arr), index=time_index)
            if 'min' in fun.lower():
                df = df.resample('1D').min()
            elif 'max' in fun.lower():
                df = df.resample('1D').max()
            elif 'mean' in fun.lower():
                df = df.resample('1D').mean()
            elif 'sum' in fun.lower():
                df = df.resample('1D').sum()
            arr = df.values
        return arr

    def _get_fpath(self, dset, year):
        """Get a list of filepaths for a dataset by first checking
        the supplemental data sources and then the default source.

        Parameters
        ----------
        dset : str
            Dataset / variable name with optional min_ max_ mean_ sum_ prefix
        year : int
            Year of interest

        Returns
        -------
        fpaths : list
            List of filepaths for dset (considering supplemental data sources)
        Handler : Resource | MultiFileResource
            Resource handler object to use to open fpaths
        """

        fpath = self._nsrdb_base_fp
        dset = self._strip_dset_fun(dset)[0]
        if dset in self._supplemental:
            fpath = self._supplemental[dset]

        Handler = Resource
        if '*' in fpath:
            Handler = MultiFileResource

        return fpath.format(year), Handler

    def _get_my_arr_raw(self, dset, unscale=True):
        """Get a multi-year 2D numpy array for a given dataset at source
        temporal resolution.

        Parameters
        ----------
        dset : str
            Dataset / variable name with optional min_ max_ mean_ sum_ prefix.
        unscale : bool
            Flag to unscale data from h5 disk storage precision to float.

        Returns
        -------
        arr : np.ndarray
            Multi-year multi-site array of dset data at source
            temporal resolution.
        """
        arr = None

        if unscale:
            dtype = np.float32
        else:
            dtype = np.int32

        for year in self._years:
            fpath, Handler = self._get_fpath(dset, year)
            with FS(fpath) as f:
                with Handler(f, unscale=unscale) as res:
                    temp = res[dset, :, self._site_slice]

            if arr is None:
                shape = (len(self.my_time_index), temp.shape[1])
                arr = np.zeros(shape, dtype=dtype)

            mask = self.time_masks[year]

            if len(temp) < mask.sum():
                ind = int(mask.sum() % 8760)
                temp2 = temp[:ind, :].copy()
                temp = np.vstack((temp, temp2))

            elif len(temp) != mask.sum():
                with FS(fpath) as f:
                    with Resource(f, unscale=False) as res:
                        ti = res.time_index

                temp = self.drop_leap(temp, ti)[0]

            arr[mask, :] = temp

        return arr

    def _get_my_arr(self, dset, unscale=True):
        """Get a multi-year 2D numpy array for a given dataset possibly
        resampled to daily values.

        Parameters
        ----------
        dset : str
            Dataset / variable name with optional min_ max_ mean_ sum_ prefix.
        unscale : bool
            Flag to unscale data from h5 disk storage precision to float.

        Returns
        -------
        arr : np.ndarray
            Multi-year multi-site array of dset data (possibly resampled
            daily values).
        """

        if dset == 'sum_ghi' and self._d_total_ghi is not None:
            return self.daily_total_ghi
        elif dset == 'mean_air_temperature' and self._d_mean_temp is not None:
            return self.daily_mean_temp

        dset, fun = self._strip_dset_fun(dset)

        if dset in self._raw_data_cache:
            arr = self._raw_data_cache[dset]
        else:
            arr = self._get_my_arr_raw(dset, unscale=unscale)
            self._raw_data_cache[dset] = arr

        arr = self._resample_arr_daily(arr, self.my_time_index, fun)

        return arr

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

    def _init_daily_ghi_temp(self):
        """Initialize daily total ghi and daily mean air temperature
        (used multiple times)"""
        self._d_total_ghi = self._get_my_arr('sum_ghi')
        self._d_mean_temp = self._get_my_arr('mean_air_temperature')

    @property
    def daily_total_ghi(self):
        """Daily GHI multi-year timeseries.

        Returns
        -------
        _d_total_ghi  : np.ndarray
            Multi-year timeseries of daily total GHI.
        """

        if self._d_total_ghi is None:
            self._d_total_ghi = self._get_my_arr('sum_ghi')
        return self._d_total_ghi

    @property
    def daily_mean_temp(self):
        """Daily mean temperature multi-year timeseries.

        Returns
        -------
        _d_mean_temp  : np.ndarray
            Multi-year timeseries of daily mean temperature.
        """

        if self._d_mean_temp is None:
            self._d_mean_temp = self._get_my_arr('mean_air_temperature')
        return self._d_mean_temp

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
            self._my_time_index = pd_date_range(start=start, end=end,
                                                freq=self.source_freq,
                                                closed='left')
        return self._my_time_index

    @property
    def my_daily_time_index(self):
        """Full multi-year time index.

        Returns
        -------
        my_time_index : pd.Datetimeindex
            Multi-year datetime index corresponding to multi-year data arrays.
        """
        if self._my_daily_time_index is None:
            df = pd.DataFrame(np.arange(len(self.my_time_index)),
                              index=self.my_time_index)
            df = df.resample('1D').sum()
            self._my_daily_time_index = df.index
        return self._my_daily_time_index

    @property
    def time_index(self):
        """Time index for last TMY year without leap day.

        Returns
        -------
        time_index : pd.Datetimeindex
            Single-year datetime index corresponding to TMY output.
        """
        if self._time_index is None:
            start = '1-1-{}'.format(self.years[-1])
            end = '1-1-{}'.format(self.years[-1] + 1)
            self._time_index = pd_date_range(start=start, end=end,
                                             freq='1h', closed='left')
            self._time_index += datetime.timedelta(minutes=30)
            if len(self._time_index) != 8760:
                _, self._time_index = self.drop_leap(
                    np.zeros((len(self._time_index), 1)), self._time_index)
        return self._time_index

    @property
    def tmy_years_short(self):
        """Get a short montly array of selected TMY years.

        Returns
        -------
        tmy_years_short : np.ndarray
            Array of best TMY years for every month for every site.
            Shape is (months, sites).
        """
        return self._tmy_years_short

    @property
    def tmy_years_long(self):
        """Get a long 8760 array of selected TMY years.

        Returns
        -------
        tmy_years_long : np.ndarray
            Array of best TMY years for every month for every site.
            Shape is (8760, sites).
        """
        return self._tmy_years_long

    @property
    def time_masks(self):
        """Get a time index mask lookup dict.

        Returns
        -------
        masks : dict
            Lookup of boolean masks keyed by month and year integer.
        """
        if self._time_masks is None:
            self._time_masks = self._make_time_masks(self.my_time_index)
        return self._time_masks

    @property
    def daily_time_masks(self):
        """Get a daily time index mask lookup dict.

        Returns
        -------
        masks : dict
            Lookup of boolean masks for a daily timeseries keyed by month and
            year integer.
        """
        if self._daily_time_masks is None:
            self._daily_time_masks = self._make_time_masks(
                self.my_daily_time_index)
        return self._daily_time_masks

    @property
    def meta(self):
        """Get the meta data from the current site slice.

        Returns
        -------
        meta : pd.DataFrame
            Meta data from the current site slice.
        """
        if self._meta is None:
            fpath, Handler = self._get_fpath('ghi', self._years[0])
            with FS(fpath) as f:
                with Handler(f) as res:
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
        if len(arr) % 8760 != 0:
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
        fpath, Handler = self._get_fpath('ghi', self._years[0])
        with Handler(fpath) as f:
            ti_len = len(f.time_index)

        if ti_len % 8760 == 0:
            freq = f'{60 * 8760 // ti_len}min'
        elif ti_len % 8784 == 0:
            freq = f'{60 * 8784 // ti_len}min'
        else:
            raise ValueError('Could not parse source temporal frequency '
                             'from time index length {}'.format(ti_len))
        if freq == '60min':
            freq = '1h'

        return freq

    def get_weighted_fs(self):
        """Get the FS metric for all datasets and weight and combine.

        This is part of STEP #1 of the NSRDB TMY.

        Returns
        -------
        ws : dict
            Dictionary with month keys. Each dict value is a (y, n) array where
            y is years and n is sites. Each array entry is the summed and
            weighted FS metric for all datasets in self._weights.
        """
        ws = {}
        for dset, weight in self._weights.items():
            arr = self._get_my_arr(dset)
            ti = self.my_daily_time_index
            if len(arr) != len(ti):
                raise ValueError('Bad array length of {} when daily ti is {}'
                                 .format(len(arr), len(ti)))
            cdf = Cdf(arr, ti, time_masks=self.daily_time_masks)
            if not ws:
                for m, fs in cdf.fs_all.items():
                    ws[m] = weight * fs
            else:
                for m, fs in cdf.fs_all.items():
                    ws[m] += weight * fs
        return ws

    def select_fs_years(self, fs_all, n=5):
        """Select 5 best TMY years for each month based on the FS statistic

        This is part of STEP #1 of the NSRDB TMY.

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

        years = {m: np.zeros((n, len(self.meta)), dtype=np.uint16)
                 for m in range(1, 13)}
        fs = {m: np.zeros((n, len(self.meta))) for m in range(1, 13)}

        for m, ranks in fs_all.items():
            years[m] = int(self.years[0]) + np.argsort(ranks, axis=0)[:n, :]
            fs[m] = np.sort(ranks, axis=0)[:n, :]

        return years, fs

    def _get_lt_mean_ghi(self):
        """Get the monthly long term (multi-year) mean daily total GHI.

        This is STEP #4 of the NSRDB TMY.

        Returns
        -------
        lt_mean : np.ndarray
            Array of monthly mean daily total GHI values.
            Shape is (12, n_sites). Each value is the mean for that
            month across all TMY years.
        """
        lt_mean = np.zeros((12, len(self.meta)), dtype=np.float32)
        for m in range(1, 13):
            mask = self.daily_time_masks[m]
            lt_mean[(m - 1), :] = np.mean(self.daily_total_ghi[mask, :],
                                          axis=0)
        return lt_mean

    def _get_lt_median_ghi(self):
        """Get the monthly long term (multi-year) median daily total GHI.

        Returns
        -------
        lt_median : np.ndarray
            Array of monthly median daily total GHI values.
            Shape is (12, n_sites). Each value is the median for that
            month across all TMY years.
        """
        lt_median = np.zeros((12, len(self.meta)), dtype=np.float32)
        for m in range(1, 13):
            mask = self.daily_time_masks[m]
            lt_median[(m - 1), :] = np.median(self.daily_total_ghi[mask, :],
                                              axis=0)
        return lt_median

    def _lt_mm_diffs(self):
        """Calculate the difference from the long term mean and median GHI.

        Returns
        -------
        diffs : dict
            Month-keyed dictionary of arrays of shape (n_tmy_years, n_sites)
            differences for every year
        """

        lt_mean = self._get_lt_mean_ghi()
        lt_median = self._get_lt_median_ghi()

        shape = (len(self.years), len(self.meta))
        diffs = {m: np.zeros(shape, dtype=np.float32)
                 for m in range(1, 13)}

        for i, y in enumerate(self.years):
            for m in range(1, 13):
                mask = (self.daily_time_masks[m] & self.daily_time_masks[y])
                this_mean = np.mean(self.daily_total_ghi[mask, :], axis=0)
                this_median = np.median(self.daily_total_ghi[mask, :], axis=0)

                im = m - 1
                diffs[m][i, :] = (np.abs(lt_mean[im, :] - this_mean)
                                  + np.abs(lt_median[im, :] - this_median))
        return diffs

    def sort_years_mm(self, tmy_years_5):
        """Sort candidate TMY months/years based on deviation from the
        multi-year mean and median GHI.

        This is STEP #2 of the NSRDB TMY.

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
        diffs : dict
            Month-keyed dictionary of arrays of shape (n_tmy_years, n_sites)
            differences for every year
        """

        shape = (len(self.years), len(self.meta))
        sorted_years = {m: np.zeros(shape, dtype=np.float32)
                        for m in range(1, 13)}
        diffs = self._lt_mm_diffs()

        for m in range(1, 13):
            sorted_years[m] = np.argsort(diffs[m], axis=0) + self.years[0]

        for site in range(len(self.meta)):
            for m in range(1, 13):
                temp = [y for y in sorted_years[m][:, site]
                        if y in tmy_years_5[m][:, site]]
                tmy_years_5[m][:, site] = temp
        return tmy_years_5, diffs

    @staticmethod
    def _count_runs(arr):
        """Count the run length and number in a boolean array.

        Parameters
        ----------
        arr : np.ndarray
            1D array of boolean values.

        Returns
        -------
        max_run_len : int
            Maximum length of a consecutive True run in arr.
        n_runs : int
            Number of consecutive True runs in arr.
        """

        max_run_len = 0
        n_runs = 0

        for k, g in groupby(arr):
            if k:
                max_run_len = np.max((max_run_len, len(list(g))))
                n_runs += 1

        return max_run_len, n_runs

    def persistence_filter(self, tmy_years_5):
        """Use persistence of extreme mean temp and daily ghi to filter tmy.

        This is STEP #3 of the NSRDB TMY.

        Parameters
        ----------
        tmy_years_5 : dict
            Month-keyed dictionary of arrays of sorted best 5 TMY years for
            every month for every site. Shape is (5, sites).

        Returns
        -------
        tmy_years : np.ndarray
            Array of best TMY years for every month for every site.
            Shape is (months, sites).
        max_run_len : dict
            Nested dictionary (max_run_len[month][site][tmy_year_index])
            of maximum length of a consecutive run.
        n_runs : dict
            Nested dictionary (max_run_len[month][site][tmy_year_index])
            Number of consecutive runs.
        """
        tmy_years = np.zeros((12, len(self.meta)), dtype=np.uint16)
        max_run_len = {}
        n_runs = {}
        for m in range(1, 13):
            max_run_len[m] = {}
            n_runs[m] = {}
            m_mask = self.daily_time_masks[m]
            t33 = np.percentile(self.daily_mean_temp[m_mask, :], 33, axis=0)
            t67 = np.percentile(self.daily_mean_temp[m_mask, :], 67, axis=0)
            g33 = np.percentile(self.daily_total_ghi[m_mask, :], 33, axis=0)

            t_low = (self.daily_mean_temp < t33)
            t_high = (self.daily_mean_temp > t67)
            g_low = (self.daily_total_ghi < g33)

            for j in range(len(self.meta)):
                max_run_len[m][j] = [0] * tmy_years_5[m].shape[0]
                n_runs[m][j] = [0] * tmy_years_5[m].shape[0]

                for i, y in enumerate(tmy_years_5[m][:, j]):
                    y_mask = self.daily_time_masks[y]
                    mask = y_mask & m_mask

                    for arr in [t_low, t_high, g_low]:
                        m_temp, n_temp = self._count_runs(arr[mask, j])
                        max_run_len[m][j][i] = np.max((max_run_len[m][j][i],
                                                       m_temp))
                        n_runs[m][j][i] += n_temp

                tmy_years[(m - 1), j] = tmy_years_5[m][0, j]
                for i, y in enumerate(tmy_years_5[m][:, j]):
                    screen_out = ((max(max_run_len[m][j])
                                   == max_run_len[m][j][i])
                                  | (max(n_runs[m][j]) == n_runs[m][j][i])
                                  | (n_runs == 0))
                    if not screen_out:
                        tmy_years[(m - 1), j] = y
                        break

        return tmy_years, max_run_len, n_runs

    def _make_tmy_timeseries(self, dset, tmy_years, unscale=True):
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
            As of 2018, the NSRDB has inconsistent scale factors so unscaling
            to physical units is onorous but necessary.

        Returns
        -------
        data : np.ndarray
            (8760 x n_sites) timeseries array of dset with data in each month
            taken from the selected tmy_year.
        """
        year_set = sorted(list(set(list(tmy_years.flatten()))))
        data = None
        masks = None
        self._tmy_years_long = None

        for year in year_set:
            fpath, Handler = self._get_fpath(dset, year)
            with FS(fpath) as f:
                with Handler(f, unscale=unscale) as res:
                    ti = res.time_index
                    temp = res[dset, :, self._site_slice]
                    temp, ti = self.drop_leap(temp, ti)

            if masks is None:
                masks = {m: (ti.month == m) for m in range(1, 13)}
            if data is None:
                data = np.zeros(temp.shape, dtype=temp.dtype)
            if self._tmy_years_long is None:
                self._tmy_years_long = np.zeros(temp.shape, dtype=np.uint16)

            mask = (tmy_years == year)
            locs = np.where(mask)
            months = locs[0] + 1
            sites = locs[1]

            for month, site in zip(months, sites):
                data[masks[month], site] = temp[masks[month], site]
                self._tmy_years_long[masks[month], site] = \
                    tmy_years[(month - 1), site]

        step = len(data) // 8760
        msg = ('Original TMY timeseries is not divisible by 8760 '
               f'length = {len(data)}.')
        assert len(data) % 8760 == 0, msg

        data = data[1::step, :]
        self._tmy_years_long = self._tmy_years_long[1::step, :]

        return data

    def calculate_tmy_years(self):
        """Calculate the TMY based on the multiple-dataset weights.

        Returns
        -------
        tmy_years : np.ndarray
            Array of best TMY years for every month for every site.
            Shape is (months, sites).
        """

        ws = self.get_weighted_fs()
        tmy_years_5, _ = self.select_fs_years(ws)
        tmy_years_5, _ = self.sort_years_mm(tmy_years_5)
        tmy_years, _, _ = self.persistence_filter(tmy_years_5)
        self._tmy_years_short = tmy_years
        return tmy_years

    def get_tmy_timeseries(self, dset, unscale=True):
        """Get a complete TMY timeseries of data for dset.

        Parameters
        ----------
        dset : str
            Name of the dataset to get data for.
        unscale : bool
            Flag to unscale data from dset storage dtype to physical dtype.
            As of 2018, the NSRDB has inconsistent scale factors so unscaling
            to physical units is onorous but necessary.

        Returns
        -------
        data : np.ndarray
            (8760 x n_sites) timeseries array of dset with data in each month
            taken from the selected tmy_year.
        """
        if dset == 'tmy_year_short':
            return self.tmy_years_short
        elif dset in ('tmy_year_long', 'tmy_year'):
            return self.tmy_years_long
        else:
            tmy_years = self.calculate_tmy_years()
            data = self._make_tmy_timeseries(dset, tmy_years, unscale=unscale)
            return data


class TmyRunner:
    """Class to handle running TMY, collecting outs, and writing to files."""

    def __init__(self, nsrdb_base_fp, years, weights, sites_per_worker=100,
                 n_nodes=1, node_index=0, site_slice=None,
                 out_dir='/tmp/scratch/tmy/', fn_out='tmy.h5',
                 supplemental_fp=None, var_meta=None):
        """
        Parameters
        ----------
        nsrdb_base_fp : str
            Base nsrdb filepath to retrieve annual files from. Must include
            a single {} format option for the year. Can include * for an
            NSRDB multi file source.
        years : iterable
            Iterable of years to include in the TMY calculation.
        weights : dict
            Lookup of {dset: weight} where dset is a variable h5 dset name
            and weight is a fractional TMY weighting. All weights must
            sum to 1.0
        sites_per_worker : int
            Number of sites to run at once (sites per core/worker).
        n_nodes : int
            Number of nodes being run.
        node_index : int
            Index of this node job.
        site_slice : slice
            Sites to consider in the GLOBAL TMY run. If multiple jobs are being
            run, the site slice should be the same for all jobs, and slices the
            full spatial extent meta data.
        out_dir : str
            Directory to dump temporary output files.
        fn_out : str
            Final output filename.
        supplemental_fp : None | dict
            Supplemental data base filepaths including {} for year for
            uncommon dataset inputs to the TMY calculation. For example:
            {'poa': '/projects/pxs/poa_h5_dir/poa_out_{}.h5'}
        var_meta : str
            CSV filepath containing meta data for all NSRDB variables.
            Defaults to the NSRDB var meta csv in git repo.
        """

        logger.info('Initializing TMY runner for years: {}'.format(years))
        logger.info('TMY weights: {}'.format(weights))

        self._nsrdb_base_fp = nsrdb_base_fp
        self._years = years
        self._weights = weights

        self._sites_per_worker = sites_per_worker
        self._n_nodes = n_nodes
        self._node_index = node_index
        self._site_chunks = None
        self._site_chunks_index = None

        self._site_slice = slice(None)
        if site_slice is not None:
            self._site_slice = site_slice
        self._meta = None
        self._dsets = None

        self._out_dir = out_dir
        self._fn_out = fn_out
        self._final_fpath = os.path.join(self._out_dir, self._fn_out)

        self._supplemental_fp = supplemental_fp
        self._var_meta = var_meta

        if not os.path.exists(self._out_dir):
            os.makedirs(self._out_dir)

        self._tmy_obj = Tmy(self._nsrdb_base_fp, self._years, self._weights,
                            site_slice=slice(0, 1),
                            supplemental_fp=supplemental_fp)

        out = self._setup_job_chunks(self.meta, self._sites_per_worker,
                                     self._n_nodes, self._node_index,
                                     self._out_dir)
        self._site_chunks, self._site_chunks_index, self._f_out_chunks = out

        logger.info('Node meta data is: \n{}'.format(self.meta))
        logger.info('Node index {} with n_nodes {} running site chunks: '
                    '{} ... {}'
                    .format(node_index, n_nodes,
                            str(self._site_chunks)[:100],
                            str(self._site_chunks)[-100:]))
        logger.info('Node index {} with n_nodes {} running site chunks '
                    'indices: {} ... {}'
                    .format(node_index, n_nodes,
                            str(self._site_chunks_index)[:100],
                            str(self._site_chunks_index)[-100:]))
        logger.info('Node index {} with n_nodes {} running fout chunks: '
                    '{} ... {}'
                    .format(node_index, n_nodes,
                            str(self._f_out_chunks)[:100],
                            str(self._f_out_chunks)[-100:]))

    @staticmethod
    def _setup_job_chunks(meta, sites_per_worker, n_nodes, node_index,
                          out_dir):
        """Setup chunks and file names for a multi-chunk multi-node job.

        Parameters
        ----------
        meta : pd.DataFrame
            FULL NSRDB meta data.
        sites_per_worker : int
            Number of sites to run at once (sites per core/worker).
        n_nodes : int
            Number of nodes being run.
        node_index : int
            Index of this node job (if a multi node job is being run).
        out_dir : str
            Directory to dump temporary output files.

        Returns
        -------
        site_chunks : list
             List of slices setting the site chunks to be run by this job.
        site_chunks_index : list
            List of integers setting the site chunk indices to be run by
            this job.
        f_out_chunks : dict
            Dictionary of file output paths keyed by the site chunk indices.
        """

        arr = meta.index.values
        tmp = np.array_split(arr, np.ceil(len(arr) / sites_per_worker))
        site_chunks = [slice(x.min(), x.max() + 1) for x in tmp]
        site_chunks_index = list(range(len(site_chunks)))

        site_chunks = np.array_split(np.array(site_chunks),
                                     n_nodes)[node_index].tolist()
        site_chunks_index = np.array_split(np.array(site_chunks_index),
                                           n_nodes)[node_index].tolist()

        f_out_chunks = {}
        chunk_dir = os.path.join(out_dir, 'chunks/')
        if not os.path.exists(chunk_dir):
            os.makedirs(chunk_dir)
        for ichunk in site_chunks_index:
            f_out = os.path.join(chunk_dir, 'temp_out_{}.h5'.format(ichunk))
            f_out_chunks[ichunk] = f_out

        return site_chunks, site_chunks_index, f_out_chunks

    @property
    def meta(self):
        """Get the full NSRDB meta data."""
        if self._meta is None:
            fpath, Handler = self._tmy_obj._get_fpath('ghi', self._years[0])
            with FS(fpath) as f:
                with Handler(f) as res:
                    self._meta = res.meta.iloc[self._site_slice, :]

        return self._meta

    @property
    def dsets(self):
        """Get the NSRDB datasets excluding meta and time index."""
        if self._dsets is None:
            fpath, Handler = self._tmy_obj._get_fpath('ghi', self._years[0])
            with FS(fpath) as f:
                with Handler(f) as res:
                    self._dsets = []
                    for d in res.dsets:
                        if res.shapes[d] == res.shape:
                            self._dsets.append(d)

            if self._supplemental_fp is not None:
                self._dsets += list(self._supplemental_fp.keys())

            self._dsets.append('tmy_year')
            self._dsets.append('tmy_year_short')
            self._dsets = list(set(self._dsets))

        return self._dsets

    @property
    def site_chunks(self):
        """Get a list of site chunk slices to parallelize across"""
        return self._site_chunks

    @staticmethod
    def get_dset_attrs(dsets, var_meta=None):
        """Get output file dataset attributes for a set of datasets.

        Parameters
        ----------
        dsets : list
            List of dataset / variable names.
        var_meta : str | pd.DataFrame | None
            CSV file or dataframe containing meta data for all NSRDB variables.
            Defaults to the NSRDB var meta csv in git repo.

        Returns
        -------
        attrs : dict
            Dictionary of dataset attributes keyed by dset name.
        chunks : dict
            Dictionary of chunk tuples keyed by dset name.
        dtypes : dict
            dictionary of numpy datatypes keyed by dset name.
        """

        attrs = {}
        chunks = {}
        dtypes = {}

        for dset in dsets:
            var_obj = VarFactory.get_base_handler(dset, var_meta=var_meta)
            attrs[dset] = var_obj.attrs
            chunks[dset] = var_obj.chunks
            dtypes[dset] = var_obj.final_dtype

            if 'units' in attrs[dset]:
                attrs[dset]['psm_units'] = attrs[dset]['units']
            if 'scale_factor' in attrs[dset]:
                attrs[dset]['psm_scale_factor'] = attrs[dset]['scale_factor']

        return attrs, chunks, dtypes

    def _collect(self, purge_chunks=False):
        """Collect all chunked files into the final fout."""

        status_file = os.path.join(self._out_dir, 'collect_status.txt')
        status = self._pre_collect(status_file)
        self._init_final_fout()

        with Outputs(self._final_fpath, mode='a', unscale=True) as out:
            for i, f_out_chunk in self._f_out_chunks.items():
                site_slice = self.site_chunks[i]

                if os.path.basename(f_out_chunk) in status:
                    logger.info('Skipping file, already collected: {}'
                                .format(os.path.basename(f_out_chunk)))
                else:
                    with Resource(f_out_chunk, unscale=True) as chunk:
                        for dset in self.dsets:

                            try:
                                data = chunk[dset]
                            except Exception as e:
                                m = ('Could not read file dataset "{}" from '
                                     'file "{}". Received the following '
                                     'exception: \n{}'
                                     .format(dset,
                                             os.path.basename(f_out_chunk), e))
                                logger.exception(m)
                                raise e
                            else:
                                out[dset, :, site_slice] = data

                    logger.info('Finished collecting #{} out of {} for sites '
                                '{} from file {}'
                                .format(i + 1, len(self._f_out_chunks),
                                        site_slice,
                                        os.path.basename(f_out_chunk)))
                    with open(status_file, 'a') as f:
                        f.write('{}\n'.format(os.path.basename(f_out_chunk)))

        if purge_chunks:
            chunk_dir = os.path.dirname(list(self._f_out_chunks.values())[0])
            logger.info('Purging chunk directory: {}'.format(chunk_dir))
            shutil.rmtree(chunk_dir)

    def _pre_collect(self, status_file):
        """Check to see if all chunked files exist before running collect

        Parameters
        ----------
        status_file : str
            Filepath to status file with a line for each file that has been
            collected.

        Returns
        -------
        status : list
            List of filenames that have already been collected.
        """
        missing = [fp for fp in self._f_out_chunks.values()
                   if not os.path.exists(fp)]
        if any(missing):
            emsg = 'Chunked file outputs are missing: {}'.format(missing)
            logger.error(emsg)
            raise FileNotFoundError(emsg)
        else:
            msg = 'All chunked files found. Running collection.'
            logger.info(msg)

        status = []
        if os.path.exists(status_file):
            with open(status_file) as f:
                status = f.readlines()
                status = [s.strip('\n') for s in status]
        return status

    def _init_final_fout(self):
        """Initialize the final output file."""
        self._init_file(self._final_fpath, self.dsets,
                        self._tmy_obj.time_index, self.meta,
                        var_meta=self._var_meta)

    @staticmethod
    def _init_file(f_out, dsets, time_index, meta, var_meta=None):
        """Initialize an output file.

        Parameters
        ----------
        f_out : str
            File path to final .h5 file.
        dsets : list
            List of dataset names to initialize
        time_index : pd.datetimeindex
            Time index to init to file.
        meta : pd.DataFrame
            Meta data to init to file.
        var_meta : str | pd.DataFrame | None
            CSV file or dataframe containing meta data for all NSRDB variables.
            Defaults to the NSRDB var meta csv in git repo.
        """

        if not os.path.isfile(f_out):
            dsets_mod = [d for d in dsets if 'tmy_year' not in d]
            attrs, chunks, dtypes = TmyRunner.get_dset_attrs(dsets_mod,
                                                             var_meta=var_meta)
            dsets_mod.append('tmy_year')
            attrs['tmy_year'] = {'units': 'selected_year',
                                 'scale_factor': 1,
                                 'psm_units': 'selected_year',
                                 'psm_scale_factor': 1}
            chunks['tmy_year'] = chunks['dni']
            dtypes['tmy_year'] = np.uint16
            Outputs.init_h5(f_out, dsets_mod, attrs, chunks, dtypes,
                            time_index, meta)

            with h5py.File(f_out, mode='a') as f:
                d = 'tmy_year_short'
                f.create_dataset(d, shape=(12, len(meta)), dtype=np.uint16)
                f[d].attrs['units'] = 'selected_year'
                f[d].attrs['scale_factor'] = 1
                f[d].attrs['psm_units'] = 'selected_year'
                f[d].attrs['psm_scale_factor'] = 1

    @staticmethod
    def _write_output(f_out, data_dict, time_index, meta, var_meta=None):
        """Initialize and write an output file chunk.

        Parameters
        ----------
        f_out : str
            File path to final .h5 file.
        data_dict : dict
            {Dset: data_arr} dictionary
        time_index : pd.datetimeindex
            Time index to init to file.
        meta : pd.DataFrame
            Meta data to init to file.
        var_meta : str | pd.DataFrame | None
            CSV file or dataframe containing meta data for all NSRDB variables.
            Defaults to the NSRDB var meta csv in git repo.
        """

        logger.debug('Saving TMY results to: {}'.format(f_out))

        TmyRunner._init_file(f_out, list(data_dict.keys()), time_index, meta,
                             var_meta=var_meta)

        with Outputs(f_out, mode='a') as f:
            for dset, arr in data_dict.items():
                f[dset] = arr

    @staticmethod
    def _run_file(fp):
        """Check whether to run tmy for a given output filepath based on
        whether that file already exists and its file size."""
        run = True
        if os.path.exists(fp):
            size = os.path.getsize(fp)
            if size > 1e6:
                run = False

        return run

    @staticmethod
    def run_single(nsrdb_base_fp, years, weights, site_slice, dsets, f_out,
                   supplemental_fp=None, var_meta=None):
        """Run TMY for a single site chunk (slice) and save to disk.

        Parameters
        ----------
        nsrdb_base_fp : str
            Base nsrdb filepath to retrieve annual files from. Must include
            a single {} format option for the year. Can include * for an
            NSRDB multi file source.
        years : iterable
            Iterable of years to include in the TMY calculation.
        weights : dict
            Lookup of {dset: weight} where dset is a variable h5 dset name
            and weight is a fractional TMY weighting. All weights must
            sum to 1.0
        site_slice : slice
            Sites to consider in this TMY chunk.
        dsets : list
            List of TMY datasets to make.
        f_out : str
            Filepath to save file for this chunk.
        supplemental_fp : None | dict
            Supplemental data base filepaths including {} for year for
            uncommon dataset inputs to the TMY calculation. For example:
            {'poa': '/projects/pxs/poa_h5_dir/poa_out_{}.h5'}
        var_meta : str | pd.DataFrame | None
            CSV file or dataframe containing meta data for all NSRDB variables.
            Defaults to the NSRDB var meta csv in git repo.

        Returns
        -------
        True
        """

        run = TmyRunner._run_file(f_out)

        if run:
            data_dict = {}
            tmy = Tmy(nsrdb_base_fp, years, weights, site_slice,
                      supplemental_fp=supplemental_fp)
            for dset in dsets:
                data_dict[dset] = tmy.get_tmy_timeseries(dset)
            TmyRunner._write_output(f_out, data_dict, tmy.time_index, tmy.meta,
                                    var_meta=var_meta)
        else:
            logger.info('Skipping chunk, f_out already exists: {}'
                        .format(f_out))

        return True

    def _run_serial(self):
        """Run serial tmy futures and save temp chunks to disk."""

        for i, site_slice in enumerate(self.site_chunks):
            fi = self._site_chunks_index[i]
            f_out = self._f_out_chunks[fi]
            if self._run_file(f_out):
                self.run_single(self._nsrdb_base_fp, self._years,
                                self._weights, site_slice, self.dsets, f_out,
                                supplemental_fp=self._supplemental_fp,
                                var_meta=self._var_meta)
            else:
                logger.info('Skipping, already exists: {}'.format(f_out))

            logger.info('{} out of {} TMY chunks completed.'
                        .format(i + 1, len(self.site_chunks)))

    def _run_parallel(self):
        """Run parallel tmy futures and save temp chunks to disk."""
        futures = {}
        loggers = ['nsrdb']
        with SpawnProcessPool(loggers=loggers) as exe:
            logger.info('Kicking off {} futures.'
                        .format(len(self.site_chunks)))
            for i, site_slice in enumerate(self.site_chunks):
                fi = self._site_chunks_index[i]
                f_out = self._f_out_chunks[fi]
                if self._run_file(f_out):
                    future = exe.submit(
                        self.run_single, self._nsrdb_base_fp, self._years,
                        self._weights, site_slice, self.dsets, f_out,
                        supplemental_fp=self._supplemental_fp,
                        var_meta=self._var_meta)
                    futures[future] = i
                else:
                    logger.info('Skipping, already exists: {}'.format(f_out))

            logger.info('Finished kicking off {} futures.'
                        .format(len(futures)))

            for i, future in enumerate(as_completed(futures)):
                if future.result():
                    logger.info('{} out of {} futures completed.'
                                .format(i + 1, len(futures)))
                else:
                    logger.warning('Future #{} failed!'.format(i + 1))

    @classmethod
    def tgy(cls, nsrdb_base_fp, years, out_dir, fn_out, weights=None,
            n_nodes=1, node_index=0,
            log=True, log_level='INFO', log_file=None,
            site_slice=None, supplemental_fp=None, var_meta=None):
        """Run the TGY."""
        if log:
            init_logger('nsrdb.tmy', log_level=log_level, log_file=log_file)
        if weights is None:
            weights = {'sum_ghi': 1.0}
        tgy = cls(nsrdb_base_fp, years, weights, out_dir=out_dir,
                  fn_out=fn_out, n_nodes=n_nodes, node_index=node_index,
                  site_slice=site_slice, supplemental_fp=supplemental_fp,
                  var_meta=var_meta)
        tgy._run_parallel()

    @classmethod
    def tdy(cls, nsrdb_base_fp, years, out_dir, fn_out, weights=None,
            n_nodes=1, node_index=0,
            log=True, log_level='INFO', log_file=None,
            site_slice=None, supplemental_fp=None, var_meta=None):
        """Run the TDY."""
        if log:
            init_logger('nsrdb.tmy', log_level=log_level, log_file=log_file)
        if weights is None:
            weights = {'sum_dni': 1.0}
        tdy = cls(nsrdb_base_fp, years, weights, out_dir=out_dir,
                  fn_out=fn_out, n_nodes=n_nodes, node_index=node_index,
                  site_slice=site_slice, supplemental_fp=supplemental_fp,
                  var_meta=var_meta)
        tdy._run_parallel()

    @classmethod
    def tmy(cls, nsrdb_base_fp, years, out_dir, fn_out, weights=None,
            n_nodes=1, node_index=0, log=True, log_level='INFO', log_file=None,
            site_slice=None, supplemental_fp=None, var_meta=None):
        """Run the TMY. Option for custom weights."""
        if log:
            init_logger('nsrdb.tmy', log_level=log_level, log_file=log_file)

        if weights is None:
            weights = {'max_air_temperature': 0.05,
                       'min_air_temperature': 0.05,
                       'mean_air_temperature': 0.1,
                       'max_dew_point': 0.05,
                       'min_dew_point': 0.05,
                       'mean_dew_point': 0.1,
                       'max_wind_speed': 0.05,
                       'mean_wind_speed': 0.05,
                       'sum_dni': 0.25,
                       'sum_ghi': 0.25}

        tmy = cls(nsrdb_base_fp, years, weights, out_dir=out_dir,
                  fn_out=fn_out, n_nodes=n_nodes, node_index=node_index,
                  site_slice=site_slice, supplemental_fp=supplemental_fp,
                  var_meta=var_meta)
        tmy._run_parallel()

    @classmethod
    def collect(cls, nsrdb_base_fp, years, out_dir, fn_out,
                site_slice=None, supplemental_fp=None, var_meta=None,
                log=True, log_level='INFO', log_file=None):
        """Run TMY collection.

        Parameters
        ----------
        nsrdb_base_fp : str
            Base nsrdb filepath to retrieve annual files from. Must include
            a single {} format option for the year. Can include * for an
            NSRDB multi file source.
        years : iterable
            Iterable of years to include in the TMY calculation.
        out_dir : str
            Directory to dump temporary output files.
        fn_out : str
            Final output filename.
        site_slice : slice
            Sites to consider in this TMY.
        supplemental_fp : None | dict
            Supplemental data base filepaths including {} for year for
            uncommon dataset inputs to the TMY calculation. For example:
            {'poa': '/projects/pxs/poa_h5_dir/poa_out_{}.h5'}
        var_meta : str
            CSV filepath containing meta data for all NSRDB variables.
            Defaults to the NSRDB var meta csv in git repo.
        log : bool
            Whether to write to logger
        log_level : str
            Log level for log output
        log_file : str | None
            Optional log file to write log output
        """
        if log:
            init_logger('nsrdb.tmy', log_level=log_level, log_file=log_file)
        weights = {'sum_ghi': 1.0}
        tgy = cls(nsrdb_base_fp, years, weights, out_dir=out_dir,
                  fn_out=fn_out, n_nodes=1, node_index=0,
                  site_slice=site_slice, supplemental_fp=supplemental_fp,
                  var_meta=var_meta)
        tgy._collect()

    @staticmethod
    def _hpc(fun_str, arg_str, alloc='pxs', memory=90, walltime=4,
             feature='--qos=high', node_name='tmy', stdout_path=None):
        """Run a TmyRunner method on an HPC node.

        Format: TmyRunner.fun_str(arg_str)

        Parameters
        ----------
        fun_str : str
            Name of the class or static method belonging to the TmyRunner class
            to execute in the SLURM job.
        arg_str : str
            Arguments passed to the target method.
        alloc : str
            SLURM project allocation.
        memory : int
            Node memory request in GB.
        walltime : int
            Node walltime request in hours.
        feature : str
            Additional flags for SLURM job. Format is "--qos=high"
            or "--depend=[state:job_id]".
        node_name : str
            Name for the SLURM job.
        stdout_path : str
            Path to dump the stdout/stderr files.
        """

        if stdout_path is None:
            stdout_path = os.getcwd()

        cmd = ("python -c 'from nsrdb.tmy.tmy import TmyRunner;"
               "TmyRunner.{f}({a})'")

        cmd = cmd.format(f=fun_str, a=arg_str)

        slurm_manager = SLURM()
        out = slurm_manager.sbatch(cmd,
                                   alloc=alloc,
                                   memory=memory,
                                   walltime=walltime,
                                   feature=feature,
                                   name=node_name,
                                   stdout_path=stdout_path)[0]

        print('\ncmd:\n{}\n'.format(cmd))

        if out:
            msg = ('Kicked off job "{}" (SLURM jobid #{}) on '
                   'HPC.'.format(node_name, out))
        else:
            msg = ('Was unable to kick off job "{}". '
                   'Please see the stdout error messages'
                   .format(node_name))
        print(msg)

    @classmethod
    def hpc_tmy(cls, fun_str, nsrdb_base_fp, years, out_dir, fn_out,
                weights=None, n_nodes=1, site_slice=None,
                supplemental_fp=None, var_meta=None, **kwargs):
        """Run a TMY/TDY/TGY job on an HPC node."""

        if isinstance(weights, dict):
            weights = json.dumps(weights)
        if isinstance(supplemental_fp, dict):
            supplemental_fp = json.dumps(supplemental_fp)

        node_name = kwargs.get('node_name', None)

        for node_index in range(n_nodes):
            arg_str = ('"{nsrdb_base_fp}", {years}, "{out_dir}", "{fn_out}", '
                       'weights={weights}, '
                       'n_nodes={n_nodes}, '
                       'node_index={node_index}, '
                       'site_slice={site_slice}, '
                       'supplemental_fp={sfps}, '
                       'var_meta="{var_meta}"')
            arg_str = arg_str.format(nsrdb_base_fp=nsrdb_base_fp, years=years,
                                     out_dir=out_dir, fn_out=fn_out,
                                     weights=weights, n_nodes=n_nodes,
                                     node_index=node_index,
                                     site_slice=site_slice,
                                     sfps=supplemental_fp,
                                     var_meta=var_meta)
            sig = signature(getattr(cls, fun_str))
            for k in sig.parameters:
                if k in kwargs:
                    arg_str += f', {k}="{kwargs.pop(k)}"'

            if 'stdout_path' not in kwargs:
                kwargs['stdout_path'] = os.path.join(out_dir, 'stdout/')
            if node_name is None:
                kwargs['node_name'] = '{}{}'.format(fun_str, node_index)
            else:
                kwargs['node_name'] = '{}{}'.format(node_name, node_index)

            cls._hpc(fun_str, arg_str, **kwargs)

    @classmethod
    def hpc_all(cls, nsrdb_base_fp, years, out_dir, n_nodes=1,
                site_slice=None, supplemental_fp=None, var_meta=None,
                **kwargs):
        """Submit three hpc jobs for TMY, TGY, and TDY.

        Parameters
        ----------
        nsrdb_base_fp : str
            Base nsrdb filepath to retrieve annual files from. Must include
            a single {} format option for the year. Can include * for an
            NSRDB multi file source.
        years : iterable
            Iterable of years to include in the TMY calculation.
        out_dir : str
            Directory to dump temporary output files.
        n_nodes : int
            Number of hpc nodes to use for jobs.
        site_slice : slice
            Sites to consider in this TMY.
        supplemental_fp : None | dict
            Supplemental data base filepaths including {} for year for
            uncommon dataset inputs to the TMY calculation. For example:
            {'poa': '/projects/pxs/poa_h5_dir/poa_out_{}.h5'}
        var_meta : str
            CSV filepath containing meta data for all NSRDB variables.
            Defaults to the NSRDB var meta csv in git repo.
        kwargs : dict
            Optional keyword arguments.
        """

        for fun_str in ('tmy', 'tgy', 'tdy'):
            y = sorted(list(years))[-1]
            fun_out_dir = os.path.join(out_dir, '{}_{}/'.format(fun_str, y))
            fun_fn_out = 'nsrdb_{}-{}.h5'.format(fun_str, y)
            cls.hpc_tmy(fun_str, nsrdb_base_fp, years, fun_out_dir,
                        fun_fn_out, n_nodes=n_nodes, site_slice=site_slice,
                        supplemental_fp=supplemental_fp,
                        var_meta=var_meta, **kwargs)

    @classmethod
    def hpc_collect(cls, nsrdb_base_fp, years, out_dir, fn_out,
                    site_slice=None, supplemental_fp=None,
                    var_meta=None, **kwargs):
        """Run a TMY/TDY/TGY file collection job on an HPC node."""

        if isinstance(supplemental_fp, dict):
            supplemental_fp = json.dumps(supplemental_fp)

        arg_str = ('"{nsrdb_base_fp}", {years}, "{out_dir}", "{fn_out}", '
                   'site_slice={site_slice}, supplemental_fp={supp_dirs}, '
                   'var_meta="{var_meta}"')
        arg_str = arg_str.format(nsrdb_base_fp=nsrdb_base_fp, years=years,
                                 out_dir=out_dir, fn_out=fn_out,
                                 site_slice=site_slice,
                                 supp_dirs=supplemental_fp,
                                 var_meta=var_meta)
        sig = signature(cls.collect)
        for k in sig.parameters:
            if k in kwargs:
                arg_str += f', {k}="{kwargs.pop(k)}"'

        if 'stdout_path' not in kwargs:
            kwargs['stdout_path'] = os.path.join(out_dir, 'stdout/')
        if 'node_name' not in kwargs:
            kwargs['node_name'] = \
                'col_{}'.format(os.path.basename(fn_out.replace('.h5', '')))

        cls._hpc('collect', arg_str, **kwargs)

    @classmethod
    def hpc_collect_all(cls, nsrdb_base_fp, years, out_dir,
                        site_slice=None, var_meta=None, **kwargs):
        """Submit three hpc jobs to collect TMY, TGY, and TDY
        (directory setup depends on having run hpc_all() first)."""

        for fun_str in ('tmy', 'tgy', 'tdy'):
            y = sorted(years)[-1]
            fun_out_dir = os.path.join(out_dir, '{}_{}/'.format(fun_str, y))
            fun_fn_out = 'nsrdb_{}-{}.h5'.format(fun_str, y)
            cls.hpc_collect(nsrdb_base_fp, years, fun_out_dir,
                            fun_fn_out, site_slice=site_slice,
                            var_meta=var_meta, **kwargs)
