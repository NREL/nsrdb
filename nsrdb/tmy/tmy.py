"""NSRDB Typical Meteorological Year (TMY) code."""

import datetime
import logging
import os
from copy import deepcopy
from itertools import groupby

import numpy as np
import pandas as pd
from cloud_fs import FileSystem

from nsrdb.file_handlers.resource import MultiFileResource, Resource
from nsrdb.tmy.cdf import Cdf
from nsrdb.utilities.file_utils import pd_date_range

from .utilities import drop_leap, make_time_masks

logger = logging.getLogger(__name__)


class Tmy:
    """NSRDB Typical Meteorological Year (TMY) calculation framework."""

    def __init__(
        self,
        nsrdb_base_fp,
        years,
        weights,
        site_slice=None,
        supplemental_fp=None,
    ):
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

        logger.debug(
            'Initializing TMY algorithm for sites {}...'.format(site_slice)
        )

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
            for dset in self._weights:
                fpath, Handler = self._get_fpath(dset, year)
                dset = self._strip_dset_fun(dset)[0]
                with Handler(fpath) as f:
                    if dset not in f.dsets:
                        e = 'Weight dset "{}" not found in file: "{}"'.format(
                            dset, os.path.basename(fpath)
                        )
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

        dtype = np.float32 if unscale else np.int32

        for year in self._years:
            fpath, Handler = self._get_fpath(dset, year)
            with FileSystem(fpath) as f, Handler(f, unscale=unscale) as res:
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
                with FileSystem(fpath) as f, Resource(f, unscale=False) as res:
                    ti = res.time_index

                temp = drop_leap(temp, ti)[0]

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
        if dset == 'mean_air_temperature' and self._d_mean_temp is not None:
            return self.daily_mean_temp

        dset, fun = self._strip_dset_fun(dset)

        if dset in self._raw_data_cache:
            arr = self._raw_data_cache[dset]
        else:
            arr = self._get_my_arr_raw(dset, unscale=unscale)
            self._raw_data_cache[dset] = arr

        arr = self._resample_arr_daily(arr, self.my_time_index, fun)

        return arr

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
            self._my_time_index = pd_date_range(
                start=start, end=end, freq=self.source_freq, closed='left'
            )
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
            df = pd.DataFrame(
                np.arange(len(self.my_time_index)), index=self.my_time_index
            )
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
            self._time_index = pd_date_range(
                start=start, end=end, freq='1h', closed='left'
            )
            self._time_index += datetime.timedelta(minutes=30)
            if len(self._time_index) != 8760:
                _, self._time_index = drop_leap(
                    np.zeros((len(self._time_index), 1)), self._time_index
                )
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
        if self._tmy_years_short is None:
            self._tmy_years_short = self.calculate_tmy_years()
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
            self._time_masks = make_time_masks(self.my_time_index)
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
            self._daily_time_masks = make_time_masks(self.my_daily_time_index)
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
            with FileSystem(fpath) as f, Handler(f) as res:
                meta = res.meta

            self._meta = meta.iloc[self._site_slice, :]
        return self._meta

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
            raise ValueError(
                'Could not parse source temporal frequency '
                'from time index length {}'.format(ti_len)
            )
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
                raise ValueError(
                    'Bad array length of {} when daily ti is {}'.format(
                        len(arr), len(ti)
                    )
                )
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

        years = {
            m: np.zeros((n, len(self.meta)), dtype=np.uint16)
            for m in range(1, 13)
        }
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
            lt_mean[(m - 1), :] = np.mean(
                self.daily_total_ghi[mask, :], axis=0
            )
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
            lt_median[(m - 1), :] = np.median(
                self.daily_total_ghi[mask, :], axis=0
            )
        return lt_median

    def _lt_mm_difFileSystem(self):
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
        diffs = {m: np.zeros(shape, dtype=np.float32) for m in range(1, 13)}

        for i, y in enumerate(self.years):
            for m in range(1, 13):
                mask = self.daily_time_masks[m] & self.daily_time_masks[y]
                this_mean = np.mean(self.daily_total_ghi[mask, :], axis=0)
                this_median = np.median(self.daily_total_ghi[mask, :], axis=0)

                im = m - 1
                diffs[m][i, :] = np.abs(lt_mean[im, :] - this_mean) + np.abs(
                    lt_median[im, :] - this_median
                )
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
        sorted_years = {
            m: np.zeros(shape, dtype=np.float32) for m in range(1, 13)
        }
        diffs = self._lt_mm_difFileSystem()

        for m in range(1, 13):
            sorted_years[m] = np.argsort(diffs[m], axis=0) + self.years[0]

        for site in range(len(self.meta)):
            for m in range(1, 13):
                temp = [
                    y
                    for y in sorted_years[m][:, site]
                    if y in tmy_years_5[m][:, site]
                ]
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

            t_low = self.daily_mean_temp < t33
            t_high = self.daily_mean_temp > t67
            g_low = self.daily_total_ghi < g33

            for j in range(len(self.meta)):
                max_run_len[m][j] = [0] * tmy_years_5[m].shape[0]
                n_runs[m][j] = [0] * tmy_years_5[m].shape[0]

                for i, y in enumerate(tmy_years_5[m][:, j]):
                    y_mask = self.daily_time_masks[y]
                    mask = y_mask & m_mask

                    for arr in [t_low, t_high, g_low]:
                        m_temp, n_temp = self._count_runs(arr[mask, j])
                        max_run_len[m][j][i] = np.max((
                            max_run_len[m][j][i],
                            m_temp,
                        ))
                        n_runs[m][j][i] += n_temp

                tmy_years[(m - 1), j] = tmy_years_5[m][0, j]
                for i, y in enumerate(tmy_years_5[m][:, j]):
                    screen_out = (
                        (max(max_run_len[m][j]) == max_run_len[m][j][i])
                        | (max(n_runs[m][j]) == n_runs[m][j][i])
                        | (n_runs == 0)
                    )
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
        year_set = sorted(set(tmy_years.flatten()))
        data = None
        masks = None
        self._tmy_years_long = None

        for year in year_set:
            fpath, Handler = self._get_fpath(dset, year)
            with FileSystem(fpath) as f, Handler(f, unscale=unscale) as res:
                ti = res.time_index
                temp = res[dset, :, self._site_slice]
                temp, ti = drop_leap(temp, ti)

            if masks is None:
                masks = {m: (ti.month == m) for m in range(1, 13)}
            if data is None:
                data = np.zeros(temp.shape, dtype=temp.dtype)
            if self._tmy_years_long is None:
                self._tmy_years_long = np.zeros(temp.shape, dtype=np.uint16)

            mask = tmy_years == year
            locs = np.where(mask)
            months = locs[0] + 1
            sites = locs[1]

            for month, site in zip(months, sites):
                data[masks[month], site] = temp[masks[month], site]
                self._tmy_years_long[masks[month], site] = tmy_years[
                    (month - 1), site
                ]

        step = len(data) // 8760
        msg = (
            'Original TMY timeseries is not divisible by 8760 '
            f'length = {len(data)}.'
        )
        assert len(data) % 8760 == 0, msg

        data = data[::step, :]
        self._tmy_years_long = self._tmy_years_long[::step, :]

        return data

    def calculate_tmy_years(self):
        """Calculate the TMY based on the multiple-dataset weights.

        Returns
        -------
        tmy_years : np.ndarray
            Array of best TMY years for every month for every site.
            Shape is (months, sites).
        """
        logger.info(
            'Calculating best TMY years for every month, for '
            f'{len(self.meta)} sites.'
        )
        ws = self.get_weighted_fs()
        tmy_years_5, _ = self.select_fs_years(ws)
        tmy_years_5, _ = self.sort_years_mm(tmy_years_5)
        tmy_years, _, _ = self.persistence_filter(tmy_years_5)
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
        if dset in ('tmy_year_long', 'tmy_year'):
            return self.tmy_years_long
        return self._make_tmy_timeseries(
            dset, self.tmy_years_short, unscale=unscale
        )
