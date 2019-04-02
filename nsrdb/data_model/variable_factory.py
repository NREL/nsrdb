# -*- coding: utf-8 -*-
"""Frameworks for handling NSRDB data sources."""

import numpy as np
import pandas as pd
import h5py
import os
from netCDF4 import Dataset as NetCDF
import logging
import datetime
from warnings import warn

from nsrdb import DATADIR


logger = logging.getLogger(__name__)


class AncillaryVar:
    """Base class for ancillary variable processing."""

    # default source data directory
    DEFAULT_DIR = DATADIR

    def __init__(self, var_meta, name, date):
        """
        Parameters
        ----------
        var_meta : str | pd.DataFrame
            CSV file or dataframe containing meta data for all NSRDB variables.
        name : str
            NSRDB var name.
        date : datetime.date
            Single day to extract data for.
        """
        self._var_meta = self._parse_var_meta(var_meta)
        self._name = name
        self._date = date

    @staticmethod
    def _parse_var_meta(inp):
        """Set the meta data for NSRDB variables.

        Parameters
        ----------
        inp : str
            CSV file containing meta data for all NSRDB variables.
        """
        var_meta = None
        if isinstance(inp, str):
            if inp.endswith('.csv'):
                var_meta = pd.read_csv(inp)
                logger.debug('Imported NSRDB variable meta file.')
        elif isinstance(inp, pd.DataFrame):
            var_meta = inp

        if var_meta is None:
            raise TypeError('Could not parse meta data for NSRDB variables '
                            'from: {}'.format(inp))
        return var_meta

    @property
    def var_meta(self):
        """Return the meta data for NSRDB variables.

        Returns
        -------
        _var_meta : pd.DataFrame
            Meta data for NSRDB variables.
        """
        return self._var_meta

    @property
    def name(self):
        """Get the NSRDB variable name."""
        return self._name

    @property
    def mask(self):
        """Get a boolean mask to locate the current variable in the meta data.
        """
        if not hasattr(self, '_mask'):
            self._mask = self.var_meta['var'] == self._name
        return self._mask

    @property
    def elevation_correct(self):
        """Get the elevation correction preference.

        Returns
        -------
        elevation_correct : bool
            Whether or not to use elevation correction for the current var.
        """
        temp = self.var_meta.loc[self.mask, 'elevation_correct']
        return bool(temp.values[0])

    @property
    def spatial_method(self):
        """Get the spatial interpolation method.

        Returns
        -------
        spatial_method : str
            NN or IDW
        """
        return str(self.var_meta.loc[self.mask, 'spatial_interp'].values[0])

    @property
    def source_dir(self):
        """Get the source directory containing the variable data files.

        Returns
        -------
        source_dir : str
            Directory containing source data files (with possible sub folders).
        """
        d = self.var_meta.loc[self.mask, 'source_directory'].values[0]
        if not d:
            warn('Using default data directory for "{}"'.format(self.name))
            d = self.DEFAULT_DIR
        return str(d)

    @property
    def temporal_method(self):
        """Get the temporal interpolation method.

        Returns
        -------
        temporal_method : str
            linear or nearest
        """
        return str(self.var_meta.loc[self.mask, 'temporal_interp'].values[0])

    @property
    def dset(self):
        """Get the MERRA dset name from the NSRDB variable name.

        Returns
        -------
        dset : str
            MERRA dset name, e.g.:
                tavg1_2d_aer_Nx
                tavg1_2d_ind_Nx
                tavg1_2d_rad_Nx
                tavg1_2d_slv_Nx
        """
        return str(self.var_meta.loc[self.mask, 'merra_dset'].values[0])

    @staticmethod
    def _get_time_index(date, freq='1h'):
        """Get a pandas date time object for the given analysis date.

        Parameters
        ----------
        date : datetime.date
            Single day to get time index for.
        freq : str
            Pandas datetime frequency, e.g. '1h', '5min', etc...

        Returns
        -------
        ti : pd.DatetimeIndex
            Pandas datetime index for the current day.
        """

        ti = pd.date_range('1-1-{y}'.format(y=date.year),
                           '1-1-{y}'.format(y=date.year + 1),
                           freq=freq)[:-1]
        mask = (ti.month == date.month) & (ti.day == date.day)
        ti = ti[mask]
        return ti


class AsymVar(AncillaryVar):
    """Framework for Asymmetry variable data extraction."""

    def __init__(self, var_meta, name='asymmetry',
                 date=datetime.date(year=2017, month=1, day=1),
                 fname='asymmetry_clim.h5'):
        """
        Parameters
        ----------
        var_meta : str | pd.DataFrame
            CSV file or dataframe containing meta data for all NSRDB variables.
        name : str
            NSRDB var name.
        date : datetime.date
            Single day to extract data for.
        fname : str
            Asymmetry source data filename.
        """
        self._fname = fname
        super().__init__(var_meta, name, date)

    @property
    def fpath(self):
        """Get the Asymmetry source file path with file name."""
        return os.path.join(self.source_dir, self._fname)

    @property
    def time_index(self):
        """Get the MERRA native time index.

        Returns
        -------
        asym_ti : pd.DatetimeIndex
            Pandas datetime index for the current day at the asymmetry
            resolution (1-month).
        """
        return self._get_time_index(self._date, freq='1D')

    @property
    def source_data(self):
        """Get the asymmetry source data.

        Returns
        -------
        data : np.ndarray
            Single month of asymmetry data with shape (1 x n_sites).
        """

        with h5py.File(self.fpath, 'r') as f:
            # take the data at all sites for the zero-indexed month
            i = self._date.month - 1
            data = f[self.name][i, :]

        # reshape to (1 x n_sites)
        data = data.reshape((1, len(data)))

        return data

    @property
    def grid(self):
        """Get the asymmetry grid.

        Returns
        -------
        _asym_grid : pd.DataFrame
            Asymmetry grid data with columns 'latitude' and 'longitude'.
        """

        if not hasattr(self, '_asym_grid'):
            with h5py.File(self.fpath, 'r') as f:
                self._asym_grid = pd.DataFrame(f['meta'][...])

            if ('latitude' not in self._asym_grid or
                    'longitude' not in self._asym_grid):
                raise ValueError('Asymmetry file did not have '
                                 'latitude/longitude meta data. '
                                 'Please check: {}'.format(self.fpath))

        return self._asym_grid


class CloudVar(AncillaryVar):
    """Framework for cloud data extraction (GOES data processed by UW)."""

    def __init__(self, var_meta, name, date, extent='east'):
        """
        Parameters
        ----------
        var_meta : str | pd.DataFrame
            CSV file or dataframe containing meta data for all NSRDB variables.
        name : str
            NSRDB var name.
        date : datetime.date
            Single day to extract data for.
        """

        self._extent = extent
        self._extent_path = None
        self._year_path = None
        self._day_path = None
        self._fpath = None
        self._flist = None
        super().__init__(var_meta, name, date)

    def __len__(self):
        """Length of this object is the number of source files."""
        return len(self.flist)

    @property
    def extent_path(self):
        """Path for cloud files in a given extent."""
        if self._extent_path is None:
            self._extent_path = os.path.join(self.source_dir, self._extent)
        return self._extent_path

    @property
    def year_path(self):
        """Path for cloud files in a single year."""
        if self._year_path is None:
            self._year_path = os.path.join(self.extent_path,
                                           str(self._date.year))
        return self._year_path

    @property
    def day_path(self):
        """Path for cloud files in a single day."""
        if self._day_path is None:
            doy = str(self._date.timetuple().tm_yday).zfill(3)
            self._day_path = os.path.join(self.year_path, doy)
        return self._day_path

    @property
    def fpath(self):
        """Final path containing cloud data files."""
        if self._fpath is None:
            self._fpath = os.path.join(self.day_path, 'level2')
        return self._fpath

    @property
    def flist(self):
        """List of cloud data files for one day. Each file is a timestep."""
        if self._flist is None:
            fl = os.listdir(self.fpath)
            self._flist = [f for f in fl if f.endswith('.h5')]
        return self._flist

    @property
    def time_index(self):
        """Get the GOES cloud data time index.

        Returns
        -------
        cloud_ti : pd.DatetimeIndex
            Pandas datetime index for the current day at the cloud temporal
            resolution (should match the NSRDB resolution).
        """
        # the number of files for a given day dictates the temporal frequency
        len_to_freq = {48: '30min',
                       96: '15min',
                       288: '5min'}
        if len(self) in len_to_freq:
            freq = len_to_freq[len(self)]
        else:
            raise KeyError('Number of cloud data files is inconsistent with '
                           'expectations. Counted {} files in {} but expected '
                           'one of the following: {}'
                           .format(len(self), self.fpath,
                                   list(len_to_freq.keys())))
        return self._get_time_index(self._date, freq=freq)


class MerraVar(AncillaryVar):
    """Framework for MERRA source data extraction."""

    # default MERRA paths.
    MERRA_ELEV = os.path.join(DATADIR, 'merra_grid_srtm_500m_stats')

    def __init__(self, var_meta, name, date):
        """
        Parameters
        ----------
        var_meta : str | pd.DataFrame
            CSV file or dataframe containing meta data for all NSRDB variables.
        name : str
            NSRDB var name.
        date : datetime.date
            Single day to extract data for.
        """

        super().__init__(var_meta, name, date)

    @property
    def date_stamp(self):
        """Get the MERRA datestamp corresponding to the specified datetime date

        Returns
        -------
        date : str
            Date stamp that should be in the MERRA file, format is YYYYMMDD
        """

        y = str(self._date.year)
        m = str(self._date.month).zfill(2)
        d = str(self._date.day).zfill(2)
        date = '{y}{m}{d}'.format(y=y, m=m, d=d)
        return date

    @property
    def file(self):
        """Get the MERRA file path for the target NSRDB variable name.

        Returns
        -------
        fmerra : str
            MERRA file path containing the target NSRDB variable.
        """

        path = os.path.join(self.source_dir, self.dset)
        flist = os.listdir(path)
        for f in flist:
            if self.date_stamp in f:
                fmerra = os.path.join(path, f)
                break
        return fmerra

    @property
    def merra_name(self):
        """Get the MERRA variable name from the NSRDB variable name.

        Returns
        -------
        merra_name : str
            MERRA var name.
        """
        return str(self.var_meta.loc[self.mask, 'merra_name'].values[0])

    @property
    def time_index(self):
        """Get the MERRA native time index.

        Returns
        -------
        MERRA_time_index: pd.DatetimeIndex
            Pandas datetime index for the current day at the MERRA2 resolution
            (1-hour).
        """
        return self._get_time_index(self._date, freq='1h')

    @staticmethod
    def _format_2d(data):
        """Format MERRA data as a flat 2D array: (time X sites).

        MERRA data is sourced as a 3D array: (time X sitex X sitey).

        Parameters
        ----------
        data : np.ndarray
            3D numpy array of MERRA data. 1st dim is time, 2nd and 3rd are
            both spatial.

        Returns
        -------
        flat_data : np.ndarray
            2D numpy array of flattened MERRA data. 1st dim is time, 2nd is
            spatial.
        """
        flat_data = np.zeros(shape=(data.shape[0],
                                    data.shape[1] * data.shape[2]),
                             dtype=np.float32)
        for i in range(data.shape[0]):
            flat_data[i, :] = data[i, :, :].ravel()
        return flat_data

    @property
    def source_data(self):
        """Get single variable data from the MERRA source file.

        Returns
        -------
        data : np.ndarray
            2D numpy array (time X space) of MERRA data for the specified var.
        """

        # open NetCDF file
        with NetCDF(self.file, 'r') as f:

            # depending on variable, might need extra logic
            if self.merra_name in ['wind_speed', 'wind_direction']:
                u_vector = f['U2M'][:]
                v_vector = f['V2M'][:]
                if self.merra_name == 'wind_speed':
                    data = np.sqrt(u_vector**2 + v_vector**2)
                else:
                    data = np.degrees(
                        np.arctan2(u_vector, v_vector)) + 180

            elif self.merra_name == 'TOTSCATAU':
                # Single scatter albedo is total scatter / aod
                data = f[self.merra_name][:] / f['TOTEXTTAU'][:]

            else:
                data = f[self.merra_name][:]

        # make the data a flat 2d array
        data = self._format_2d(data)

        return data

    @property
    def grid(self):
        """Return the MERRA source coordinates with elevation.

        It seems that all MERRA files DO NOT have the same grid.

        Returns
        -------
        self._merra_grid : pd.DataFrame
            MERRA source coordinates with elevation
        """

        if not hasattr(self, '_merra_grid'):

            with NetCDF(self.file, 'r') as nc:
                lon2d, lat2d = np.meshgrid(nc['lon'][:], nc['lat'][:])

            self._merra_grid = pd.DataFrame({'longitude': lon2d.ravel(),
                                             'latitude': lat2d.ravel()})

            # merra grid has some bad values around 0 lat/lon
            # quick fix is to set to zero
            self._merra_grid.loc[(self._merra_grid['latitude'] > -0.1) &
                                 (self._merra_grid['latitude'] < 0.1),
                                 'latitude'] = 0
            self._merra_grid.loc[(self._merra_grid['longitude'] > -0.1) &
                                 (self._merra_grid['longitude'] < 0.1),
                                 'longitude'] = 0

            # add elevation to coordinate set
            merra_elev = pd.read_pickle(self.MERRA_ELEV)
            self._merra_grid = self._merra_grid.merge(merra_elev,
                                                      on=['latitude',
                                                          'longitude'],
                                                      how='left')

            # change column name from merra default
            if 'mean_elevation' in self._merra_grid.columns.values:
                self._merra_grid = self._merra_grid.rename(
                    {'mean_elevation': 'elevation'}, axis='columns')

        return self._merra_grid


class VarFactory:
    """Factory pattern to retrieve ancillary variable helper objects."""

    # mapping of NSRDB variable names to helper objects
    MAPPING = {'asymmetry': AsymVar,
               'air_temperature': MerraVar,
               'alpha': MerraVar,
               'aod': MerraVar,
               'surface_pressure': MerraVar,
               'ozone': MerraVar,
               'total_precipitable_water': MerraVar,
               'specific_humidity': MerraVar,
               'ssa': MerraVar,
               'wind_speed': MerraVar,
               }

    def get(self, var_name, *args, **kwargs):
        """Get a processing variable instance for the given var name.

        Parameters
        ----------
        var_name : str
            NSRDB variable name.
        *args : list
            List of positional args for instantiation of ancillary var.
        **kwargs : dict
            List of keyword args for instantiation of ancillary var.

        Returns
        -------
        instance : ancillary object
            Instantiated ancillary variable helper object (AsymVar, MerraVar).
        """

        if var_name in self.MAPPING:
            return self.MAPPING[var_name](*args, **kwargs)

        else:
            raise KeyError('Did not recognize "{}" as an available ancillary '
                           'variable. The following variables are available: '
                           '{}'.format(var_name, list(self.MAPPING.keys())))
