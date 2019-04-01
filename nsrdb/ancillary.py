# -*- coding: utf-8 -*-
"""Performs spatial and temporal interpolation of ancillary data.

Non-standard module dependencies (pip-installable):
    - netCDF4

The following variables can be processed using this module. For details see the
class variables in Ancillary() below.
    ('alpha',
     'asymmetry',
     'aod',
     'surface_pressure',
     'relative_humidity',
     'ssa',
     'ozone',
     'total_precipitable_water',
     'solar_zenith_angle',
     'air_temperature',
     'specific_humidity',
     'wind_speed',
     'dew_point')
"""

import numpy as np
import pandas as pd
import h5py
import os
from netCDF4 import Dataset as NetCDF
import logging
import psutil
from dask.distributed import LocalCluster, Client
import time
import datetime

from nsrdb import NSRDBDIR, DATADIR
from nsrdb.utilities.solar_position import SolarPosition
from nsrdb.utilities.loggers import NSRDB_LOGGERS
from nsrdb.utilities.interpolation import (spatial_interp, geo_nn,
                                           temporal_lin, temporal_step)

logger = logging.getLogger(__name__)


class AncillaryVar:
    """Base class for ancillary variable processing."""

    def __init__(self, var_meta, name, source_dir, date):
        """
        Parameters
        ----------
        var_meta : str
            CSV file containing meta data for all NSRDB variables.
        name : str
            NSRDB var name.
        source_dir : str
            Directory path containing source files.
        date : datetime.date
            Single day to extract data for.
        """
        self._var_meta = var_meta
        self._name = name
        self._source_dir = source_dir
        self._date = date

    @property
    def name(self):
        """Get the NSRDB variable name."""
        return self._name

    @property
    def mask(self):
        """Get a boolean mask to locate the current variable in the meta data.
        """
        if not hasattr(self, '_mask'):
            self._mask = self._var_meta['var'] == self._name
        return self._mask

    @property
    def elevation_correct(self):
        """Get the elevation correction preference.

        Returns
        -------
        elevation_correct : bool
            Whether or not to use elevation correction for the current var.
        """
        temp = self._var_meta.loc[self.mask, 'elevation_correct']
        return bool(temp.values[0])

    @property
    def spatial_method(self):
        """Get the spatial interpolation method.

        Returns
        -------
        spatial_method : str
            NN or IDW
        """
        return str(self._var_meta.loc[self.mask, 'spatial_interp'].values[0])

    @property
    def temporal_method(self):
        """Get the temporal interpolation method.

        Returns
        -------
        temporal_method : str
            linear or nearest
        """
        return str(self._var_meta.loc[self.mask, 'temporal_interp'].values[0])

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
        return str(self._var_meta.loc[self.mask, 'merra_dset'].values[0])


class AsymVar(AncillaryVar):
    """Helper for Asymmetry variable properties and source data extraction."""

    # Default asymmetry path
    ASYM_DIR = DATADIR

    def __init__(self, var_meta, name='asymmetry', source_dir=None,
                 date=datetime.date(year=2017, month=1, day=1),
                 fname='asymmetry_clim.h5'):
        """
        Parameters
        ----------
        var_meta : str
            CSV file containing meta data for all NSRDB variables.
        name : str
            NSRDB var name.
        source_dir : str | NoneType
            Directory path containing ancillary source files. None will use
            default ancillary variables source directories.
        date : datetime.date
            Single day to extract data for.
        fname : str
            Asymmetry source data filename.
        """

        if source_dir is None:
            source_dir = self.ASYM_DIR
        self._fpath = os.path.join(source_dir, fname)
        super().__init__(var_meta, name, source_dir, date)

    @property
    def time_index(self):
        """Get the MERRA native time index.

        Returns
        -------
        asym_ti : pd.DatetimeIndex
            Pandas datetime index for the current day at the asymmetry
            resolution (1-month).
        """
        return Ancillary.get_time_index(self._date, freq='1D')

    @property
    def source_data(self):
        """Get the asymmetry source data.

        Returns
        -------
        data : np.ndarray
            Single month of asymmetry data with shape (1 x n_sites).
        """

        with h5py.File(self._fpath, 'r') as f:
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
            with h5py.File(self._fpath, 'r') as f:
                self._asym_grid = pd.DataFrame(f['meta'][...])

            if ('latitude' not in self._asym_grid or
                    'longitude' not in self._asym_grid):
                raise ValueError('Asymmetry file did not have '
                                 'latitude/longitude meta data. '
                                 'Please check: {}'.format(self._fpath))

        return self._asym_grid


class MerraVar(AncillaryVar):
    """Helper for MERRA variable properties and source data extraction."""

    # default MERRA paths.
    MERRA_DIR = '/lustre/eaglefs/projects/pxs/ancillary/source'
    MERRA_ELEV = os.path.join(DATADIR, 'merra_grid_srtm_500m_stats')

    def __init__(self, var_meta, name, source_dir, date):
        """
        Parameters
        ----------
        var_meta : str
            CSV file containing meta data for all NSRDB variables.
        name : str
            NSRDB var name.
        source_dir : str | NoneType
            Directory path containing ancillary source files. None will use
            default ancillary variables source directories.
        date : datetime.date
            Single day to extract data for.
        """

        if source_dir is None:
            source_dir = self.MERRA_DIR
        super().__init__(var_meta, name, source_dir, date)

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

        path = os.path.join(self._source_dir, self.dset)
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
        return str(self._var_meta.loc[self.mask, 'merra_name'].values[0])

    @property
    def time_index(self):
        """Get the MERRA native time index.

        Returns
        -------
        MERRA_time_index: pd.DatetimeIndex
            Pandas datetime index for the current day at the MERRA2 resolution
            (1-hour).
        """
        return Ancillary.get_time_index(self._date, freq='1h')

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
            if self.MAPPING[var_name] == AsymVar:
                # always use default source dir for Asym
                kwargs['source_dir'] = None
            return self.MAPPING[var_name](*args, **kwargs)

        else:
            raise KeyError('Did not recognize "{}" as an available ancillary '
                           'variable. The following variables are available: '
                           '{}'.format(var_name, list(self.MAPPING.keys())))


class Ancillary:
    """Framework for single-day ancillary data processing to NSRDB."""

    # directory to cache intermediate data (nearest neighbor results)
    CACHE_DIR = NSRDBDIR

    # source files for weight factors
    WEIGHTS = {
        'aod': os.path.join(
            DATADIR, 'Monthly_pixel_correction_MERRA2_AOD.txt'),
        'alpha': os.path.join(
            DATADIR, 'Monthly_pixel_correction_MERRA2_Alpha.txt')}

    # variables used for all-sky that are processed in this module
    ALL_SKY_VARS = ('alpha',
                    'aod',
                    'asymmetry',
                    'ozone',
                    'ssa',
                    'surface_pressure',
                    'total_precipitable_water',
                    'solar_zenith_angle',
                    )

    # variables from MERRA processed in this module
    MERRA_VARS = ('surface_pressure',
                  'air_temperature',
                  'ozone',
                  'total_precipitable_water',
                  'wind_speed',
                  'specific_humidity',
                  'alpha',
                  'aod',
                  'ssa',
                  )

    # calculated variables (no dependencies)
    CALCULATED_VARS = ('solar_zenith_angle',)

    # derived variables (no interp, requires: temp, spec. humidity, pressure)
    DERIVED_VARS = ('relative_humidity',
                    'dew_point',
                    )

    # all variables processed by this module
    ALL_VARS = tuple(set(ALL_SKY_VARS + MERRA_VARS + CALCULATED_VARS +
                         DERIVED_VARS))

    def __init__(self, var_meta, date, source_dir, nsrdb_grid,
                 nsrdb_freq='5min'):
        """
        Parameters
        ----------
        var_meta : str
            CSV file containing meta data for all NSRDB variables.
        date : datetime.date
            Single day to extract MERRA2 data for.
        source_dir : str | NoneType
            Directory path containing ancillary source files. None will use
            default ancillary variables source directories.
        nsrdb_grid : str
            CSV file containing the NSRDB reference grid to interpolate to.
        nsrdb_freq : str
            Final desired NSRDB temporal frequency.
        """

        logger.info('Processing MERRA data for {}'.format(date))

        self._parse_var_meta(var_meta)
        self._parse_nsrdb_grid(nsrdb_grid)
        self._date = date
        self._source_dir = source_dir
        self._nsrdb_freq = nsrdb_freq
        self._var_factory = VarFactory()
        self._processed = {}

        logger.debug('Final NSRDB output shape is: {}'
                     .format(self.nsrdb_data_shape))

    def __getitem__(self, key):
        return self._processed[key]

    def __setitem__(self, key, value):
        self._processed[key] = value

    def _parse_nsrdb_grid(self, inp):
        """Set the NSRDB reference grid from a csv file.

        Parameters
        ----------
        inp : str
            CSV file containing the NSRDB reference grid to interpolate to.
        """

        if inp.endswith('.csv'):
            self._nsrdb_grid = pd.read_csv(inp)
            logger.debug('Imported NSRDB reference grid file.')
        else:
            raise TypeError('Expected csv grid file but received: {}'
                            .format(inp))

    def _parse_var_meta(self, inp):
        """Set the meta data for NSRDB variables.

        Parameters
        ----------
        inp : str
            CSV file containing meta data for all NSRDB variables.
        """
        if inp.endswith('.csv'):
            self._var_meta = pd.read_csv(inp)
            logger.debug('Imported NSRDB variable meta file.')
        else:
            raise TypeError('Expected csv var meta file but received: {}'
                            .format(inp))

    @property
    def date(self):
        """Get the single-day datetime.date for this instance."""
        return self._date

    @property
    def nsrdb_grid(self):
        """Return the grid.

        Returns
        -------
        _nsrdb_grid : pd.DataFrame
            Reference grid data.
        """
        return self._nsrdb_grid

    @property
    def nsrdb_ti(self):
        """Get the NSRDB target time index.

        Returns
        -------
        nsrdb_ti : pd.DatetimeIndex
            Pandas datetime index for the current day at the NSRDB resolution.
        """
        return self.get_time_index(self.date, freq=self._nsrdb_freq)

    @property
    def nsrdb_data_shape(self):
        """Get the final NSRDB data shape for a single var.

        Returns
        -------
        _nsrdb_data_shape : tuple
            Two-entry shape tuple.
        """
        if not hasattr(self, '_nsrdb_data_shape'):
            self._nsrdb_data_shape = (len(self.nsrdb_ti), len(self.nsrdb_grid))
        return self._nsrdb_data_shape

    @property
    def var_meta(self):
        """Return the meta data for NSRDB variables.

        Returns
        -------
        _var_meta : pd.DataFrame
            Meta data for NSRDB variables.
        """
        return self._var_meta

    @staticmethod
    def get_time_index(date, freq='1h'):
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

    def get_nn_ind(self, df1, df2, method, labels=('latitude', 'longitude'),
                   cache=False):
        """Get the geographic nearest neighbor distances and indices.

        Parameters
        ----------
        df1/df2 : pd.DataFrame:
            Dataframes containing coodinate columns with the corresponding
            labels.
        labels : tuple | list
            Column labels corresponding to the lat/lon columns in df1/df2.
        method : str
            Spatial interpolation method - either NN or IDW
        cache : bool | str
            Flag to cache nearest neighbor results or retrieve cached results
            instead of performing NN query. Strings are evaluated as the file
            name to cache.

        Returns
        -------
        dist : ndarray
            Distance array in km returned if return_dist input arg set to True.
        indicies : ndarray
            1D array of row indicies in df1 that match df2.
            df1[df1.index[indicies[i]]] is closest to df2[df2.index[i]]
        """
        if method == 'NN':
            k = 1
        elif method == 'IDW':
            k = 4
        else:
            raise ValueError('Did not recognize spatial interp method: "{}"'
                             .format(method))

        if isinstance(cache, str):
            if not cache.endswith('.csv'):
                cache += '.csv'
            # try to get cached kdtree results. fast for prototyping.
            cache_d = os.path.join(self.CACHE_DIR,
                                   cache.replace('.csv', '_d.csv'))
            cache_i = os.path.join(self.CACHE_DIR,
                                   cache.replace('.csv', '_i.csv'))

            if os.path.exists(cache_i) and os.path.exists(cache_d):
                logger.debug('Found cached nearest neighbor indices, '
                             'importing: {}'.format(cache_i))
                dist = np.genfromtxt(cache_d, dtype=float, delimiter=',')
                ind = np.genfromtxt(cache_i, dtype=int, delimiter=',')

            else:
                logger.debug('Running geographic nearest neighbor...')
                dist, ind = geo_nn(df1, df2, labels=labels, k=k)
                logger.debug('Saving nearest neighbor indices to: {}'
                             .format(cache_i))
                np.savetxt(cache_d, dist, delimiter=',')
                np.savetxt(cache_i, ind, delimiter=',')

        else:
            logger.debug('Running geographic nearest neighbor...')
            dist, ind = geo_nn(df1, df2, labels=labels, k=k)

        return dist, ind

    def get_weights(self, var_obj):
        """Get the irradiance model weights for AOD/Alpha.

        Parameters
        ----------
        var_obj : MerraVar
            Merra processing variable object.

        Returns
        -------
        weights : np.ndarray | NoneType
            1D array of weighting values for the given var in the current
            month. Returns None if var does not require weighting.
        """

        if not hasattr(self, '_weights'):
            self._weights = {}

        if var_obj.name in self.WEIGHTS and var_obj.name not in self._weights:
            logger.debug('Extracting weights for "{}"'.format(var_obj.name))
            weights = pd.read_csv(self.WEIGHTS[var_obj.name], sep=' ',
                                  skiprows=4, skipinitialspace=1)
            weights = weights.rename(
                {'Lat.': 'latitude', 'Long.': 'longitude'}, axis='columns')

            # use geo nearest neighbors to find closest indices
            # between weights and MERRA grid
            _, i_nn = self.get_nn_ind(weights, var_obj.grid, 'NN')
            i_nn = i_nn.flatten()

            df_w = weights.iloc[i_nn.flatten()]
            df_w = df_w[df_w.columns[2:-1]].T.set_index(
                pd.date_range(str(self.date.year), freq='M', periods=12))
            df_w[df_w < 0] = 1

            self._weights[var_obj.name] = df_w

        if var_obj.name in self._weights:
            mask = (self._weights[var_obj.name].index.month ==
                    self.date.month)
            weights = self._weights[var_obj.name][mask].values[0]
        else:
            weights = None

        return weights

    @staticmethod
    def convert_units(var, data):
        """Convert MERRA data to NSRDB units.

        Parameters
        ----------
        var : str
            NSRDB Variable name.
        data : np.ndarray
            Data for var.

        Returns
        -------
        data : np.ndarray
            Data with NSRDB units if conversion is required for "var".
        """

        if var == 'air_temperature':
            if np.max(data) > 100:
                # convert Kelvin to Celsius
                data -= 273.15

        if var == 'surface_pressure':
            if np.max(data) > 10000:
                # convert surface pressure from Pa to mbar
                data /= 100

        elif var == 'ozone':
            # convert Dobson to atm-cm
            data /= 1000.

        elif var == 'total_precipitable_water':
            # Convert precip from kg/m2 to cm
            data *= 0.1

        return data

    @staticmethod
    def relative_humidity(t, h, p):
        """Calculate relative humidity.

        Parameters
        ----------
        t : np.ndarray
            Temperature in Celsius
        h : np.ndarray
            Specific humidity in kg/kg
        p : np.ndarray
            Pressure in Pa

        Returns
        -------
        rh : np.ndarray
            Relative humidity in %.
        """

        # ensure that Pressure is in Pa (scale from mbar if not)
        convert_p = False
        if np.max(p) < 10000:
            convert_p = True
            p *= 100
        # ensure that Temperature is in C (scale from Kelvin if not)
        convert_t = False
        if np.max(t) > 100:
            convert_t = True
            t -= 273.15

        # determine ps
        ps = 610.79 * np.exp(t / (t + 238.3) * 17.2694)
        # determine w
        w = h / (1 - h)
        # determine ws
        ws = 621.97 * (ps / 1000.) / (p - (ps / 1000.))
        # determine RH
        rh = w / ws * 100.
        # check values
        rh[rh > 100] = 100
        rh[rh < 2] = 2

        # ensure that pressure is reconverted to mbar
        if convert_p:
            p /= 100
        # ensure that temeprature is reconverted to Kelvin
        if convert_t:
            t += 273.15

        return rh

    @staticmethod
    def dew_point(t, h, p):
        """Calculate the dew point.

        Parameters
        ----------
        t : np.ndarray
            Temperature in Celsius
        h : np.ndarray
            Specific humidity in kg/kg
        p : np.ndarray
            Pressure in Pa

        Returns
        -------
        dp : np.ndarray
            Dew point in Celsius.
        """

        # ensure that Temperature is in C (scale from Kelvin if not)
        convert_t = False
        if np.max(t) > 100:
            convert_t = True
            t -= 273.15

        rh = Ancillary.relative_humidity(t, h, p)
        dp = (243.04 * (np.log(rh / 100.) + (17.625 * t / (243.04 + t))) /
              (17.625 - np.log(rh / 100.) - ((17.625 * t) / (243.04 + t))))

        # ensure that temeprature is reconverted to Kelvin
        if convert_t:
            t += 273.15

        return dp

    def _calculate(self, var):
        """Method for calculating variables (without dependencies).

        Parameters
        ----------
        var : str
            NSRDB var name.

        Returns
        -------
        data : np.ndarray
            NSRDB-resolution data for the given var and the current day.
        """

        if var == 'solar_zenith_angle':
            lat_lon = self.nsrdb_grid[['latitude', 'longitude']].values
            lat_lon = lat_lon.astype(np.float32)
            data = SolarPosition(self.nsrdb_ti, lat_lon).zenith

        else:
            raise KeyError('Did not recognize request to derive variable '
                           '"{}".'.format(var))

        # convert units from MERRA to NSRDB
        data = self.convert_units(var, data)

        return data

    def _derive(self, var):
        """Method for deriving variables (with dependencies).

        Parameters
        ----------
        var : str
            NSRDB var name.

        Returns
        -------
        data : np.ndarray
            NSRDB-resolution data for the given var and the current day.
        """

        if var in ('relative_humidity', 'dew_point'):
            dependencies = ('air_temperature', 'specific_humidity',
                            'surface_pressure')
            # ensure that all dependencies have been processed
            for dep in dependencies:
                if dep not in self._processed:
                    logger.info('Processing dependency "{}" in order to '
                                'derive "{}".'.format(dep, var))
                    self[dep] = self._process(dep)

            # calculate merra-derived vars
            if var == 'relative_humidity':
                data = self.relative_humidity(self['air_temperature'],
                                              self['specific_humidity'],
                                              self['surface_pressure'])

            elif var == 'dew_point':
                data = self.dew_point(self['air_temperature'],
                                      self['specific_humidity'],
                                      self['surface_pressure'])

        else:
            raise KeyError('Did not recognize request to derive variable '
                           '"{}".'.format(var))

        # convert units from MERRA to NSRDB
        data = self.convert_units(var, data)

        return data

    def _process(self, var):
        """Method for processing interpolated variables.

        Parameters
        ----------
        var : str
            NSRDB var name.

        Returns
        -------
        data : np.ndarray
            NSRDB-resolution data for the given var and the current day.
        """

        kwargs = {'var_meta': self.var_meta, 'name': var,
                  'source_dir': self._source_dir, 'date': self.date}
        var_obj = self._var_factory.get(var, **kwargs)

        # get MERRA source data
        data = var_obj.source_data

        # get mapping from MERRA to NSRDB
        dist, ind = self.get_nn_ind(var_obj.grid, self.nsrdb_grid,
                                    var_obj.spatial_method)

        # perform weighting if applicable
        if var in self.WEIGHTS:
            weights = self.get_weights(var_obj)
            if weights is not None:
                logger.debug('Applying weights to "{}".'.format(var))
                data *= weights

        # run spatial interpolation
        logger.debug('Performing spatial interpolation on "{}" '
                     'with shape {}'
                     .format(var, data.shape))
        data = spatial_interp(var, data, var_obj.grid, self.nsrdb_grid,
                              var_obj.spatial_method, dist, ind,
                              var_obj.elevation_correct)

        # run temporal interpolation
        if var_obj.temporal_method == 'linear':
            logger.debug('Performing linear temporal interpolation on '
                         '"{}" with shape {}'.format(var, data.shape))
            data = temporal_lin(data, var_obj.time_index,
                                self.nsrdb_ti)

        elif var_obj.temporal_method == 'nearest':
            logger.debug('Performing stepwise temporal interpolation on '
                         '"{}" with shape {}'.format(var, data.shape))
            data = temporal_step(data, var_obj.time_index,
                                 self.nsrdb_ti)

        # convert units from MERRA to NSRDB
        data = self.convert_units(var, data)

        return data

    @staticmethod
    def _parallel(var_list, var_meta, date, source_dir, nsrdb_grid,
                  nsrdb_freq='5min'):
        """Process ancillary variables in parallel.

        Parameters
        ----------
        var_list : list | tuple
            List of variables to process in parallel
        var_meta : str
            CSV file containing meta data for all NSRDB variables.
        date : datetime.date
            Single day to extract MERRA2 data for.
        source_dir : str | NoneType
            Directory path containing ancillary source files. None will use
            default ancillary variables source directories.
        nsrdb_grid : str
            CSV file containing the NSRDB reference grid to interpolate to.
        nsrdb_freq : str
            Final desired NSRDB temporal frequency.

        Returns
        -------
        futures : dict
            Gathered futures, namespace of nsrdb data numpy arrays keyed by
            nsrdb variable name.
        """

        logger.info('Processing all MERRA variables in parallel.')
        # start a local cluster
        n_workers = int(np.min((len(var_list), os.cpu_count())))
        cluster = LocalCluster(n_workers=n_workers,
                               threads_per_worker=1,
                               memory_limit=0)
        futures = {}
        with Client(cluster) as client:

            # initialize loggers on workers
            loggers = (__name__,)
            for logger_name in loggers:
                client.run(NSRDB_LOGGERS.init_logger, logger_name)

            # submit a future for each merra variable (non-calculated)
            for var in var_list:
                futures[var] = client.submit(
                    Ancillary.process_single, var, var_meta, date, source_dir,
                    nsrdb_grid, nsrdb_freq=nsrdb_freq)

            # watch memory during futures to get max memory usage
            logger.debug('Waiting on parallel futures...')
            max_mem = 0
            status = 0
            while status == 0:
                mem = psutil.virtual_memory()
                max_mem = np.max((mem.used / 1e9, max_mem))
                time.sleep(5)
                for var, future in futures.items():
                    if future.status != 'pending':
                        status = 1
                        break

            logger.info('Futures finishing up, maximum memory usage was '
                        '{0:.3f} GB out of {1:.3f} GB total.'
                        .format(max_mem, mem.total / 1e9))

            # gather results
            futures = client.gather(futures)

            # send to merra object
            for var in futures.keys():
                # data returned from futures as read only for some reason
                futures[var].setflags(write=True)

        return futures

    @classmethod
    def process_single(cls, var, var_meta, date, source_dir, nsrdb_grid,
                       nsrdb_freq='5min'):
        """Process ancillary data for one variable for a single day.

        Parameters
        ----------
        var : str
            NSRDB var name.
        var_meta : str
            CSV file containing meta data for all NSRDB variables.
        date : datetime.date
            Single day to extract ancillary data for.
        source_dir : str | NoneType
            Directory path containing ancillary source files. None will use
            default ancillary variables source directories.
        nsrdb_grid : str
            CSV file containing the NSRDB reference grid to interpolate to.
        nsrdb_freq : str
            Final desired NSRDB temporal frequency.

        Returns
        -------
        data : np.ndarray
            NSRDB-resolution data for the given var and the current day.
        """

        logger.info('Processing data for "{}".'.format(var))

        adp = cls(var_meta, date, source_dir, nsrdb_grid,
                  nsrdb_freq=nsrdb_freq)

        if var in adp.DERIVED_VARS:
            data = adp._derive(var)
        elif var in adp.CALCULATED_VARS:
            data = adp._calculate(var)
        else:
            data = adp._process(var)

        if data.shape != adp.nsrdb_data_shape:
            raise ValueError('Expected NSRDB data shape of {}, but received '
                             'shape {} for "{}"'
                             .format(adp.nsrdb_data_shape,
                                     data.shape, var))

        logger.info('Finished "{}".'.format(var))

        return data

    @classmethod
    def process_multiple(cls, var_list, var_meta, date, source_dir, nsrdb_grid,
                         nsrdb_freq='5min', parallel=False):
        """Process ancillary data for multiple variables for a single day.

        Parameters
        ----------
        var_list : list | None
            List of variables to process
        var_meta : str
            CSV file containing meta data for all NSRDB variables.
        date : datetime.date
            Single day to extract ancillary data for.
        source_dir : str | NoneType
            Directory path containing ancillary source files. None will use
            default ancillary variables source directories.
        nsrdb_grid : str
            CSV file containing the NSRDB reference grid to interpolate to.
        nsrdb_freq : str
            Final desired NSRDB temporal frequency.
        parallel : bool
            Flag to perform parallel processing with each variable on a
            seperate process.

        Returns
        -------
        data : dict
            Namespace of nsrdb data numpy arrays keyed by nsrdb variable name.
        """

        # Create an AncillaryDataProcessing object instance for storing data.
        adp = cls(var_meta, date, source_dir, nsrdb_grid,
                  nsrdb_freq=nsrdb_freq)

        # default multiple compute
        if var_list is None:
            var_list = cls.ALL_SKY_VARS

        # remove derived (dependent) variables from var_list to be processed
        # last (most efficient)
        derived = []
        remove = []
        for var in cls.DERIVED_VARS:
            if var in var_list:
                derived.append(var)
                remove.append(var_list.index(var))
        derived = tuple(derived)
        var_list = tuple([v for i, v in enumerate(var_list)
                          if i not in remove])

        # run in serial
        if parallel is False:
            data = {}
            for var in var_list:
                adp[var] = cls.process_single(
                    var, var_meta, date, source_dir, nsrdb_grid,
                    nsrdb_freq=nsrdb_freq)
        # run in parallel
        else:
            data = cls._parallel(
                var_list, var_meta, date, source_dir, nsrdb_grid,
                nsrdb_freq=nsrdb_freq)
            for k, v in data.items():
                adp[k] = v

        # process derived (dependent) variables last using the built
        # AncillaryDataProcessing object instance.
        for var in derived:
            adp[var] = adp._derive(var)

        logger.info('Ancillary data processing complete.')
        return adp._processed
