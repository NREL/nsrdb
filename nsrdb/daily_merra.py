# -*- coding: utf-8 -*-
"""Performs spatial and temporal interpolation of ancillary data from MERRA2.

Non-standard module dependencies (pip-installable):
    - netCDF4

The following variables can be processed using this module:
    - PS (surface_pressure, Pa)
    - T2M (air_temperature, C)
    - TO3 (ozone, atm-cm)
    - TQV (total_precipitable_water, cm)
    - wind_speed (m/s)
    - wind_direction (degrees)
    - QV2M (specific_humidity, kg_water/kg_air)
    - relative_humidity (%)
    - dew_point (C)
    - TOTANGSTR (alpha, angstrom wavelength exponent, unitless)
    - TOTEXTTAU (aod, aerosol optical depth, unitless)
    - TOTSCATAU (ssa, aerosol single scatter albedo, unitless)
    - asymmetry (Aerosol asymmetry parameter, unitless)
    - surface_albedo (unitless)
"""

import numpy as np
import pandas as pd
import os
from netCDF4 import Dataset as NetCDF
import logging
import psutil
from dask.distributed import LocalCluster, Client
import time

from nsrdb import NSRDBDIR, DATADIR
from nsrdb.utilities.loggers import NSRDB_LOGGERS
from nsrdb.utilities.interpolation import (spatial_interp, geo_nn,
                                           temporal_lin, temporal_step)

logger = logging.getLogger(__name__)


class MerraVar:
    """Helper for MERRA variable properties and source data extraction."""

    MERRA_ELEV = os.path.join(DATADIR, 'merra_grid_srtm_500m_stats')

    def __init__(self, var_meta, var, merra_dir, date_stamp):
        """
        Parameters
        ----------
        var_meta : str
            CSV file containing meta data for all NSRDB variables.
        var : str
            NSRDB var name.
        merra_dir : str
            Directory path containing MERRA source files.
        date_stamp : str
            Date stamp that should be in the MERRA file, format is YYYYMMDD
        """
        self._var_meta = var_meta
        self._var = var
        self._merra_dir = merra_dir
        self._date_stamp = date_stamp

    @property
    def file(self):
        """Get the MERRA file path for the target NSRDB variable name.

        Returns
        -------
        fmerra : str
            MERRA file path containing the target NSRDB variable.
        """

        path = os.path.join(self._merra_dir, self.dset)
        flist = os.listdir(path)
        for f in flist:
            if self._date_stamp in f:
                fmerra = os.path.join(path, f)
                break
        return fmerra

    @property
    def mask(self):
        """Get a boolean mask to locate the current variable in the meta data.
        """
        if not hasattr(self, '_mask'):
            self._mask = self._var_meta['var'] == self._var
        return self._mask

    @property
    def name(self):
        """Get the MERRA variable name from the NSRDB variable name.

        Returns
        -------
        name : str
            MERRA var name.
        """
        return str(self._var_meta.loc[self.mask, 'merra_name'].values[0])

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

    @staticmethod
    def format_2d(data):
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
            if self.name in ['wind_speed', 'wind_direction']:
                u_vector = f['U2M'][:]
                v_vector = f['V2M'][:]
                if self.name == 'wind_speed':
                    data = np.sqrt(u_vector**2 + v_vector**2)
                else:
                    data = np.degrees(
                        np.arctan2(u_vector, v_vector)) + 180

            elif self.name == 'TOTSCATAU':
                # Single scatter albedo is total scatter / aod
                data = f[self.name][:] / f['TOTEXTTAU'][:]

            else:
                data = f[self.name][:]

        # make the data a flat 2d array
        data = self.format_2d(data)

        return data

    @property
    def merra_grid(self):
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


class MerraDay:
    """Framework for single-day MERRA data interpolation to NSRDB."""

    CACHE_DIR = NSRDBDIR

    WEIGHTS = {
        'aod': os.path.join(
            DATADIR, 'Monthly_pixel_correction_MERRA2_AOD.txt'),
        'alpha': os.path.join(
            DATADIR, 'Monthly_pixel_correction_MERRA2_Alpha.txt')}

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

    CALC_VARS = ('relative_humidity',
                 'dew_point',
                 )

    ALL_VARS = MERRA_VARS + CALC_VARS

    def __init__(self, var_meta, date, merra_dir, nsrdb_grid,
                 nsrdb_freq='5min'):
        """
        Parameters
        ----------
        var_meta : str
            CSV file containing meta data for all NSRDB variables.
        date : datetime.date
            Single day to extract MERRA2 data for.
        merra_dir : str
            Directory path containing MERRA source files.
        nsrdb_grid : str
            CSV file containing the NSRDB reference grid to interpolate to.
        nsrdb_freq : str
            Final desired NSRDB temporal frequency.
        """

        logger.info('Processing MERRA data for {}'.format(date))

        self.var_meta = var_meta
        self._date = date
        self._merra_dir = merra_dir
        self.nsrdb_grid = nsrdb_grid
        self._nsrdb_freq = nsrdb_freq

        logger.debug('Final NSRDB output shape is: {}'
                     .format(self.nsrdb_data_shape))

    def __getitem__(self, var):
        return self.nsrdb_data[var]

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
        """Get an arbitrary merra file path.

        Returns
        -------
        file : str
            MERRA file path.
        """

        if not self.var_dict:
            file = MerraVar(self.var_meta, 'aod', self._merra_dir,
                            self.date_stamp).file
        else:
            # take the first file from the variable dictionary
            file = self.var_dict[list(self.var_dict.keys())[0]].file

        return file

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

    @property
    def merra_ti(self):
        """Get the MERRA native time index.

        Returns
        -------
        nsrdb_ti : pd.DatetimeIndex
            Pandas datetime index for the current day at the MERRA2 resolution
            (1-hour).
        """
        return self.time_index(freq='1h')

    @property
    def nsrdb_grid(self):
        """Return the grid.

        Returns
        -------
        _nsrdb_grid : pd.DataFrame
            Reference grid data.
        """
        return self._nsrdb_grid

    @nsrdb_grid.setter
    def nsrdb_grid(self, inp):
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

    @property
    def nsrdb_ti(self):
        """Get the NSRDB target time index.

        Returns
        -------
        nsrdb_ti : pd.DatetimeIndex
            Pandas datetime index for the current day at the NSRDB resolution.
        """
        return self.time_index(freq=self._nsrdb_freq)

    @property
    def nsrdb_data(self):
        """Get the internal namespace of final NSRDB resolution data.

        Returns
        -------
        _nsrdb_data : dict
            Namespace of NSRDB data arrays keyed with the NSRDB variable names.
        """
        if not hasattr(self, '_nsrdb_data'):
            self._nsrdb_data = {}
        return self._nsrdb_data

    @nsrdb_data.setter
    def nsrdb_data(self, inp):
        """Set a final NSRDB resolution data array to namespace.

        Parameters
        -------
        inp : list | tuple
            Two-entry list/tuple with (var_name, data_array).
        """

        if not hasattr(self, '_nsrdb_data'):
            self._nsrdb_data = {}

        if not isinstance(inp[1], np.ndarray):
            raise TypeError('Expected numpy array but received {} for "{}"'
                            .format(type(inp[1]), inp[0]))

        if inp[1].shape != self.nsrdb_data_shape:
            raise ValueError('Expected NSRDB data shape of {}, but received '
                             'shape {} for "{}"'
                             .format(self.nsrdb_data_shape,
                                     inp[1].shape, inp[0]))

        self._nsrdb_data[inp[0]] = inp[1]

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

    def time_index(self, freq='1h'):
        """Get a pandas date time object for the current analysis day.

        Parameters
        ----------
        freq : str
            Pandas datetime frequency, e.g. '1h', '5min', etc...

        Returns
        -------
        ti : pd.DatetimeIndex
            Pandas datetime index for the current day.
        """

        ti = pd.date_range('1-1-{y}'.format(y=self._date.year),
                           '1-1-{y}'.format(y=self._date.year + 1),
                           freq=freq)[:-1]
        mask = (ti.month == self._date.month) & (ti.day == self._date.day)
        ti = ti[mask]
        return ti

    @property
    def var_dict(self):
        """Get the internal namespace of MERRA variable objects.

        Returns
        -------
        _var_dict : dict
            Namespace of MerraVar objects keyed with the NSRDB variable names.
        """
        if not hasattr(self, '_var_dict'):
            self._var_dict = {}
        return self._var_dict

    @property
    def var_meta(self):
        """Return the meta data for NSRDB variables.

        Returns
        -------
        _var_meta : pd.DataFrame
            Meta data for NSRDB variables.
        """
        return self._var_meta

    @var_meta.setter
    def var_meta(self, inp):
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

    def _get_weights(self, var):
        """Get the irradiance model weights for AOD/Alpha.

        Parameters
        ----------
        var : str
            NSRDB variable name

        Returns
        -------
        weights : np.ndarray | NoneType
            1D array of weighting values for the given var in the current
            month. Returns None if var does not require weighting.
        """

        if not hasattr(self, '_weights'):
            self._weights = {}

        if var in self.WEIGHTS and var not in self._weights:
            logger.debug('Extracting weights for "{}"'.format(var))
            weights = pd.read_csv(self.WEIGHTS[var], sep=' ', skiprows=4,
                                  skipinitialspace=1)
            weights = weights.rename(
                {'Lat.': 'latitude', 'Long.': 'longitude'}, axis='columns')

            # use geo nearest neighbors to find closest indices
            # between weights and MERRA grid
            _, i_nn = self.get_nn_ind(weights, self.var_dict[var].merra_grid,
                                      'NN')
            i_nn = i_nn.flatten()

            df_w = weights.iloc[i_nn.flatten()]
            df_w = df_w[df_w.columns[2:-1]].T.set_index(
                pd.date_range(str(self._date.year), freq='M', periods=12))
            df_w[df_w < 0] = 1

            self._weights[var] = df_w

        if var in self._weights:
            mask = (self._weights[var].index.month == self._date.month)
            weights = self._weights[var][mask].values[0]
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

        rh = MerraDay.relative_humidity(t, h, p)
        dp = (243.04 * (np.log(rh / 100.) + (17.625 * t / (243.04 + t))) /
              (17.625 - np.log(rh / 100.) - ((17.625 * t) / (243.04 + t))))

        # ensure that temeprature is reconverted to Kelvin
        if convert_t:
            t += 273.15

        return dp

    @staticmethod
    def log_mem():
        """Log the memory usage to debug."""
        mem = psutil.virtual_memory()
        logger.debug('{0:.3f} GB used of {1:.3f} GB total ({2:.1f}% used) '
                     '({3:.3f} GB free) ({4:.3f} GB available).'
                     ''.format(mem.used / 1e9,
                               mem.total / 1e9,
                               100 * mem.used / mem.total,
                               mem.free / 1e9,
                               mem.available / 1e9))

    def process_var(self, var):
        """Process a single variable and return the data.

        Parameters
        ----------
        var : str
            NSRDB var name.

        Returns
        -------
        data : np.ndarray
            NSRDB-resolution data for the given var and the current day.
        """

        logger.info('Processing MERRA data for "{}".'.format(var))

        # initialize MERRA variable instance
        self.var_dict[var] = MerraVar(self.var_meta, var, self._merra_dir,
                                      self.date_stamp)

        if var == 'relative_humidity':
            data = self.relative_humidity(self.nsrdb_data['air_temperature'],
                                          self.nsrdb_data['specific_humidity'],
                                          self.nsrdb_data['surface_pressure'])

        elif var == 'dew_point':
            data = self.dew_point(self.nsrdb_data['air_temperature'],
                                  self.nsrdb_data['specific_humidity'],
                                  self.nsrdb_data['surface_pressure'])

        else:
            # get MERRA source data
            data = self.var_dict[var].source_data
            # get mapping from MERRA to NSRDB
            dist, ind = self.get_nn_ind(self.var_dict[var].merra_grid,
                                        self.nsrdb_grid,
                                        self.var_dict[var].spatial_method)

            # perform weighting if applicable
            if var in self.WEIGHTS:
                weights = self._get_weights(var)
                if weights is not None:
                    logger.debug('Applying weights to "{}".'.format(var))
                    data *= weights

            # run spatial interpolation
            logger.debug('Performing spatial interpolation on "{}" '
                         'with shape {}'
                         .format(var, data.shape))
            data = spatial_interp(var, data, self.var_dict[var].merra_grid,
                                  self.nsrdb_grid,
                                  self.var_dict[var].spatial_method,
                                  dist, ind,
                                  self.var_dict[var].elevation_correct)

            # run temporal interpolation
            if self.var_dict[var].temporal_method == 'linear':
                logger.debug('Performing linear temporal interpolation on '
                             '"{}" with shape {}'.format(var, data.shape))
                data = temporal_lin(data, self.merra_ti, self.nsrdb_ti)

            elif self.var_dict[var].temporal_method == 'nearest':
                logger.debug('Performing stepwise temporal interpolation on '
                             '"{}" with shape {}'.format(var, data.shape))
                data = temporal_step(data, self.merra_ti, self.nsrdb_ti)

        # convert units from MERRA to NSRDB
        data = self.convert_units(var, data)

        logger.info('Finished "{}".'.format(var))

        return data

    @classmethod
    def run(cls, var_meta, date, merra_dir, nsrdb_grid, nsrdb_freq='5min',
            var_list=None, parallel=False):
        """Run MERRA2 processing for all variables for a single day.

        Parameters
        ----------
        var_meta : str
            CSV file containing meta data for all NSRDB variables.
        date : datetime.date
            Single day to extract MERRA2 data for.
        merra_dir : str
            Directory path containing MERRA source files.
        nsrdb_grid : str
            CSV file containing the NSRDB reference grid to interpolate to.
        nsrdb_freq : str
            Final desired NSRDB temporal frequency.
        var_list : list | None
            List of variables to process

        Returns
        -------
        merra : MerraDay
            Single day merra object. Processed MERRA data can be found in the
            nsrdb_data attribute (dict) and MerraVar objects can be found in
            the var_dict attribute (dict).
        """

        merra = cls(var_meta, date, merra_dir, nsrdb_grid,
                    nsrdb_freq=nsrdb_freq)

        if var_list is not None:
            for var in var_list:
                data = merra.process_var(var)
                merra.nsrdb_data = [var, data]

        elif var_list is None and parallel is False:
            for var in cls.ALL_VARS:
                data = merra.process_var(var)
                merra.nsrdb_data = [var, data]

        elif var_list is None and parallel is True:
            logger.info('Processing all MERRA variables in parallel.')
            # start a local cluster
            n_workers = int(np.min((len(cls.MERRA_VARS), os.cpu_count())))
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
                for var in cls.MERRA_VARS:
                    futures[var] = client.submit(merra.process_var, var)

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
                for var, data in futures.items():
                    # data returned from futures as read only for some reason
                    data.setflags(write=True)
                    merra.nsrdb_data = [var, data]

            # process calculated variables in serial
            for var in cls.CALC_VARS:
                data = merra.process_var(var)
                merra.nsrdb_data = [var, data]

        logger.info('MERRA data processing complete.')
        return merra
