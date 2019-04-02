# -*- coding: utf-8 -*-
"""This data model handles NSRDB data sources, with the end goal of preparing
data inputs to the NSRDB all-sky irradiance models.

A primary function of this model is to perform spatial and temporal
interpolation of ancillary data. MERRA2 and Asymmetry data are both spatially
and temporally interpolated to the NSRDB reference grid. Cloud properties
originating from the GOES satellites and processed by the University of
Wisconsin are mapped to the NSRDB grid via nearest neighbors.

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
import os
import logging
import psutil
from dask.distributed import LocalCluster, Client
import time

from nsrdb import NSRDBDIR, DATADIR
from nsrdb.utilities.solar_position import SolarPosition
from nsrdb.utilities.loggers import NSRDB_LOGGERS
from nsrdb.utilities.interpolation import (spatial_interp, geo_nn,
                                           temporal_lin, temporal_step)
from nsrdb.data_model.variable_factory import VarFactory

logger = logging.getLogger(__name__)


class DataModel:
    """Datamodel for single-day ancillary data processing to NSRDB."""

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
                    'cloud_type',
                    'cld_opd_dcomp',
                    'cld_reff_dcomp',
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

    CLOUD_VARS = ('cloud_type',
                  'cld_opd_dcomp',
                  'cld_reff_dcomp',
                  'cld_press_acha',
                  )

    # calculated variables (no dependencies)
    CALCULATED_VARS = ('solar_zenith_angle',)

    # derived variables (no interp, requires: temp, spec. humidity, pressure)
    DERIVED_VARS = ('relative_humidity',
                    'dew_point',
                    )

    # all variables processed by this module
    ALL_VARS = tuple(set(ALL_SKY_VARS + MERRA_VARS + CALCULATED_VARS +
                         DERIVED_VARS + CLOUD_VARS))

    def __init__(self, var_meta, date, nsrdb_grid, nsrdb_freq='5min'):
        """
        Parameters
        ----------
        var_meta : str | pd.DataFrame
            CSV file or dataframe containing meta data for all NSRDB variables.
        date : datetime.date
            Single day to extract MERRA2 data for.
        nsrdb_grid : str
            CSV file containing the NSRDB reference grid to interpolate to.
        nsrdb_freq : str
            Final desired NSRDB temporal frequency.
        """

        logger.info('Processing MERRA data for {}'.format(date))

        self._var_meta = var_meta
        self._parse_nsrdb_grid(nsrdb_grid)
        self._date = date
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

        ti = pd.date_range('1-1-{y}'.format(y=self.date.year),
                           '1-1-{y}'.format(y=self.date.year + 1),
                           freq=self._nsrdb_freq)[:-1]
        mask = (ti.month == self.date.month) & (ti.day == self.date.day)
        ti = ti[mask]
        return ti

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

        rh = DataModel.relative_humidity(t, h, p)
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

        kwargs = {'var_meta': self._var_meta, 'name': var, 'date': self.date}
        var_obj = self._var_factory.get(var, **kwargs)

        # get MERRA source data
        data = var_obj.source_data

        # get mapping from source data grid to NSRDB
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
    def _parallel(var_list, var_meta, date, nsrdb_grid, nsrdb_freq='5min'):
        """Process ancillary variables in parallel.

        Parameters
        ----------
        var_list : list | tuple
            List of variables to process in parallel
        var_meta : str | pd.DataFrame
            CSV file or dataframe containing meta data for all NSRDB variables.
        date : datetime.date
            Single day to extract MERRA2 data for.
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
                    DataModel.process_single, var, var_meta, date, nsrdb_grid,
                    nsrdb_freq=nsrdb_freq)

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
    def process_single(cls, var, var_meta, date, nsrdb_grid,
                       nsrdb_freq='5min'):
        """Process ancillary data for one variable for a single day.

        Parameters
        ----------
        var : str
            NSRDB var name.
        var_meta : str | pd.DataFrame
            CSV file or dataframe containing meta data for all NSRDB variables.
        date : datetime.date
            Single day to extract ancillary data for.
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

        adp = cls(var_meta, date, nsrdb_grid, nsrdb_freq=nsrdb_freq)

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
    def process_multiple(cls, var_list, var_meta, date, nsrdb_grid,
                         nsrdb_freq='5min', parallel=False):
        """Process ancillary data for multiple variables for a single day.

        Parameters
        ----------
        var_list : list | None
            List of variables to process
        var_meta : str | pd.DataFrame
            CSV file or dataframe containing meta data for all NSRDB variables.
        date : datetime.date
            Single day to extract ancillary data for.
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
        adp = cls(var_meta, date, nsrdb_grid, nsrdb_freq=nsrdb_freq)

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
                    var, var_meta, date, nsrdb_grid, nsrdb_freq=nsrdb_freq)
        # run in parallel
        else:
            data = cls._parallel(
                var_list, var_meta, date, nsrdb_grid, nsrdb_freq=nsrdb_freq)
            for k, v in data.items():
                adp[k] = v

        # process derived (dependent) variables last using the built
        # AncillaryDataProcessing object instance.
        for var in derived:
            adp[var] = adp._derive(var)

        logger.info('Ancillary data processing complete.')
        return adp._processed
