# -*- coding: utf-8 -*-
"""This data model handles NSRDB data sources, with the end goal of preparing
data inputs to the NSRDB all-sky irradiance models.

A primary function of this model is to perform spatial and temporal
interpolation of ancillary data. MERRA2 and Asymmetry data are both spatially
and temporally interpolated to the NSRDB reference grid. Cloud properties
originating from the GOES satellites and processed by the University of
Wisconsin are mapped to the NSRDB grid via nearest neighbors (ReGrid).

Non-standard module dependencies (pip-installable):
    - netCDF4

The following variables can be processed using this module. For details see the
class variables in Ancillary() below.
    ('alpha',
     'asymmetry',
     'aod',
     'cloud_type',
     'cld_opd_dcomp',
     'cld_reff_dcomp',
     'cld_press_acha',
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
from concurrent.futures import ProcessPoolExecutor
import time
from scipy.spatial import cKDTree

from nsrdb import DATADIR
from nsrdb.utilities.solar_position import SolarPosition
from nsrdb.utilities.interpolation import (spatial_interp, geo_nn,
                                           temporal_lin, temporal_step,
                                           parse_method)
from nsrdb.data_model.variable_factory import VarFactory


logger = logging.getLogger(__name__)


class DataModel:
    """Datamodel for single-day ancillary data processing to NSRDB."""

    # directory to cache intermediate data (nearest neighbor results)
    CACHE_DIR = '/projects/pxs/reference_grids/_nn_query_cache'

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
                    'surface_albedo',
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

    def __init__(self, var_meta, date, nsrdb_grid, nsrdb_freq='5min',
                 scale=True):
        """
        Parameters
        ----------
        var_meta : str | pd.DataFrame
            CSV file or dataframe containing meta data for all NSRDB variables.
        date : datetime.date
            Single day to extract MERRA2 data for.
        nsrdb_grid : str | pd.DataFrame
            CSV file containing the NSRDB reference grid to interpolate to,
            or a pre-extracted (and reduced) dataframe. The first csv column
            must be the NSRDB site gid's.
        nsrdb_freq : str
            Final desired NSRDB temporal frequency.
        scale : bool
            Flag to scale source data to reduced (integer) precision after
            data model processing.
        """

        self._var_meta = var_meta
        self._nsrdb_grid_file = None
        self._parse_nsrdb_grid(nsrdb_grid)
        self._date = date
        self._nsrdb_freq = nsrdb_freq
        self._scale = scale
        self._var_factory = VarFactory()
        self._processed = {}
        self._ti = None

    def __getitem__(self, key):
        return self._processed[key]

    def __setitem__(self, key, value):
        if isinstance(value, (np.ndarray, pd.DatetimeIndex)):
            self._processed[key] = value
        elif isinstance(value, dict):
            self._processed.update(value)
        else:
            raise TypeError('Did not recognize dtype sent to DataModel '
                            'processed dictionary: {}'.format(type(value)))

    def _parse_nsrdb_grid(self, inp):
        """Set the NSRDB reference grid from a csv file.

        Parameters
        ----------
        inp : str
            CSV file containing the NSRDB reference grid to interpolate to.
            The first column must be the NSRDB site gid's.
        """

        if isinstance(inp, pd.DataFrame):
            self._nsrdb_grid = inp
        elif inp.endswith('.csv'):
            self._nsrdb_grid_file = inp
            self._nsrdb_grid = pd.read_csv(inp, index_col=0)
        else:
            raise TypeError('Expected csv grid file or DataFrame but '
                            'received: {}'.format(inp))

        # copy index to gid data column that will be saved in output files
        # used in the case that the grid is chunked into subsets of sites
        self._nsrdb_grid['gid'] = self._nsrdb_grid.index

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
        if self._ti is None:
            self._ti = pd.date_range('1-1-{y}'.format(y=self.date.year),
                                     '1-1-{y}'.format(y=self.date.year + 1),
                                     freq=self._nsrdb_freq)[:-1]
            mask = ((self._ti.month == self.date.month) &
                    (self._ti.day == self.date.day))
            self._ti = self._ti[mask]
            self['time_index'] = self._ti
        return self._ti

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
    def processed_data(self):
        """Get the processed data dictionary.

        Returns
        -------
        _processed : dict
            Namespace of processed data set with __setitem__. Keys should be
            NSRDB variable names.
        """
        return self._processed

    def get_geo_nn(self, df1, df2, method, labels=('latitude', 'longitude'),
                   cache=False):
        """Get the geographic nearest neighbor distances (km) and indices.

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
            instead of performing NN query. Strings are evaluated as the csv
            file name to cache.

        Returns
        -------
        dist : ndarray
            Distance array in km returned if return_dist input arg set to True.
        indicies : ndarray
            1D array of row indicies in df1 that match df2.
            df1[df1.index[indicies[i]]] is closest to df2[df2.index[i]]
        """
        if 'NN' in method.upper():
            # always get 1 nearest neighbor for NN data copy
            k = 1
        elif 'IDW' in method.upper():
            # always get 4 nearest neighbors for dist interp method
            k = 4
        elif 'AGG' in method.upper():
            # aggregation can be from any number of neighbors, default to 4
            k = parse_method(method)
            if k is None:
                k = 4
        else:
            raise ValueError('Did not recognize spatial interp method: "{}"'
                             .format(method))

        # Do not cache results if the intended Cache directory isn't available
        if not os.path.exists(self.CACHE_DIR):
            cache = False

        if isinstance(cache, str):

            if not cache.endswith('.csv'):
                # cache file must be csv
                cache += '.csv'

            if self._nsrdb_grid_file is not None:
                # make sure cache file is nsrdb-grid-specific
                cache = cache.replace(
                    '.csv', '_' + os.path.basename(self._nsrdb_grid_file))

            # try to get cached kdtree results. fast for prototyping.
            cache_d = os.path.join(self.CACHE_DIR,
                                   cache.replace('.csv', '_d.csv'))
            cache_i = os.path.join(self.CACHE_DIR,
                                   cache.replace('.csv', '_i.csv'))

            if os.path.exists(cache_i) and os.path.exists(cache_d):
                logger.debug('Found cached nearest neighbor indices, '
                             'importing: {}'.format(cache_i))
                dist = np.genfromtxt(cache_d, dtype=np.float32, delimiter=',')
                ind = np.genfromtxt(cache_i, dtype=np.uint32, delimiter=',')

            else:
                dist, ind = geo_nn(df1, df2, labels=labels, k=k)
                logger.debug('Saving nearest neighbor indices to: {}'
                             .format(cache_i))
                np.savetxt(cache_d, dist, delimiter=',')
                np.savetxt(cache_i, ind, delimiter=',')

        else:
            dist, ind = geo_nn(df1, df2, labels=labels, k=k)

        return dist, ind

    @staticmethod
    def get_cloud_nn(fpath, nsrdb_grid, labels=('latitude', 'longitude')):
        """Nearest neighbors computation for cloud data regrid.

        Parameters
        ----------
        fpath : str
            Full filepath to a single UW cloud data file.
        nsrdb_grid : pd.DataFrame
            Reference grid data for NSRDB.
        labels : list | tuple
            lat/lon column lables for the NSRDB grid and the cloud grid

        Returns
        -------
        index : np.ndarray
            KDTree query results mapping cloud data to the NSRDB grid. e.g.
            nsrdb_data = cloud_data[index]
        """

        if isinstance(labels, tuple):
            labels = list(labels)
        # Build NN tree based on the unique cloud grid at single timestep
        tree = cKDTree(VarFactory.get_cloud_handler(fpath).grid[labels])
        # Get the index of NN to NSRDB grid
        _, index = tree.query(nsrdb_grid[labels], k=1)
        index = index.astype(np.uint32)
        return index

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
            _, i_nn = self.get_geo_nn(weights, var_obj.grid, 'NN')

            df_w = weights.iloc[i_nn.ravel()]
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

    def scale_data(self, var, data):
        """Perform safe scaling and datatype conversion of data.

        Parameters
        ----------
        var : str
            NSRDB variable name.
        data : np.ndarray
            Data array to scale.

        Returns
        -------
        data : np.ndarray
            Scaled data array with final dtype.
        """

        if self._scale:
            var_obj = self._var_factory.get_base_handler(
                self._var_meta, var, self.date)
            data = var_obj.scale_data(data)

        return data

    def unscale_data(self, var, data):
        """Perform safe un-scaling and datatype conversion of data.

        Parameters
        ----------
        var : str
            NSRDB variable name.
        data : np.ndarray
            Scaled data array to unscale.

        Returns
        -------
        data : np.ndarray
            Unscaled float32 data array.
        """

        if self._scale:
            var_obj = self._var_factory.get_base_handler(
                self._var_meta, var, self.date)
            data = var_obj.unscale_data(data)

        return data

    def run_pre_flight(self, var_list, cloud_extent='east', cloud_path=None):
        """Run pre-flight checks, raise if specified paths/files are not found.

        Parameters
        ----------
        var_list : list
            List of variable names
        cloud_extent : str
            Cloud data extent ('east' or 'west') for cloud variables in
            var_list.
        cloud_path : str | NoneType
            Optional path string to force a cloud data directory. If this is
            None, the file path will be infered from the extent, year, and day
            of year.
        """

        missing_list = []
        for var in var_list:
            kwargs = {'var_meta': self._var_meta, 'name': var,
                      'date': self.date}
            if 'cld' in var or 'cloud' in var:
                kwargs['extent'] = cloud_extent
                kwargs['path'] = cloud_path
                kwargs['dsets'] = [var]

            if var in self._var_factory.MAPPING:
                var_obj = self._var_factory.get(var, **kwargs)

                if hasattr(var_obj, 'pre_flight'):
                    missing = var_obj.pre_flight()
                else:
                    missing = ''

                if missing:
                    missing_list.append(missing)

        if missing_list:
            e = ('The data model pre-flight checks could not find '
                 'some required directories and/or files. '
                 'The following are missing: {}'.format(missing_list))
            logger.exception(e)
            raise IOError(e)

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

        elif var == 'surface_pressure':
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
            raise KeyError('Did not recognize request to calculate variable '
                           '"{}".'.format(var))

        # convert units from MERRA to NSRDB
        data = self.convert_units(var, data)

        # scale if requested
        data = self.scale_data(var, data)

        return data

    def _cloud_regrid(self, cloud_vars, extent='east', path=None,
                      parallel=False):
        """ReGrid data for multiple cloud variables to the NSRDB grid.

        (most efficient to process all cloud variables together to minimize
        number of kdtrees during regrid)

        Parameters
        ----------
        cloud_vars : tuple | list
            Source datasets to extract. It is more efficient to extract all
            required datasets at once from each cloud file, so that only one
            kdtree is built for each unique coordinate set in each cloud file.
        extent : str
            Regional (satellite) extent to process, used to form file paths.
        path : str | NoneType
            Optional path string to force a cloud data directory. If this is
            None, the file path will be infered from the extent, year, and day
            of year.
        parallel : bool
            Flag to perform regrid in parallel.

        Returns
        -------
        data : dict
            Data dictionary of cloud datasets mapped to the NSRDB grid. Keys
            are the cloud variables names, values are 2D numpy arrays.
            Array shape is (n_time, n_sites).
        """

        for var in cloud_vars:
            if var not in self.CLOUD_VARS:
                raise KeyError('Did not recognize request to process cloud '
                               'variable "{}".'.format(var))

        kwargs = {'var_meta': self._var_meta, 'name': cloud_vars[0],
                  'date': self.date, 'extent': extent, 'path': path,
                  'dsets': cloud_vars}

        # use the first cloud var name to get object,
        # full cloud_var list is passed in kwargs
        var_obj = self._var_factory.get(cloud_vars[0], **kwargs)

        logger.debug('Starting cloud data ReGrid for {} cloud timesteps.'
                     .format(len(var_obj)))

        if parallel:
            regrid_ind = self._cloud_regrid_parallel(var_obj.flist)
        else:
            regrid_ind = {}
            # make the nearest neighbors regrid index mapping for all timesteps
            for i, fpath in enumerate(var_obj.flist):
                logger.debug('Calculating ReGrid nearest neighbors for: {}'
                             .format(fpath))
                regrid_ind[i] = self.get_cloud_nn(fpath, self.nsrdb_grid)

        logger.debug('Finished processing ReGrid nearest neighbors. Starting '
                     'to extract and map cloud data to the NSRDB grid.')

        data = {}
        # extract the regrided data for all timesteps
        for i, obj in enumerate(var_obj):
            # save all datasets
            for dset, array in obj.source_data.items():
                if dset not in data:
                    # initialize array based on time index and NN index result
                    if np.issubdtype(array.dtype, np.float):
                        data[dset] = np.full(self.nsrdb_data_shape, np.nan,
                                             dtype=array.dtype)
                    else:
                        data[dset] = np.full(self.nsrdb_data_shape, -15,
                                             dtype=array.dtype)

                # write single timestep with NSRDB sites to appropriate row
                # map the regridded data using the regrid NN indices
                data[dset][i, :] = array[regrid_ind[i]]

        # scale if requested
        if self._scale:
            for var, arr in data.items():
                data[var] = self.scale_data(var, arr)

        return data

    def _cloud_regrid_parallel(self, flist):
        """Perform the ReGrid nearest neighbor calculations in parallel.

        Parameters
        ----------
        flist : list
            List of full file paths to cloud data files. Grid data from each
            file is mapped to the NSRDB grid in parallel.

        Returns
        -------
        regrid_ind : dict
            Dictionary of NN index results keyed by the enumerated file list.
        """

        logger.debug('Starting cloud ReGrid parallel.')

        # start a local cluster
        max_workers = int(np.min((len(flist), os.cpu_count())))
        logger.debug('Starting local cluster with {} workers.'
                     .format(max_workers))
        regrid_ind = {}
        with ProcessPoolExecutor(max_workers=max_workers) as exe:
            # make the nearest neighbors regrid index mapping for all timesteps
            for i, fpath in enumerate(flist):
                regrid_ind[i] = exe.submit(self.get_cloud_nn, fpath,
                                           self.nsrdb_grid)

            # watch memory during futures to get max memory usage
            logger.debug('Waiting on parallel futures...')
            max_mem = 0
            running = len(regrid_ind)
            while running > 0:
                mem = psutil.virtual_memory()
                max_mem = np.max((mem.used / 1e9, max_mem))
                time.sleep(5)
                running = 0
                complete = 0
                for future in regrid_ind.values():
                    if future.running():
                        running += 1
                    elif future.done():
                        complete += 1
                logger.debug('{} ReGrid futures are running, {} are complete.'
                             .format(running, complete))

            logger.info('Futures finished, maximum memory usage was '
                        '{0:.3f} GB out of {1:.3f} GB total.'
                        .format(max_mem, mem.total / 1e9))

            # gather results
            for k, v in regrid_ind.items():
                regrid_ind[k] = v.result()

        return regrid_ind

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
                    # process and save data to processed attribute
                    self[dep] = self._interpolate(dep)

                # unscale data to physical units for input to physical eqns
                self[dep] = self.unscale_data(dep, self[dep])

            # get the calculation method from the var factory
            method = self._var_factory.get(var)

            # calculate merra-derived vars
            data = method(self['air_temperature'],
                          self['specific_humidity'],
                          self['surface_pressure'])

        else:
            raise KeyError('Did not recognize request to derive variable '
                           '"{}".'.format(var))

        # convert units from MERRA to NSRDB
        data = self.convert_units(var, data)

        # scale if requested
        data = self.scale_data(var, data)

        # re-scale dependencies
        # (they had to be previously unscaled for physical eqns)
        for dep in dependencies:
            self[dep] = self.scale_data(dep, self[dep])

        return data

    def _interpolate(self, var):
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

        if 'albedo' in var:
            # special exclusions for large-extent albedo
            var_obj.exclusions_from_nsrdb(self.nsrdb_grid)

        # get ancillary data source data array
        data = var_obj.source_data

        # get mapping from source data grid to NSRDB
        dist, ind = self.get_geo_nn(var_obj.grid, self.nsrdb_grid,
                                    var_obj.spatial_method,
                                    cache=var_obj.cache_file)

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

        # scale if requested
        data = self.scale_data(var, data)

        return data

    @classmethod
    def _process_multiple(cls, var_list, var_meta, date,
                          nsrdb_grid, nsrdb_freq='5min', parallel=False,
                          cloud_extent='east', cloud_path=None):
        """Process ancillary data for multiple variables for a single day.

        Parameters
        ----------
        var_list : list | None
            List of variables to process
        var_meta : str | pd.DataFrame
            CSV file or dataframe containing meta data for all NSRDB variables.
        date : datetime.date
            Single day to extract ancillary data for.
        nsrdb_grid : str | pd.DataFrame
            CSV file containing the NSRDB reference grid to interpolate to,
            or a pre-extracted (and reduced) dataframe.
        nsrdb_freq : str
            Final desired NSRDB temporal frequency.
        parallel : bool
            Flag to perform parallel processing with each variable on a
            seperate process.
        cloud_extent : str
            Regional (satellite) extent to process for cloud data processing,
            used to form file paths to cloud data files.
        cloud_path : str | NoneType
            Optional path string to force a cloud data directory. If this is
            None, the file path will be infered from the extent, year, and day
            of year.
        return_obj : bool
            Flag to return full DataModel object instead of just the processed
            data dictionary.

        Returns
        -------
        out : DataModel
            Full DataModel object with the data in the .processed property.
        """

        data_model = cls(var_meta, date, nsrdb_grid, nsrdb_freq=nsrdb_freq)

        # run pre-flight checks
        data_model.run_pre_flight(var_list, cloud_extent=cloud_extent,
                                  cloud_path=cloud_path)

        # default multiple compute
        if var_list is None:
            var_list = cls.ALL_SKY_VARS

        # remove cloud variables from var_list to be processed together
        # (most efficient to process all cloud variables together to minimize
        # number of kdtrees during regrid)
        cloud_vars = []
        remove = []
        for var in cls.CLOUD_VARS:
            if var in var_list:
                cloud_vars.append(var)
                remove.append(var_list.index(var))
        cloud_vars = tuple(cloud_vars)
        var_list = tuple([v for i, v in enumerate(var_list)
                          if i not in remove])

        # remove derived (dependent) variables from var_list to be processed
        # last (most efficient to process depdencies first, dependents last)
        derived_vars = []
        remove = []
        for var in cls.DERIVED_VARS:
            if var in var_list:
                derived_vars.append(var)
                remove.append(var_list.index(var))
        derived_vars = tuple(derived_vars)
        var_list = tuple([v for i, v in enumerate(var_list)
                          if i not in remove])

        logger.info('First processing data for variable list: {}'
                    .format(var_list))
        logger.info('Then processing cloud data for variable list: {}'
                    .format(cloud_vars))
        logger.info('Finally, processing derived data for variable list: {}'
                    .format(derived_vars))

        if var_list:
            # run in serial
            if parallel is False:
                data = {}
                for var in var_list:
                    data_model[var] = cls.run_single(
                        var, var_meta, date, nsrdb_grid,
                        nsrdb_freq=nsrdb_freq)
            # run in parallel
            else:
                data = cls._process_parallel(
                    var_list, var_meta, date, nsrdb_grid,
                    nsrdb_freq=nsrdb_freq)
                for k, v in data.items():
                    data_model[k] = v

        # process cloud variables together
        if cloud_vars:
            data_model['clouds'] = cls.run_clouds(
                cloud_vars, var_meta, date, nsrdb_grid, nsrdb_freq=nsrdb_freq,
                parallel=parallel, extent=cloud_extent, path=cloud_path)

        # process derived (dependent) variables last using the built
        # AncillaryDataProcessing object instance.
        if derived_vars:
            for var in derived_vars:
                data_model[var] = data_model._derive(var)

        # scale if requested
        for var, arr in data_model._processed.items():
            data_model[var] = data_model.scale_data(var, arr)

        return data_model

    @staticmethod
    def _process_parallel(var_list, var_meta, date, nsrdb_grid,
                          nsrdb_freq='5min'):
        """Process ancillary variables in parallel.

        Parameters
        ----------
        var_list : list | tuple
            List of variables to process in parallel
        var_meta : str | pd.DataFrame
            CSV file or dataframe containing meta data for all NSRDB variables.
        date : datetime.date
            Single day to extract MERRA2 data for.
        nsrdb_grid : str | pd.DataFrame
            CSV file containing the NSRDB reference grid to interpolate to,
            or a pre-extracted (and reduced) dataframe.
        nsrdb_freq : str
            Final desired NSRDB temporal frequency.

        Returns
        -------
        futures : dict
            Gathered futures, namespace of nsrdb data numpy arrays keyed by
            nsrdb variable name.
        """

        logger.info('Processing variables in parallel: {}'.format(var_list))
        # start a local cluster
        max_workers = int(np.min((len(var_list), os.cpu_count())))
        futures = {}
        logger.debug('Starting local cluster with {} workers.'
                     .format(max_workers))
        with ProcessPoolExecutor(max_workers=max_workers) as exe:
            # submit a future for each merra variable (non-calculated)
            for var in var_list:
                futures[var] = exe.submit(
                    DataModel.run_single, var, var_meta, date, nsrdb_grid,
                    nsrdb_freq=nsrdb_freq)

            # watch memory during futures to get max memory usage
            logger.debug('Waiting on parallel futures...')
            max_mem = 0
            running = len(futures)
            while running > 0:
                mem = psutil.virtual_memory()
                max_mem = np.max((mem.used / 1e9, max_mem))
                time.sleep(5)
                running = 0
                keys = []
                for key, future in futures.items():
                    if future.running():
                        running += 1
                        keys += [key]
                logger.debug('{} DataModel processing futures are running: {}'
                             .format(running, keys))

            logger.info('Futures finished, maximum memory usage was '
                        '{0:.3f} GB out of {1:.3f} GB total.'
                        .format(max_mem, mem.total / 1e9))

            # gather results
            for k, v in futures.items():
                futures[k] = v.result()

        return futures

    @classmethod
    def run_single(cls, var, var_meta, date, nsrdb_grid, nsrdb_freq='5min'):
        """Run ancillary data processing for one variable for a single day.

        Parameters
        ----------
        var : str
            NSRDB var name.
        var_meta : str | pd.DataFrame
            CSV file or dataframe containing meta data for all NSRDB variables.
        date : datetime.date
            Single day to extract ancillary data for.
        nsrdb_grid : str | pd.DataFrame
            CSV file containing the NSRDB reference grid to interpolate to,
            or a pre-extracted (and reduced) dataframe. The first csv column
            must be the NSRDB site gid's.
        nsrdb_freq : str
            Final desired NSRDB temporal frequency.

        Returns
        -------
        data : np.ndarray
            NSRDB-resolution data for the given var and the current day.
        """

        logger.info('Processing data for "{}".'.format(var))

        data_model = cls(var_meta, date, nsrdb_grid, nsrdb_freq=nsrdb_freq)

        if var in cls.CALCULATED_VARS:
            method = data_model._calculate
        elif var in cls.CLOUD_VARS:
            method = data_model._cloud_regrid
        elif var in cls.DERIVED_VARS:
            method = data_model._derive
        else:
            method = data_model._interpolate

        try:
            data = method(var)
        except Exception as e:
            logger.exception('Processing method "DataModel.{}()" failed for '
                             '"{}"'.format(method.__name__, var))
            raise e

        if data.shape != data_model.nsrdb_data_shape:
            raise ValueError('Expected NSRDB data shape of {}, but '
                             'received shape {} for "{}"'
                             .format(data_model.nsrdb_data_shape,
                                     data.shape, var))

        logger.info('Finished "{}".'.format(var))

        return data

    @classmethod
    def run_clouds(cls, cloud_vars, var_meta, date, nsrdb_grid,
                   nsrdb_freq='5min', extent='east', path=None,
                   parallel=False):
        """Run cloud processing for multiple cloud variables.

        (most efficient to process all cloud variables together to minimize
        number of kdtrees during regrid)

        Parameters
        ----------
        cloud_vars : list | tuple
            NSRDB cloud variables names.
        var_meta : str | pd.DataFrame
            CSV file or dataframe containing meta data for all NSRDB variables.
        date : datetime.date
            Single day to extract ancillary data for.
        nsrdb_grid : str | pd.DataFrame
            CSV file containing the NSRDB reference grid to interpolate to,
            or a pre-extracted (and reduced) dataframe. The first csv column
            must be the NSRDB site gid's.
        nsrdb_freq : str
            Final desired NSRDB temporal frequency.
        extent : str
            Regional (satellite) extent to process, used to form file paths.
        path : str | NoneType
            Optional path string to force a cloud data directory. If this is
            None, the file path will be infered from the extent, year, and day
            of year.
        parallel : bool
            Flag to perform regrid in parallel.

        Returns
        -------
        data : dict
            Namespace of nsrdb data numpy arrays keyed by nsrdb variable name.
        """

        logger.info('Processing data for multiple cloud variables: {}'
                    .format(cloud_vars))

        data_model = cls(var_meta, date, nsrdb_grid, nsrdb_freq=nsrdb_freq)

        try:
            data = data_model._cloud_regrid(cloud_vars, extent=extent,
                                            path=path, parallel=parallel)
        except Exception as e:
            logger.exception('Processing method "DataModel._cloud_regrid()" '
                             'failed for "{}"'.format(cloud_vars))
            raise e

        for k, v in data.items():
            if v.shape != data_model.nsrdb_data_shape:
                raise ValueError('Expected NSRDB data shape of {}, but '
                                 'received shape {} for "{}"'
                                 .format(data_model.nsrdb_data_shape,
                                         v.shape, k))

        logger.info('Finished "{}".'.format(cloud_vars))

        return data

    @classmethod
    def run_multiple(cls, var_list, var_meta, date, nsrdb_grid,
                     nsrdb_freq='5min', parallel=False, cloud_extent='east',
                     cloud_path=None, return_obj=False):
        """Run ancillary data processing for multiple variables for single day.

        Parameters
        ----------
        var_list : list | None
            List of variables to process
        var_meta : str | pd.DataFrame
            CSV file or dataframe containing meta data for all NSRDB variables.
        date : datetime.date
            Single day to extract ancillary data for.
        nsrdb_grid : str | pd.DataFrame
            CSV file containing the NSRDB reference grid to interpolate to,
            or a pre-extracted (and reduced) dataframe. The first csv column
            must be the NSRDB site gid's.
        nsrdb_freq : str
            Final desired NSRDB temporal frequency.
        parallel : bool
            Flag to perform parallel processing with each variable on a
            seperate process.
        cloud_extent : str
            Regional (satellite) extent to process for cloud data processing,
            used to form file paths to cloud data files.
        cloud_path : str | NoneType
            Optional path string to force a cloud data directory. If this is
            None, the file path will be infered from the extent, year, and day
            of year.
        return_obj : bool
            Flag to return full DataModel object instead of just the processed
            data dictionary.

        Returns
        -------
        out : dict | DataModel
            Either the dictionary of data or the full DataModel object with
            the data in the .processed property. Controlled by the return_obj
            flag.
        """

        logger.info('Building NSRDB data model for {} at a {} temporal '
                    'resolution.'.format(date, nsrdb_freq))

        if isinstance(nsrdb_grid, str):
            logger.info('Using the NSRDB reference grid file: {}'
                        .format(nsrdb_grid))
        elif isinstance(nsrdb_grid, pd.DataFrame):
            logger.info('Using the NSRDB reference grid dataframe with '
                        'shape, head, tail:\n{}\n{}\n{}'
                        .format(nsrdb_grid.shape, nsrdb_grid.head(),
                                nsrdb_grid.tail()))
        else:
            raise TypeError('Expected csv grid file or DataFrame but '
                            'received: {}'.format(nsrdb_grid))

        data_model = cls._process_multiple(
            var_list, var_meta, date, nsrdb_grid, nsrdb_freq=nsrdb_freq,
            parallel=parallel, cloud_extent=cloud_extent,
            cloud_path=cloud_path)

        # Create an AncillaryDataProcessing object instance for storing data.
        logger.info('Final NSRDB output shape is: {}'
                    .format(data_model.nsrdb_data_shape))

        logger.info('DataModel processing complete for: {}'.format(date))

        if return_obj:
            return data_model
        else:
            return data_model.processed_data
