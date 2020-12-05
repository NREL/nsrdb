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
from warnings import warn

from nsrdb import DATADIR
from nsrdb.utilities.interpolation import (spatial_interp, temporal_lin,
                                           temporal_step, parse_method)
from nsrdb.utilities.nearest_neighbor import geo_nn, knn
from nsrdb.data_model.variable_factory import VarFactory
from nsrdb.file_handlers.outputs import Outputs


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
                  'wind_direction',
                  'specific_humidity',
                  'alpha',
                  'aod',
                  'ssa',
                  )

    # cloud variables from UW/GOES
    CLOUD_VARS = ('cloud_type',
                  'cld_opd_dcomp',
                  'cld_reff_dcomp',
                  'cld_press_acha',
                  )

    MLCLOUDS_VARS = ('cloud_fraction',
                     'cloud_probability',
                     'temp_3_75um_nom',
                     'temp_11_0um_nom',
                     'temp_11_0um_nom_stddev_3x3',
                     'refl_0_65um_nom',
                     'refl_0_65um_nom_stddev_3x3',
                     'refl_3_75um_nom',
                     )

    # derived variables
    DERIVED_VARS = ('relative_humidity',
                    'dew_point',
                    'solar_zenith_angle',
                    )

    # dependencies for derived variables.
    DEPENDENCIES = {'relative_humidity': ('air_temperature',
                                          'specific_humidity',
                                          'surface_pressure'),
                    'dew_point': ('air_temperature',
                                  'specific_humidity',
                                  'surface_pressure'),
                    'solar_zenith_angle': ('air_temperature',
                                           'surface_pressure'),
                    }

    # all variables processed by this module
    ALL_VARS = tuple(set(ALL_SKY_VARS + MERRA_VARS + DERIVED_VARS
                         + CLOUD_VARS))

    # all variables processed by this module WITH mlclouds gap fill
    ALL_VARS_ML = tuple(set(ALL_SKY_VARS + MERRA_VARS + DERIVED_VARS
                            + CLOUD_VARS + MLCLOUDS_VARS))

    def __init__(self, date, nsrdb_grid, nsrdb_freq='5min', var_meta=None,
                 factory_kwargs=None, scale=True, max_workers=None):
        """
        Parameters
        ----------
        date : datetime.date
            Single day to extract MERRA2 data for.
        nsrdb_grid : str | pd.DataFrame
            CSV file containing the NSRDB reference grid to interpolate to,
            or a pre-extracted (and reduced) dataframe. The first csv column
            must be the NSRDB site gid's.
        nsrdb_freq : str
            Final desired NSRDB temporal frequency.
        var_meta : str | pd.DataFrame | None
            CSV file or dataframe containing meta data for all NSRDB variables.
            Defaults to the NSRDB var meta csv in git repo.
        factory_kwargs : dict | None
            Optional namespace of kwargs to use to initialize variable data
            handlers from the data model's variable factory. Keyed by
            variable name. Values can be "source_dir", "handler", etc...
            source_dir for cloud variables can be a normal directory
            path or /directory/prefix*suffix where /directory/ can have
            more sub dirs
        scale : bool
            Flag to scale source data to reduced (integer) precision after
            data model processing.
        max_workers : int | None
            Maximum number of workers to use in parallel. 1 runs serial,
            None uses all available workers.
        """

        self._nsrdb_data_shape = None
        self._nsrdb_grid_file = None
        self._parse_nsrdb_grid(nsrdb_grid)
        self._date = date
        self._nsrdb_freq = nsrdb_freq
        self._var_meta = var_meta
        if factory_kwargs is None:
            factory_kwargs = {}
        self._factory_kwargs = factory_kwargs
        self._scale = scale
        self._var_factory = VarFactory()
        self._processed = {}
        self._ti = None
        self._weights = {}
        self._max_workers = max_workers

    def __getitem__(self, key):
        return self._processed[key]

    def __setitem__(self, key, value):
        if isinstance(value, (np.ndarray, pd.DatetimeIndex, str)):
            self._processed[key] = value
        elif isinstance(value, dict):
            self._processed.update(value)
        else:
            raise TypeError('Did not recognize dtype sent to DataModel '
                            'processed dictionary: {}'.format(type(value)))

    def _parse_nsrdb_grid(self, inp, req=('latitude', 'longitude',
                                          'elevation')):
        """Set the NSRDB reference grid from a csv file.

        Parameters
        ----------
        inp : str
            CSV file containing the NSRDB reference grid to interpolate to.
            The first column must be the NSRDB site gid's.
        req : tuple | list
            Required column labels in nsrdb grid file.
        """

        if isinstance(inp, pd.DataFrame):
            self._nsrdb_grid = inp
        elif inp.endswith('.csv'):
            self._nsrdb_grid_file = inp
            self._nsrdb_grid = pd.read_csv(inp, index_col=0)
        else:
            raise TypeError('Expected csv grid file or DataFrame but '
                            'received: {}'.format(inp))

        # check requirements
        for r in req:
            if r not in self._nsrdb_grid:
                raise KeyError('Could not find "{}" in nsrdb grid labels: "{}"'
                               .format(r, self._nsrdb_grid.columns.values))

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
            mask = ((self._ti.month == self.date.month)
                    & (self._ti.day == self.date.day))
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
        if self._nsrdb_data_shape is None:
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

    def get_geo_nn(self, df1, df2, interp_method='NN', nn_method='haversine',
                   labels=('latitude', 'longitude'), cache=False):
        """Get the geographic nearest neighbor distances (km) and indices.

        Parameters
        ----------
        df1/df2 : pd.DataFrame:
            Dataframes containing coodinate columns with the corresponding
            labels.
        interp_method : str
            Spatial interpolation method - either NN or IDW
        nn_method : str | None
            NSRDB nearest_neighbor tree search method, either
            "haversine" or "kdtree". None defaults to geo_nn.
        labels : tuple | list
            Column labels corresponding to the lat/lon columns in df1/df2.
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

        if nn_method.lower() == 'haversine':
            nn_method = geo_nn
        elif nn_method.lower() == 'kdtree':
            nn_method = knn
        else:
            e = 'Did not recognize nn_method "{}"'.format(nn_method)
            logger.error(e)
            raise ValueError(e)

        if 'NN' in interp_method.upper():
            # always get 1 nearest neighbor for NN data copy
            k = 1
        elif 'IDW' in interp_method.upper():
            # always get 4 nearest neighbors for dist interp interp_method
            k = 4
        elif 'AGG' in interp_method.upper():
            # aggregation can be from any number of neighbors, default to 4
            k = parse_method(interp_method)
            if k is None:
                k = 4
        else:
            e = ('Did not recognize spatial interp_method: "{}"'
                 .format(interp_method))
            logger.error(e)
            raise ValueError(e)

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
                logger.warning('Found cached nearest neighbor indices, '
                               'importing: {}'.format(cache_i))
                dist = np.genfromtxt(cache_d, dtype=np.float32, delimiter=',')
                ind = np.genfromtxt(cache_i, dtype=np.uint32, delimiter=',')

            else:
                dist, ind = nn_method(df1, df2, labels=labels, k=k)
                logger.info('Saving nearest neighbor indices to: {}'
                            .format(cache_i))
                np.savetxt(cache_d, dist, delimiter=',')
                np.savetxt(cache_i, ind, delimiter=',')

        else:
            dist, ind = nn_method(df1, df2, labels=labels, k=k)

        if (dist.shape[0] != len(df2) or ind.shape[0] != len(df2)
                or dist.shape[1] != k or ind.shape[1] != k):
            e = ('NSRDB DataModel.get_geo_nn() method returned dist of '
                 'shape {} and ind of shape {} while the query dataframe '
                 'is of shape {} and k is {}. Maybe check the cached NN file.'
                 .format(dist.shape, ind.shape, df2.shape, k))
            logger.error(e)
            raise RuntimeError(e)

        return dist, ind

    @staticmethod
    def get_cloud_nn(fpath, nsrdb_grid, labels=('latitude', 'longitude'),
                     var_kwargs=None):
        """Nearest neighbors computation for cloud data regrid.

        Parameters
        ----------
        fpath : str
            Full filepath to a single UW cloud data file.
        nsrdb_grid : pd.DataFrame
            Reference grid data for NSRDB.
        labels : list | tuple
            lat/lon column lables for the NSRDB grid and the cloud grid
        var_kwargs : dict | None
            Optional kwargs for the instantiation of the cloud var handler

        Returns
        -------
        index : np.ndarray | None
            KDTree query results mapping cloud data to the NSRDB grid. e.g.
            nsrdb_data = cloud_data[index]. None if bad grid data.
        """

        if isinstance(labels, tuple):
            labels = list(labels)

        if var_kwargs is None:
            var_kwargs = {}

        # Build NN tree based on the unique cloud grid at single timestep
        try:
            grid = VarFactory.get_cloud_handler(fpath, **var_kwargs).grid
        except Exception as e:
            msg = ('Exception building cloud NN '
                   'tree for {}: {}'.format(fpath, e))
            logger.error(msg)
            raise RuntimeError(msg)

        if grid is not None:
            tree = cKDTree(grid[labels])
            # Get the index of NN to NSRDB grid
            _, index = tree.query(nsrdb_grid[labels], k=1)
            index = index.astype(np.uint32)
            return index
        else:
            return None

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

        if var_obj.name in self.WEIGHTS and var_obj.name not in self._weights:
            logger.debug('Extracting weights for "{}"'.format(var_obj.name))
            wdf = pd.read_csv(self.WEIGHTS[var_obj.name], sep=' ',
                              skiprows=4, skipinitialspace=1)

            # use lat/lon and the current month and drop everything else
            current_col = str(self.date.month).zfill(2)
            wdf = wdf.rename({'Lat.': 'latitude', 'Long.': 'longitude',
                              current_col: 'weights'}, axis='columns')
            wdf = wdf[['latitude', 'longitude', 'weights']]

            # use geo nearest neighbors to find closest indices
            # between weights and MERRA grid
            _, i_nn = self.get_geo_nn(wdf, var_obj.grid, interp_method='NN',
                                      nn_method='haversine')

            weight_arr = wdf['weights'].values[i_nn.ravel()]
            weight_arr[(weight_arr < 0)] = 1
            self._weights[var_obj.name] = weight_arr

        if var_obj.name in self._weights:
            weights = self._weights[var_obj.name]
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

        if self._scale and isinstance(data, np.ndarray):
            var_obj = self._var_factory.get_base_handler(
                var, var_meta=self._var_meta, date=self.date)
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

        if self._scale and isinstance(data, np.ndarray):
            var_obj = self._var_factory.get_base_handler(
                var, var_meta=self._var_meta, date=self.date)
            data = var_obj.unscale_data(data)

        return data

    def run_pre_flight(self, var_list):
        """Run pre-flight checks, raise if specified paths/files are not found.

        Parameters
        ----------
        var_list : list
            List of variable names
        """

        missing_list = []
        for var in var_list:
            if var in self._var_factory.MAPPING:
                var_kwargs = self._factory_kwargs.get(var, {})
                var_obj = self._var_factory.get(var, var_meta=self._var_meta,
                                                name=var, date=self.date,
                                                dsets=[var], **var_kwargs)

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

    def _cloud_regrid(self, cloud_vars):
        """ReGrid data for multiple cloud variables to the NSRDB grid.

        (most efficient to process all cloud variables together to minimize
        number of kdtrees during regrid)

        Parameters
        ----------
        cloud_vars : tuple | list
            Source datasets to extract. It is more efficient to extract all
            required datasets at once from each cloud file, so that only one
            kdtree is built for each unique coordinate set in each cloud file.

        Returns
        -------
        data : dict
            Data dictionary of cloud datasets mapped to the NSRDB grid. Keys
            are the cloud variables names, values are 2D numpy arrays.
            Array shape is (n_time, n_sites).
        """

        # use the first cloud var name to get object,
        # full cloud_var list is passed in kwargs
        logger.info('Starting DataModel Cloud Regrid process')
        var_kwargs = self._factory_kwargs.get(cloud_vars[0], {})
        var_obj = self._var_factory.get(cloud_vars[0],
                                        var_meta=self._var_meta,
                                        name=cloud_vars[0],
                                        date=self.date,
                                        dsets=cloud_vars,
                                        freq=self.nsrdb_ti.freqstr,
                                        **var_kwargs)

        if self._max_workers != 1:
            logger.debug('Starting cloud data ReGrid with {} futures '
                         '(cloud timesteps).'.format(len(var_obj.flist)))
            regrid_ind = self._cloud_regrid_parallel(
                var_obj.flist, var_kwargs=var_kwargs)
        else:
            logger.debug('Starting cloud data ReGrid with {} iterations '
                         '(cloud timesteps) in serial.'
                         .format(len(var_obj.flist)))
            regrid_ind = {}
            # make the nearest neighbors regrid index mapping for all timesteps
            for fpath in var_obj.flist:
                logger.debug('Calculating ReGrid nearest neighbors for: {}'
                             .format(fpath))
                regrid_ind[fpath] = self.get_cloud_nn(
                    fpath, self.nsrdb_grid, var_kwargs=var_kwargs)

        logger.debug('Finished processing ReGrid nearest neighbors. Starting '
                     'to extract and map cloud data to the NSRDB grid.')

        data = {}
        # extract the regrided data for all timesteps
        for i, (timestamp, obj) in enumerate(var_obj):

            if timestamp != self.nsrdb_ti[i]:
                raise ValueError('Cloud iteration timestamp "{}" did not '
                                 'match NSRDB timestamp "{}" at index #{}'
                                 .format(timestamp, self.nsrdb_ti[i], i))

            # obj is None if cloud data file is missing
            if obj is not None:

                # save all datasets
                for dset, array in obj.source_data.items():

                    # initialize array based on time index and NN index result
                    if dset not in data:
                        if np.issubdtype(array.dtype, np.float):
                            data[dset] = np.full(self.nsrdb_data_shape, np.nan,
                                                 dtype=array.dtype)
                        else:
                            data[dset] = np.full(self.nsrdb_data_shape, -15,
                                                 dtype=array.dtype)

                    # write single timestep with NSRDB sites to appropriate row
                    # map the regridded data using the regrid NN indices
                    if regrid_ind[obj.fpath] is not None:
                        data[dset][i, :] = array[regrid_ind[obj.fpath]]
                    else:
                        wmsg = ('Cloud data does not appear to have valid '
                                'coordinates: {}'.format(obj.fpath))
                        warn(wmsg)
                        logger.warning(wmsg)

        # scale if requested
        if self._scale:
            for var, arr in data.items():
                data[var] = self.scale_data(var, arr)

        logger.info('Finished extracting cloud data '
                    'and writing to NSRDB arrays.')
        mem = psutil.virtual_memory()
        logger.info('Current memory usage is '
                    '{0:.3f} GB out of {1:.3f} GB total.'
                    .format(mem.used / 1e9, mem.total / 1e9))

        return data

    def _cloud_regrid_parallel(self, flist, var_kwargs=None):
        """Perform the ReGrid nearest neighbor calculations in parallel.

        Parameters
        ----------
        flist : list
            List of full file paths to cloud data files. Grid data from each
            file is mapped to the NSRDB grid in parallel.
        var_kwargs : dict | None
            Optional kwargs for the instantiation of the cloud var handler

        Returns
        -------
        regrid_ind : dict
            Dictionary of NN index results keyed by the file paths in the
            file list.
        """

        logger.debug('Starting cloud ReGrid parallel.')

        # start a local cluster
        logger.debug('Starting local cluster with {} workers.'
                     .format(self._max_workers))
        regrid_ind = {}
        with ProcessPoolExecutor(max_workers=self._max_workers) as exe:
            # make the nearest neighbors regrid index mapping for all timesteps
            for fpath in flist:
                regrid_ind[fpath] = exe.submit(self.get_cloud_nn, fpath,
                                               self.nsrdb_grid,
                                               var_kwargs=var_kwargs)

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
                logger.info('{} ReGrid futures are running, {} are complete. '
                            'Memory usage is {:.3f} GB '
                            'out of {:.3f} GB total.'
                            .format(running, complete, mem.used / 1e9,
                                    mem.total / 1e9))

            logger.info('Futures finished, maximum memory usage was '
                        '{0:.3f} GB out of {1:.3f} GB total.'
                        .format(max_mem, mem.total / 1e9))

            mem = psutil.virtual_memory()
            logger.info('Current memory usage is '
                        '{0:.3f} GB out of {1:.3f} GB total.'
                        .format(mem.used / 1e9, mem.total / 1e9))

            # gather results
            for k, v in regrid_ind.items():
                regrid_ind[k] = v.result()

        return regrid_ind

    def _process_dependencies(self, dependencies):
        """Ensure that all dependencies have been processed and set to self.

        Parameters
        ----------
        dependencies : list | tuple
            List of variable names representing dependencies that have to be
            processed before a downstream variable.
        """

        for dep in dependencies:

            # process and save data to processed attribute
            # (unscale to physical units)
            if dep not in self._processed:
                logger.info('Processing dependency "{}".'.format(dep))
                self[dep] = self._interpolate(dep)
                self[dep] = self.unscale_data(dep, self[dep])

            # dependency data dumped to disk, load from disk
            elif isinstance(self._processed[dep], str):
                logger.debug('Importing dependency "{}" from: {}'
                             .format(dep, self._processed[dep]))
                with Outputs(self._processed[dep]) as dep_out:
                    self[dep] = dep_out[dep]

            # dependency already in memory. Ensure physical units.
            else:
                self[dep] = self.unscale_data(dep, self[dep])

    def _derive(self, var, fpath_out=None):
        """Method for deriving variables (with dependencies).

        Parameters
        ----------
        var : str
            NSRDB var name.
        fpath_out : str | None
            File path to dump results. If no file path is given, results will
            be returned as an object.

        Returns
        -------
        data : np.ndarray
            NSRDB-resolution data for the given var and the current day.
        """

        if fpath_out is not None:
            fpath_out = fpath_out.format(var=var, i=self.nsrdb_grid.index[0])
            if os.path.exists(fpath_out):
                logger.info('Skipping DataModel for "{}" with existing '
                            'fpath_out: {}'.format(var, fpath_out))
                return fpath_out
            else:
                logger.info('Processing DataModel for "{}" with fpath_out: {}'
                            .format(var, fpath_out))

        if var in self.DEPENDENCIES:
            dependencies = self.DEPENDENCIES[var]

        else:
            raise KeyError('Did not recognize request to derive variable '
                           '"{}".'.format(var))

        # ensure dependencies are processed before working on derived var
        self._process_dependencies(dependencies)

        # get the derivation object from the var factory
        obj = self._var_factory.get(var)

        try:
            if var == 'solar_zenith_angle':
                data = obj.derive(
                    self.nsrdb_ti,
                    self.nsrdb_grid[['latitude', 'longitude']].values,
                    self.nsrdb_grid['elevation'].values,
                    self['surface_pressure'],
                    self['air_temperature'])
            else:
                data = obj.derive(*[self[k] for k in dependencies])

        except Exception as e:
            logger.exception('Could not derive "{}", received the exception: '
                             '{}'.format(var, e))
            raise e

        # convert units from MERRA to NSRDB
        data = self.convert_units(var, data)

        # scale if requested
        data = self.scale_data(var, data)

        # re-scale dependencies
        # (they had to be previously unscaled for physical eqns)
        for dep in dependencies:
            self[dep] = self.scale_data(dep, self[dep])

        if fpath_out is not None:
            data = self._dump(var, fpath_out, data)

        logger.info('Finished "{}".'.format(var))
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

        var_kwargs = self._factory_kwargs.get(var, {})
        var_obj = self._var_factory.get(var, var_meta=self._var_meta,
                                        name=var, date=self.date,
                                        **var_kwargs)

        if 'albedo' in var:
            # special exclusions for large-extent albedo
            var_obj.exclusions_from_nsrdb(self.nsrdb_grid)

        # get ancillary data source data array
        data = var_obj.source_data

        # get mapping from source data grid to NSRDB
        dist, ind = self.get_geo_nn(var_obj.grid, self.nsrdb_grid,
                                    interp_method=var_obj.spatial_method,
                                    nn_method=var_obj.NN_METHOD,
                                    cache=var_obj.cache_file)

        # perform weighting if applicable
        if var in self.WEIGHTS and 'merra' in str(type(var_obj)).lower():
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
    def _process_multiple(cls, var_list, date, nsrdb_grid,
                          nsrdb_freq='5min', var_meta=None,
                          max_workers=None, max_workers_clouds=None,
                          fpath_out=None, factory_kwargs=None):
        """Process ancillary data for multiple variables for a single day.

        Parameters
        ----------
        var_list : list | None
            List of variables to process
        date : datetime.date
            Single day to extract ancillary data for.
        nsrdb_grid : str | pd.DataFrame
            CSV file containing the NSRDB reference grid to interpolate to,
            or a pre-extracted (and reduced) dataframe.
        nsrdb_freq : str
            Final desired NSRDB temporal frequency.
        var_meta : str | pd.DataFrame
            CSV file or dataframe containing meta data for all NSRDB variables.
        max_workers : int | None
            Maximum workers to use in parallel. 1 will run serial, None will
            use all available parallel workers.
        max_workers_clouds : int | None
            Maximum workers to use in parallel for the cloud regrid algorithm.
        return_obj : bool
            Flag to return full DataModel object instead of just the processed
            data dictionary.
        fpath_out : str | None
            File path to dump results. If no file path is given, results will
            be returned as an object.
        factory_kwargs : dict | None
            Optional namespace of kwargs to use to initialize variable data
            handlers from the data model's variable factory. Keyed by
            variable name. Values can be "source_dir", "handler", etc...
            source_dir for cloud variables can be a normal directory
            path or /directory/prefix*suffix where /directory/ can have
            more sub dirs

        Returns
        -------
        out : DataModel
            Full DataModel object with the data in the .processed property.
        """

        data_model = cls(date, nsrdb_grid, nsrdb_freq=nsrdb_freq,
                         var_meta=var_meta, factory_kwargs=factory_kwargs,
                         max_workers=max_workers)

        # run pre-flight checks
        data_model.run_pre_flight(var_list)

        # default multiple compute
        if var_list is None:
            var_list = cls.ALL_SKY_VARS

        # remove cloud variables from var_list to be processed together
        # (most efficient to process all cloud variables together to minimize
        # number of kdtrees during regrid)
        cloud_vars = []
        for cv in var_list:
            is_cv = cv in cls.CLOUD_VARS or cv in cls.MLCLOUDS_VARS
            var_fact_kwargs = data_model._factory_kwargs.get(cv, {})
            if 'handler' in var_fact_kwargs:
                is_cv = var_fact_kwargs['handler'].lower() == 'cloudvar'
            if is_cv:
                cloud_vars.append(cv)

        var_list = [v for v in var_list if v not in cloud_vars]

        # remove derived (dependent) variables from var_list to be processed
        # last (most efficient to process depdencies first, dependents last)
        derived_vars = [dv for dv in var_list if dv in cls.DERIVED_VARS]
        var_list = [v for v in var_list if v not in cls.DERIVED_VARS]

        logger.info('First processing data for variable list: {}'
                    .format(var_list))
        logger.info('Then processing cloud data for variable list: {}'
                    .format(cloud_vars))
        logger.info('Finally, processing derived data for variable list: {}'
                    .format(derived_vars))

        if var_list:
            # run in serial
            if max_workers == 1:
                data = {}
                for var in var_list:
                    data_model[var] = cls.run_single(
                        var, date, nsrdb_grid, nsrdb_freq=nsrdb_freq,
                        var_meta=var_meta, fpath_out=fpath_out,
                        factory_kwargs=factory_kwargs)
            # run in parallel
            else:
                data = cls._process_parallel(
                    var_list, date, nsrdb_grid,
                    max_workers=max_workers,
                    nsrdb_freq=nsrdb_freq,
                    var_meta=var_meta,
                    fpath_out=fpath_out,
                    factory_kwargs=factory_kwargs)

                for k, v in data.items():
                    data_model[k] = v

        # process cloud variables together
        if cloud_vars:
            data_model['clouds'] = cls.run_clouds(
                cloud_vars, date, nsrdb_grid,
                nsrdb_freq=nsrdb_freq,
                var_meta=var_meta,
                max_workers=max_workers_clouds,
                fpath_out=fpath_out,
                factory_kwargs=factory_kwargs)

        # process derived (dependent) variables last using the built
        # AncillaryDataProcessing object instance.
        if derived_vars:
            for var in derived_vars:
                data_model[var] = data_model._derive(var, fpath_out=fpath_out)

        # scale if requested
        for var, arr in data_model._processed.items():
            data_model[var] = data_model.scale_data(var, arr)

        return data_model

    @staticmethod
    def _process_parallel(var_list, date, nsrdb_grid, max_workers=None,
                          **kwargs):
        """Process ancillary variables in parallel.

        Parameters
        ----------
        var_list : list | tuple
            List of variables to process in parallel
        date : datetime.date
            Single day to extract MERRA2 data for.
        nsrdb_grid : str | pd.DataFrame
            CSV file containing the NSRDB reference grid to interpolate to,
            or a pre-extracted (and reduced) dataframe.
        max_workers : int | None
            Optional limit for maximum parallel workers.
            None will use all available.
        **kwargs : dict
            Keyword args to pass to DataModel.run_single.

        Returns
        -------
        futures : dict
            Gathered futures, namespace of nsrdb data numpy arrays keyed by
            nsrdb variable name.
        """

        logger.info('Processing variables in parallel: {}'.format(var_list))
        # start a local cluster
        futures = {}
        logger.debug('Starting local cluster with {} workers.'
                     .format(max_workers))
        with ProcessPoolExecutor(max_workers=max_workers) as exe:
            # submit a future for each merra variable (non-calculated)
            for var in var_list:
                futures[var] = exe.submit(
                    DataModel.run_single, var, date, nsrdb_grid, **kwargs)

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
                logger.info('{} DataModel processing futures are running: {} '
                            'memory usage is {:.3f} GB out of {:.3f} GB total'
                            .format(running, keys, mem.used / 1e9,
                                    mem.total / 1e9))

            logger.info('Futures finished, maximum memory usage was '
                        '{0:.3f} GB out of {1:.3f} GB total.'
                        .format(max_mem, mem.total / 1e9))

            mem = psutil.virtual_memory()
            logger.info('Current memory usage is '
                        '{0:.3f} GB out of {1:.3f} GB total.'
                        .format(mem.used / 1e9, mem.total / 1e9))

            # gather results
            for k, v in futures.items():
                futures[k] = v.result()

        return futures

    def _dump(self, var, fpath_out, data, purge=True):
        """Run ancillary data processing for one variable for a single day.

        Parameters
        ----------
        var : str
            NSRDB var name.
        fpath_out : str | None
            File path to dump results. If no file path is given, results will
            be returned as an object.
        data : np.ndarray
            NSRDB-resolution data for the given var and the current day.
        purge : bool
            Flag to purge data from memory after dumping to disk

        Returns
        -------
        data : str | np.ndarray
            Input data array if no purge, else file path to dump results.
        """

        if isinstance(fpath_out, str):
            if '{var}' in fpath_out and '{i}' in fpath_out:
                fpath_out = fpath_out.format(var=var,
                                             i=self.nsrdb_grid.index[0])

            logger.info('Writing: {}'.format(os.path.basename(fpath_out)))

            # make file for each var
            with Outputs(fpath_out, mode='w') as fout:
                fout.time_index = self.nsrdb_ti
                fout.meta = self.nsrdb_grid

                var_obj = VarFactory.get_base_handler(
                    var, var_meta=self._var_meta, date=self.date)
                attrs = var_obj.attrs

                fout._add_dset(dset_name=var, data=data,
                               dtype=var_obj.final_dtype,
                               chunks=var_obj.chunks, attrs=attrs)

            if purge:
                del data
                data = fpath_out

        return data

    @classmethod
    def run_single(cls, var, date, nsrdb_grid, nsrdb_freq='5min',
                   var_meta=None, fpath_out=None, factory_kwargs=None):
        """Run ancillary data processing for one variable for a single day.

        Parameters
        ----------
        var : str
            NSRDB var name.
        date : datetime.date
            Single day to extract ancillary data for.
        nsrdb_grid : str | pd.DataFrame
            CSV file containing the NSRDB reference grid to interpolate to,
            or a pre-extracted (and reduced) dataframe. The first csv column
            must be the NSRDB site gid's.
        nsrdb_freq : str
            Final desired NSRDB temporal frequency.
        var_meta : str | pd.DataFrame | None
            CSV file or dataframe containing meta data for all NSRDB variables.
            Defaults to the NSRDB var meta csv in git repo.
        fpath_out : str | None
            File path to dump results. If no file path is given, results will
            be returned as an object.
        factory_kwargs : dict | None
            Optional namespace of kwargs to use to initialize variable data
            handlers from the data model's variable factory. Keyed by
            variable name. Values can be "source_dir", "handler", etc...
            source_dir for cloud variables can be a normal directory
            path or /directory/prefix*suffix where /directory/ can have
            more sub dirs

        Returns
        -------
        data : np.ndarray
            NSRDB-resolution data for the given var and the current day.
        """

        data_model = cls(date, nsrdb_grid, nsrdb_freq=nsrdb_freq,
                         var_meta=var_meta, factory_kwargs=factory_kwargs)

        if fpath_out is None:
            logger.info('Processing data for "{}".'.format(var))
        else:
            if '{var}' not in fpath_out or '{i}' not in fpath_out:
                raise IOError('Cannot write to fpath_out, need "var" and "i" '
                              'format keywords: {}'.format(fpath_out))
            fpath_out = fpath_out.format(var=var,
                                         i=data_model.nsrdb_grid.index[0])
            if os.path.exists(fpath_out):
                logger.info('Skipping DataModel for "{}" with existing '
                            'fpath_out: {}'.format(var, fpath_out))
                return fpath_out
            else:
                logger.info('Processing DataModel for "{}" with fpath_out: {}'
                            .format(var, fpath_out))

        is_cv = var in cls.CLOUD_VARS or var in cls.MLCLOUDS_VARS
        var_fact_kwargs = data_model._factory_kwargs.get(var, {})
        if 'handler' in var_fact_kwargs:
            is_cv = var_fact_kwargs['handler'].lower() == 'cloudvar'

        if is_cv:
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

        if fpath_out is not None:
            data = data_model._dump(var, fpath_out, data)
        logger.info('Finished "{}".'.format(var))
        return data

    @classmethod
    def run_clouds(cls, cloud_vars, date, nsrdb_grid,
                   nsrdb_freq='5min', var_meta=None, max_workers=None,
                   fpath_out=None, factory_kwargs=None):
        """Run cloud processing for multiple cloud variables.

        (most efficient to process all cloud variables together to minimize
        number of kdtrees during regrid)

        Parameters
        ----------
        cloud_vars : list | tuple
            NSRDB cloud variables names.
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
        var_meta : str | pd.DataFrame | None
            CSV file or dataframe containing meta data for all NSRDB variables.
            Defaults to the NSRDB var meta csv in git repo.
        max_workers : int | None
            Maximum workers to use in parallel. 1 will run serial,
            None will use all available.
        fpath_out : str | None
            File path to dump results. If no file path is given, results will
            be returned as an object.
        factory_kwargs : dict | None
            Optional namespace of kwargs to use to initialize variable data
            handlers from the data model's variable factory. Keyed by
            variable name. Values can be "source_dir", "handler", etc...
            source_dir for cloud variables can be a normal directory
            path or /directory/prefix*suffix where /directory/ can have
            more sub dirs

        Returns
        -------
        data : dict
            Namespace of nsrdb data numpy arrays keyed by nsrdb variable name.
        """

        logger.info('Processing data for multiple cloud variables: {}'
                    .format(cloud_vars))

        data_model = cls(date, nsrdb_grid, nsrdb_freq=nsrdb_freq,
                         var_meta=var_meta, factory_kwargs=factory_kwargs,
                         max_workers=max_workers)

        if fpath_out is not None:
            data = {}
            skip_list = []
            for var in cloud_vars:
                fpath_out_var = fpath_out.format(
                    var=var, i=data_model.nsrdb_grid.index[0])
                if os.path.exists(fpath_out_var):
                    logger.info('Skipping DataModel for "{}" with existing '
                                'fpath_out: {}'.format(var, fpath_out_var))
                    skip_list.append(var)
                    data[var] = fpath_out_var
            if any(skip_list):
                cloud_vars = [cv for cv in cloud_vars if cv not in skip_list]
            if not any(cloud_vars):
                return data

        try:
            data = data_model._cloud_regrid(cloud_vars)
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

        if fpath_out is not None:
            for var, arr in data.items():
                fpath_out_var = fpath_out.format(
                    var=var, i=data_model.nsrdb_grid.index[0])
                data[var] = data_model._dump(var, fpath_out_var, arr)

        logger.info('Finished "{}".'.format(cloud_vars))

        return data

    @classmethod
    def run_multiple(cls, var_list, date, nsrdb_grid,
                     nsrdb_freq='5min', var_meta=None,
                     max_workers=None, max_workers_clouds=None,
                     return_obj=False, fpath_out=None, factory_kwargs=None):
        """Run ancillary data processing for multiple variables for single day.

        Parameters
        ----------
        var_list : list | None
            List of variables to process
        date : datetime.date
            Single day to extract ancillary data for.
        nsrdb_grid : str | pd.DataFrame
            CSV file containing the NSRDB reference grid to interpolate to,
            or a pre-extracted (and reduced) dataframe. The first csv column
            must be the NSRDB site gid's.
        nsrdb_freq : str
            Final desired NSRDB temporal frequency.
        var_meta : str | pd.DataFrame | None
            CSV file or dataframe containing meta data for all NSRDB variables.
            Defaults to the NSRDB var meta csv in git repo.
        max_workers : int | None
            Number of workers to run in parallel. 1 will run serial,
            None will use all available.
        max_workers_clouds : int | None
            Number of workers to run in parallel for the cloud regrid algorithm
        return_obj : bool
            Flag to return full DataModel object instead of just the processed
            data dictionary.
        fpath_out : str | None
            File path to dump results. If no file path is given, results will
            be returned as an object.
        factory_kwargs : dict | None
            Optional namespace of kwargs to use to initialize variable data
            handlers from the data model's variable factory. Keyed by
            variable name. Values can be "source_dir", "handler", etc...
            source_dir for cloud variables can be a normal directory
            path or /directory/prefix*suffix where /directory/ can have
            more sub dirs

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

        logger.debug('Running DataModel with the following var meta: {}'
                     .format(var_meta))
        logger.debug('Running DataModel with the following variable '
                     'factory kwargs: {}'.format(factory_kwargs))

        data_model = cls._process_multiple(
            var_list, date, nsrdb_grid,
            nsrdb_freq=nsrdb_freq,
            var_meta=var_meta,
            max_workers=max_workers,
            max_workers_clouds=max_workers_clouds,
            fpath_out=fpath_out,
            factory_kwargs=factory_kwargs)

        # Create an AncillaryDataProcessing object instance for storing data.
        logger.info('Final NSRDB output shape is: {}'
                    .format(data_model.nsrdb_data_shape))

        logger.info('DataModel processing complete for: {}'.format(date))

        if return_obj:
            return data_model
        else:
            return data_model.processed_data
