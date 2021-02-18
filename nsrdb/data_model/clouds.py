# -*- coding: utf-8 -*-
"""A framework for handling UW/GOES source data."""
import datetime
import math
import statistics
import numpy as np
import pandas as pd
import h5py
import re
import os
import psutil
import netCDF4
import logging
from warnings import warn

from nsrdb.data_model.base_handler import AncillaryVarHandler


logger = logging.getLogger(__name__)


class CloudCoords:
    """Class to correct cloud coordinates based on solar position / shading."""

    EARTH_RADIUS = 6371
    TO_RAD = math.pi / 180
    TO_DEG = 180 / math.pi

    REQUIRED = ('latitude_pc',
                'longitude_pc',
                'solar_zenith_angle',
                'solar_azimuth_angle',
                'cld_height_acha')

    @staticmethod
    def check_file(fp):
        """Check if file has required vars for cloud coord correction"""

        if fp.endswith('.nc'):
            with netCDF4.Dataset(fp, 'r') as f:
                dsets = sorted(list(f.variables.keys()))

        elif fp.endswith('.h5'):
            with h5py.File(fp, 'r') as f:
                dsets = list(f)

        else:
            e = ('Could not parse cloud file, expecting .h5 or .nc but '
                 'received: {}'.format(os.path.basename(fp)))
            logger.error(e)
            raise IOError(e)

        check = [d in dsets for d in CloudCoords.REQUIRED]

        return all(check)

    @staticmethod
    def dist_to_latitude(dist):
        """Calculate change in latitude in decimal degrees given distance
        differences north/south in km..

        Parameters
        ----------
        dist : np.ndarray
            Array of change in east/west location in km.

        Returns
        -------
        delta_lat : np.ndarray
            Array of change in north/south location in decimal degrees.
        """
        delta_lat = (dist / CloudCoords.EARTH_RADIUS) * CloudCoords.TO_DEG
        return delta_lat

    @staticmethod
    def dist_to_longitude(latitude, dist):
        """Calculate change in longitude in decimal degrees given a latitude
        and distance differences east/west in km.

        Parameters
        ----------
        latitude : np.ndarray
            Array of latitude values in decimal degrees.
        dist : np.ndarray
            Array of change in east/west location in km.

        Returns
        -------
        delta_lon : np.ndarray
            Array of change in east/west location in decimal degrees.
        """
        # Find the radius of a circle around the earth at given latitude.
        r = CloudCoords.EARTH_RADIUS * np.cos(latitude * CloudCoords.TO_RAD)
        delta_lon = (dist / r) * CloudCoords.TO_DEG
        return delta_lon

    @classmethod
    def adjust_coords(cls, lat, lon, sza, azi, cld_height, sza_threshold=85):
        """Adjust parallax corrected coordinates for sun position shading.

        Parameters
        ----------
        lat : np.ndarray
            Latitude values in decimal degrees
        lon : np.ndarray
            Longitude values in decimal degrees
        sza : np.ndarray
            Solar zenith angle for every lat/lon value for one timestep
            in degrees.
        azi : np.ndarray
            Solar azimuth angle for every lat/lon value for one timestep
            in degrees.
        cld_height : np.ndarray
            Cloud height in km.
        sza_threshold : float | int
            Thresold over which coordinate adjustments are truncated.
            Coordinate solar adjustments to to infinity at sza of 90.

        Returns
        -------
        lat : np.ndarray
            Latitude values in decimal degrees adjusted for the sun position
            so that clouds are mapped to the coordinates they are shading.
        lon : np.ndarray
            Longitude values in decimal degrees adjusted for the sun position
            so that clouds are mapped to the coordinates they are shading.
        """

        shapes = {'lat': lat.shape, 'lon': lon.shape,
                  'sza': sza.shape, 'azi': azi.shape,
                  'cld_height': cld_height.shape}
        shapes_list = list(shapes.values())
        check = [shapes_list[0] == x for x in shapes_list[1:]]
        if not all(check):
            e = ('Cannot run cloud coordinate shading adjustment. '
                 'Input shapes: {}'.format(shapes))
            logger.error(e)
            raise ValueError(e)

        if np.nanmax(cld_height) > 1000:
            cld_height /= 1000

        cld_height[(cld_height < 0)] = np.nan
        sza[(sza > sza_threshold)] = sza_threshold
        sza *= cls.TO_RAD
        azi *= cls.TO_RAD

        delta_dist = cld_height * np.tan(sza)
        delta_x = -1 * delta_dist * np.sin(azi)
        delta_y = -1 * delta_dist * np.cos(azi)

        logger.debug('Cloud height min / mean / max: {} / {} / {}'
                     .format(np.nanmin(cld_height),
                             np.nanmean(cld_height),
                             np.nanmax(cld_height)))
        logger.debug('Delta dist min / mean / max: {} / {} / {}'
                     .format(np.nanmin(delta_dist),
                             np.nanmean(delta_dist),
                             np.nanmax(delta_dist)))

        delta_lon = cls.dist_to_longitude(lat, delta_x)
        delta_lat = cls.dist_to_latitude(delta_y)

        logger.debug('Delta lon min / mean / max: {} / {} / {}'
                     .format(np.nanmin(delta_lon),
                             np.nanmean(delta_lon),
                             np.nanmax(delta_lon)))
        logger.debug('Delta lat min / mean / max: {} / {} / {}'
                     .format(np.nanmin(delta_lat),
                             np.nanmean(delta_lat),
                             np.nanmax(delta_lat)))

        delta_lon[np.isnan(delta_lon)] = 0.0
        delta_lat[np.isnan(delta_lat)] = 0.0

        lon += delta_lon
        lat += delta_lat

        return lat, lon


class CloudVarSingle:
    """Base framework for single-file/single-timestep cloud data extraction."""

    def __init__(self, fpath, pre_proc_flag=True, index=None,
                 dsets=('cloud_type', 'cld_opd_dcomp', 'cld_reff_dcomp',
                        'cld_press_acha')):
        """
        Parameters
        ----------
        fpath : str
            Full filepath for the cloud data at a single timestep.
        pre_proc_flag : bool
            Flag to pre-process and sparsify data.
        index : np.ndarray
            Nearest neighbor results array to extract a subset of the data.
        dsets : tuple | list
            Source datasets to extract.
        """

        self._fpath = fpath
        self._dsets = dsets
        self.pre_proc_flag = pre_proc_flag
        self._index = index
        self._grid = None

    @property
    def fpath(self):
        """Get the full file path for this cloud data timestep."""
        return self._fpath

    @property
    def grid(self):
        """Return the cloud data grid for the current timestep.

        Returns
        -------
        self._grid : pd.DataFrame | None
            GOES source coordinates (labels: ['latitude', 'longitude']).
            None if bad dataset
        """
        return self._grid


class CloudVarSingleH5(CloudVarSingle):
    """Framework for .h5 single-file/single-timestep cloud data extraction."""

    def __init__(self, fpath, pre_proc_flag=True, index=None,
                 dsets=('cloud_type', 'cld_opd_dcomp', 'cld_reff_dcomp',
                        'cld_press_acha'), adjust_coords=False):
        """
        Parameters
        ----------
        fpath : str
            Full filepath for the cloud data at a single timestep.
        pre_proc_flag : bool
            Flag to pre-process and sparsify data.
        index : np.ndarray
            Nearest neighbor results array to extract a subset of the data.
        dsets : tuple | list
            Source datasets to extract.
        adjust_coords : bool
            Flag to adjust cloud coordinates so clouds are assigned to the
            coordiantes they shade.
        """

        super().__init__(fpath, pre_proc_flag=pre_proc_flag, index=index,
                         dsets=dsets)
        self._grid = self._parse_grid(self._fpath, adjust_coords=adjust_coords)
        if self.pre_proc_flag:
            self._grid, self._sparse_mask = self.make_sparse(self._grid)

    @property
    def dsets(self):
        """Get a list of the available datasets in the cloud file."""
        with h5py.File(self._fpath, 'r') as f:
            out = list(f)

        return out

    @classmethod
    def _parse_grid(cls, fpath, dsets=('latitude_pc', 'longitude_pc'),
                    adjust_coords=False):
        """Extract the cloud data grid for the current timestep.

        Parameters
        ----------
        fpath : str
            Full file path to netcdf4 file.
        dsets : tuple
            Latitude, longitude datasets to retrieve from cloud file. H5
            files have parallax corrected datasets (*_pc). Output dataset
            names will always be 'latitude' and 'longitude'.
        adjust_coords : bool
            Flag to adjust cloud coordinates so clouds are assigned to the
            coordiantes they shade.

        Returns
        -------
        grid : pd.DataFrame | None
            GOES source coordinates (labels: ['latitude', 'longitude']).
            None if bad dataset
        """

        grid = {}
        with h5py.File(fpath, 'r') as f:
            for dset in dsets:

                if 'lat' in dset:
                    dset_out = 'latitude'
                elif 'lon' in dset:
                    dset_out = 'longitude'
                else:
                    raise KeyError('Did not recognize dataset as latitude '
                                   'or longitude: "{}"'.format(dset))

                if dset not in list(f):
                    wmsg = ('Could not find {}. Using {} instead.'
                            .format(dset, dset_out))
                    warn(wmsg)
                    logger.warning(wmsg)
                    dset = dset_out

                grid[dset_out] = cls.pre_process(dset, f[dset][...],
                                                 dict(f[dset].attrs))

        if grid and CloudCoords.check_file(fpath) and adjust_coords:
            grid = cls.solpo_grid_adjust(fpath, grid)

        grid = pd.DataFrame(grid)

        if grid.empty:
            grid = None

        return grid

    @classmethod
    def solpo_grid_adjust(cls, fpath, grid):
        """Adjust grid lat/lon values based on solar position

        Parameters
        ----------
        fpath : str
            Filepath to cloud h5 file containing required datasets for solpo
            coodinate adjustment.
        grid : dict
            Dictionary with latitude and longitude keys and corresponding
            numpy array values.

        Returns
        -------
        grid : dict
            Dictionary with latitude and longitude keys and corresponding
            numpy array values. Coordinates are adjusted for solar position
            so that clouds are linked to the coordinate that they are shading.
        """
        with h5py.File(fpath, 'r') as f:
            sza = cls.pre_process(
                'solar_zenith_angle', f['solar_zenith_angle'][...],
                dict(f['solar_zenith_angle'].attrs))
            azi = cls.pre_process(
                'solar_azimuth_angle', f['solar_azimuth_angle'][...],
                dict(f['solar_azimuth_angle'].attrs))
            cld_height = cls.pre_process(
                'cld_height_acha', f['cld_height_acha'][...],
                dict(f['cld_height_acha'].attrs))

        try:
            lat, lon = CloudCoords.adjust_coords(grid['latitude'],
                                                 grid['longitude'],
                                                 sza, azi, cld_height)
        except Exception as e:
            logger.warning('Could not perform cloud coordinate adjustment '
                           'for: {}, received error: {}'
                           .format(os.path.basename(fpath), e))
        else:
            logger.debug('Successfully performed solar position shading '
                         'adjustment of cloud coordinates')
            grid['latitude'], grid['longitude'] = lat, lon

        return grid

    @staticmethod
    def pre_process(dset, data, attrs, sparse_mask=None, index=None):
        """Pre-process cloud data by filling missing values and unscaling.

        Pre-processing steps (different for .nc vs .h5):
            1. flatten (ravel)
            2. convert to float32 (unless dset == cloud_type)
            3. convert filled values to NaN (unless dset == cloud_type)
            4. apply scale factor (multiply)
            5. apply add offset (addition)
            6. sparsify
            7. extract only data at index

        Parameters
        ----------
        dset : str
            Dataset name.
        data : np.ndarray
            Raw data extracted from the dataset in the cloud data source file.
        attrs : dict
            Dataset attributes from the dataset in the cloud data source file.
        sparse_mask : NoneType | pd.Series
            Optional boolean mask to apply to the data to sparsify.
        index : np.ndarray
            Nearest neighbor results array to extract a subset of the data.

        Returns
        -------
        data : np.ndarray
            Pre-processed data.
        """

        data = data.ravel()
        if dset != 'cloud_type':
            data = data.astype(np.float32)

        if '_FillValue' in attrs and data.dtype == np.float32:
            mask = np.where(data == attrs['_FillValue'])[0]
            data[mask] = np.nan

        if 'scale_factor' in attrs:
            data *= attrs['scale_factor']

        if 'add_offset' in attrs:
            data += attrs['add_offset']

        if sparse_mask is not None:
            if data.shape != sparse_mask.shape:
                msg = ('Data model failed while processing "{}" which has '
                       'shape {} while the coordinate grid mask has shape {}'
                       .format(dset, data.shape, sparse_mask.shape))
                logger.error(msg)
                raise RuntimeError(msg)
            else:
                data = data[sparse_mask]

        if index is not None:
            data = data[index]

        return data

    @staticmethod
    def make_sparse(grid):
        """Make the cloud grid sparse by removing NaN coordinates.

        Parameters
        ----------
        grid : pd.DataFrame
            GOES source coordinates (labels: ['latitude', 'longitude']).

        Returns
        -------
        grid : pd.DataFrame
            Sparse GOES source coordinates with all NaN rows removed.
        mask : pd.Series
            Boolean series; the mask to extract sparse data.
        """

        mask = (grid['latitude'] == -90) & (grid['longitude'] == -180)
        grid.loc[mask, :] = np.nan
        mask = ~(pd.isna(grid['latitude']) | pd.isna(grid['longitude']))
        grid = grid[mask]
        return grid, mask

    @property
    def source_data(self):
        """Get multiple-variable data dictionary from the cloud data file.

        Returns
        -------
        data : dict
            Dictionary of multiple cloud datasets. Keys are the cloud dataset
            names. Values are 1D (flattened/raveled) arrays of data.
        """

        data = {}
        with h5py.File(self._fpath, 'r') as f:
            for dset in self._dsets:
                if dset not in list(f):
                    raise KeyError('Could not find "{}" in the cloud '
                                   'file: {}'
                                   .format(dset, self._fpath))

                if self.pre_proc_flag:
                    data[dset] = self.pre_process(
                        dset, f[dset][...], dict(f[dset].attrs),
                        sparse_mask=self._sparse_mask, index=self._index)
                else:
                    data[dset] = f[dset][...].ravel()

        return data


class CloudVarSingleNC(CloudVarSingle):
    """Framework for .nc single-file/single-timestep cloud data extraction."""

    def __init__(self, fpath, pre_proc_flag=True, index=None,
                 dsets=('cloud_type', 'cld_opd_dcomp', 'cld_reff_dcomp',
                        'cld_press_acha'), adjust_coords=False):
        """
        Parameters
        ----------
        fpath : str
            Full filepath for the cloud data at a single timestep.
        pre_proc_flag : bool
            Flag to pre-process and sparsify data.
        index : np.ndarray
            Nearest neighbor results array to extract a subset of the data.
        dsets : tuple | list
            Source datasets to extract.
        adjust_coords : bool
            Flag to adjust cloud coordinates so clouds are assigned to the
            coordiantes they shade.
        """

        super().__init__(fpath, pre_proc_flag=pre_proc_flag, index=index,
                         dsets=dsets)
        self._grid, self._sparse_mask = self._parse_grid(
            self._fpath, adjust_coords=adjust_coords)

    @property
    def dsets(self):
        """Get a list of the available datasets in the cloud file."""
        with netCDF4.Dataset(self._fpath, 'r') as f:
            out = list(f.variables.keys())

        return out

    @classmethod
    def _parse_grid(cls, fpath, dsets=('latitude_pc', 'longitude_pc'),
                    adjust_coords=False):
        """Extract the cloud data grid for the current timestep.

        Parameters
        ----------
        fpath : str
            Full file path to netcdf4 file.
        dsets : tuple
            Latitude, longitude datasets to retrieve from cloud file. NetCDF4
            files have parallax corrected datasets (*_pc). Output dataset
            names will always be 'latitude' and 'longitude'.
        adjust_coords : bool
            Flag to adjust cloud coordinates so clouds are assigned to the
            coordiantes they shade.

        Returns
        -------
        grid : pd.DataFrame | None
            GOES source coordinates (labels: ['latitude', 'longitude']).
            None if bad dataset
        mask : np.ndarray
            2D boolean array to extract good data.
        """

        sparse_mask = None
        grid = {}

        with netCDF4.Dataset(fpath, 'r') as f:
            for dset in dsets:

                if 'lat' in dset:
                    dset_out = 'latitude'
                elif 'lon' in dset:
                    dset_out = 'longitude'
                else:
                    raise KeyError('Did not recognize dataset as latitude '
                                   'or longitude: "{}"'.format(dset))

                if dset not in list(f.variables.keys()):
                    wmsg = ('Could not find {}. Using {} instead.'
                            .format(dset, dset_out))
                    warn(wmsg)
                    logger.warning(wmsg)
                    dset = dset_out

                # use netCDF masked array mask to reduce ~1/4 of the data
                if sparse_mask is None:
                    try:
                        sparse_mask = ~f[dset][:].mask
                    except Exception as e:
                        msg = 'Exception masking {} in {}: {}'.format(dset,
                                                                      fpath,
                                                                      e)
                        logger.error(msg)
                        raise RuntimeError(msg)
                grid[dset_out] = f[dset][:].data[sparse_mask]

        if grid and CloudCoords.check_file(fpath) and adjust_coords:
            grid = cls.solpo_grid_adjust(fpath, grid, sparse_mask)

        grid = pd.DataFrame(grid)

        if grid.empty:
            grid = None

        if sparse_mask.sum() == 0:
            msg = ('Cloud data handler had a completely empty sparse mask '
                   'for: {}'.format(fpath))
            logger.warning(msg)
            warn(msg)

        return grid, sparse_mask

    @staticmethod
    def solpo_grid_adjust(fpath, grid, sparse_mask):
        """Adjust grid lat/lon values based on solar position

        Parameters
        ----------
        fpath : str
            Filepath to cloud nc file containing required datasets for solpo
            coodinate adjustment.
        grid : dict
            Dictionary with latitude and longitude keys and corresponding
            numpy array values.
        sparse_mask : np.ndarray
            Boolean array to mask the native dataset shapes from fpath.

        Returns
        -------
        grid : dict
            Dictionary with latitude and longitude keys and corresponding
            numpy array values. Coordinates are adjusted for solar position
            so that clouds are linked to the coordinate that they are shading.
        """
        with netCDF4.Dataset(fpath, 'r') as f:
            sza = f['solar_zenith_angle'][:].data[sparse_mask]
            azi = f['solar_azimuth_angle'][:].data[sparse_mask]
            cld_height = f['cld_height_acha'][:].data[sparse_mask]

        try:
            lat, lon = CloudCoords.adjust_coords(grid['latitude'],
                                                 grid['longitude'],
                                                 sza, azi, cld_height)
        except Exception as e:
            logger.warning('Could not perform cloud coordinate adjustment '
                           'for: {}, received error: {}'
                           .format(os.path.basename(fpath), e))
        else:
            logger.debug('Successfully performed solar position shading '
                         'adjustment of cloud coordinates')
            grid['latitude'], grid['longitude'] = lat, lon

        return grid

    @staticmethod
    def pre_process(dset, data, fill_value=None, sparse_mask=None, index=None):
        """Pre-process cloud data by filling missing values and unscaling.

        Pre-processing steps (different for .nc vs .h5):
            1. sparsify
            2. flatten (ravel)
            3. convert to float32 (unless dset == cloud_type)
            4. convert filled values to NaN (unless dset == cloud_type)
            5. extract only data at index

        Parameters
        ----------
        dset : str
            Dataset name.
        data : np.ndarray
            Raw data extracted from the dataset in the cloud data source file.
            For the .nc files, this data is already unscaled.
        fill_value : NoneType | int | float
            Value that was assigned if the data was missing. These entries
            in data will be converted to NaN if possible.
        sparse_mask : NoneType | pd.Series
            Optional boolean mask to apply to the data to sparsify. For the
            .nc files, this is taken from the masked coordinate arrays.
        index : np.ndarray
            Nearest neighbor results array to extract a subset of the data.

        Returns
        -------
        data : np.ndarray
            Pre-processed data.
        """

        if sparse_mask is not None:
            if data.shape != sparse_mask.shape:
                msg = ('Data model failed while processing "{}" which has '
                       'shape {} while the coordinate grid mask has shape {}'
                       .format(dset, data.shape, sparse_mask.shape))
                logger.error(msg)
                raise RuntimeError(msg)
            else:
                data = data[sparse_mask]

        data = data.ravel()

        if dset != 'cloud_type':
            data = data.astype(np.float32)

        if fill_value is not None and data.dtype == np.float32:
            mask = np.where(data == fill_value)[0]
            data[mask] = np.nan

        if index is not None:
            data = data[index]

        return data

    @property
    def source_data(self):
        """Get multiple-variable data dictionary from the cloud data file.

        Returns
        -------
        data : dict
            Dictionary of multiple cloud datasets. Keys are the cloud dataset
            names. Values are 1D (flattened/raveled) arrays of data.
        """

        data = {}
        with netCDF4.Dataset(self._fpath, 'r') as f:
            for dset in self._dsets:
                if dset not in list(f.variables.keys()):
                    raise KeyError('Could not find "{}" in the cloud '
                                   'file: {}'
                                   .format(dset, self._fpath))

                if self.pre_proc_flag:
                    fill_value = None
                    if hasattr(f.variables[dset], '_FillValue'):
                        fill_value = f.variables[dset]._FillValue
                    data[dset] = self.pre_process(
                        dset, f[dset][:].data, fill_value=fill_value,
                        sparse_mask=self._sparse_mask, index=self._index)
                else:
                    data[dset] = f[dset][:].data.ravel()

        return data


class CloudVar(AncillaryVarHandler):
    """Framework for cloud data extraction (GOES data processed by UW)."""

    def __init__(self, name, var_meta, date, source_dir=None, freq=None,
                 dsets=('cloud_type', 'cld_opd_dcomp', 'cld_reff_dcomp',
                        'cld_press_acha'), adjust_coords=False):
        """
        Parameters
        ----------
        name : str
            NSRDB var name.
        var_meta : str | pd.DataFrame | None
            CSV file or dataframe containing meta data for all NSRDB variables.
            Defaults to the NSRDB var meta csv in git repo.
        date : datetime.date
            Single day to extract data for.
        source_dir : str
            Cloud data directory containing nested daily directories with
            h5 or nc files from UW. Can be a normal directory path or
            /directory/prefix*suffix where /directory/ can have more sub dirs.
        freq : str | None
            Optional timeseries frequency to force cloud files to
            (time_index.freqstr). If None, the frequency of the cloud file
            list will be inferred.
        dsets : tuple | list
            Source datasets to extract. It is more efficient to extract all
            required datasets at once from each cloud file, so that only one
            kdtree is built for each unique coordinate set in each cloud file.
        adjust_coords : bool
            Flag to adjust cloud coordinates so clouds are assigned to the
            coordiantes they shade.
        """

        super().__init__(name, var_meta=var_meta, date=date,
                         source_dir=source_dir)

        x = self._parse_cloud_dir(self.source_dir)
        self._source_dir, self._prefix, self._suffix = x

        self._path = None
        self._freq = freq
        self._flist = None
        self._file_df = None
        self._dsets = dsets
        self._ftype = None
        self._i = None
        self._adjust_coords = adjust_coords

        if self._adjust_coords:
            logger.info('Cloud coordinate adjustment based on solar '
                        'position shading is active.')

        self._check_freq()

        if len(self.file_df) != len(self.flist):
            msg = ('Bad number of cloud data files for {}. Counted {} files '
                   'in {} but expected: {}'
                   .format(self._date, len(self.flist), self.path,
                           len(self.file_df)))
            warn(msg)
            logger.warning(msg)

    def __len__(self):
        """Length of this object is the number of source files."""
        return len(self.file_df)

    def __iter__(self):
        """Initialize this instance as an iter object."""
        self._i = 0
        logger.info('Iterating through {} cloud data {} files located in "{}"'
                    .format(len(self.file_df), self._ftype, self.path))
        return self

    def __next__(self):
        """Iterate through CloudVarSingle objects for each cloud data file.

        Returns
        -------
        timestamp : pd.Timestamp
            Timestamp from the datetime index.
        obj : CloudVarSingle | None
            Single cloud data retrieval object. None if there's a file missing.
        """

        # iterate through all timesteps (one file per timestep)
        if self._i < len(self.file_df):

            obj = None
            timestamp = self.file_df.index[self._i]
            fpath = self.file_df.iloc[self._i, 0]

            if not isinstance(fpath, str) or not os.path.exists(fpath):
                msg = ('Cloud data timestep {} is missing its '
                       'source file.'.format(timestamp))
                warn(msg)
                logger.warning(msg)

            else:
                # initialize a single timestep helper object
                if fpath.endswith('.h5'):
                    obj = CloudVarSingleH5(fpath, dsets=self._dsets,
                                           adjust_coords=self._adjust_coords)
                elif fpath.endswith('.nc'):
                    obj = CloudVarSingleNC(fpath, dsets=self._dsets,
                                           adjust_coords=self._adjust_coords)

                mem = psutil.virtual_memory()
                logger.info('Cloud data timestep {} has source file: {}. '
                            'Memory usage is {:.3f} GB out of '
                            '{:.3f} GB total.'
                            .format(timestamp, os.path.basename(fpath),
                                    mem.used / 1e9, mem.total / 1e9))

            self._i += 1
            return timestamp, obj

        else:
            raise StopIteration

    @staticmethod
    def _parse_cloud_dir(cloud_dir):
        """Parse a cloud directory string for cloud dir and prefix/suffix.

        Parameters
        ----------
        cloud_dir : str | None
            Complete directory path or /directory/prefix*suffix

        Returns
        -------
        cloud_dir : str | None
            Base directory path
        prefix : str | None
            File prefix if * in cloud_dir
        suffix : str | None
            File suffix if * in cloud_dir
        """

        prefix = None
        suffix = None

        if cloud_dir is None:
            return cloud_dir, prefix, suffix

        elif os.path.exists(cloud_dir):
            return cloud_dir, prefix, suffix

        elif os.path.exists(os.path.dirname(cloud_dir)):
            fbase = os.path.basename(cloud_dir)
            cloud_dir = os.path.dirname(cloud_dir)
            if '*' in fbase:
                prefix, suffix = fbase.split('*')
                return cloud_dir, prefix, suffix

        raise ValueError('Could not parse cloud dir: {}'.format(cloud_dir))

    def _check_freq(self):
        """Check the input vs inferred file frequency and warn if != """
        if self.freq.lower() != self.inferred_freq.lower():
            w = ('CloudVar handler has an input frequency of "{}" but '
                 'inferred a frequency of "{}" for directory: {}'
                 .format(self.freq, self.inferred_freq, self.path))
            logger.warning(w)
            warn(w)
        else:
            m = ('CloudVar handler has a frequency of "{}" for directory: {}'
                 .format(self.freq, self.path))
            logger.debug(m)

    @property
    def path(self):
        """Final path containing cloud data files.

        The path is searched in _source_dir based on the analysis date.

        Where _source_dir is defined in the nsrdb_vars.csv meta/config file.
        """

        if self._path is None:

            doy = str(self._date.timetuple().tm_yday).zfill(3)

            dirsearch = '/{}/'.format(doy)

            fsearch1 = '{}{}'.format(self._date.year, doy)
            fsearch2 = '{}_{}'.format(self._date.year, doy)
            fsearch3 = '{}{}{}'.format(self._date.year,
                                       str(self._date.month).zfill(2),
                                       str(self._date.day).zfill(2))
            fsearch = [fsearch1, fsearch2, fsearch3]

            # walk through current directory looking for day directory
            for dirpath, _, _ in os.walk(self._source_dir):
                dirpath = dirpath.replace('\\', '/')
                if not dirpath.endswith('/'):
                    dirpath += '/'
                if dirsearch in dirpath:
                    for fn in os.listdir(dirpath):
                        if any([tag in fn for tag in fsearch]):
                            self._path = dirpath
                            break
                if self._path is not None:
                    break

            if self._path is None:
                msg = ('Could not find cloud data dir for date {} in '
                       'source_dir {}. Looked for {} and any of: {}'
                       .format(self._date, self._source_dir, dirsearch,
                               fsearch))
                logger.exception(msg)
                raise IOError(msg)
            else:
                logger.debug('Cloud data dir for date {} found at: {}'
                             .format(self._date, self._path))

        return self._path

    def pre_flight(self):
        """Perform pre-flight checks - source dir check.

        Returns
        -------
        missing : str
            Look for the source dir and return the string if not found.
            If nothing is missing, return an empty string.
        """

        if self._source_dir is None:
            raise IOError('No cloud dir input for cloud var handler!')

        missing = ''
        if not os.path.exists(self._source_dir):
            missing = self._source_dir
        elif not os.path.exists(self.path):
            missing = self.path

        return missing

    @staticmethod
    def get_timestamp(fstr, integer=True):
        """Extract the cloud file timestamp.

        Parameters
        ----------
        fstr : str
            File path or file name with timestamp.
        integer : bool
            Flag to convert string match to integer.

        Returns
        -------
        time : int | str | None
            Integer timestamp of format: YYYYDDDHHMM (YYYY DDD HH MM)
            where DDD is day of year (1 through 366). None if not found
        """

        # YYYYDDDHHMMSSS
        match_nc = re.match(r".*s([1-2][0-9]{13})", fstr)

        # YYYY_DDD_HHMM
        match_h5 = re.match(r".*([1-2][0-9]{3}_[0-9]{3}_[0-9]{4})", fstr)

        # YYYYMMDD_HHMM_
        s = r".*([1-2][0-9]{3}[0-1][0-9]{3}_[0-2][0-9][0-5][0-9]_)"
        match_himawari = re.match(s, fstr)

        if match_nc:
            time = match_nc.group(1)
        elif match_h5:
            time = match_h5.group(1).replace('_', '')
        elif match_himawari:
            time = str(match_himawari.group(1).replace('_', ''))
            date = datetime.date(year=int(time[0:4]),
                                 month=int(time[4:6]),
                                 day=int(time[6:8]))
            doy = date.timetuple().tm_yday
            time = time[0:4] + str(doy).zfill(3) + time[8:]
        else:
            time = None

        if time is not None:
            time = time[:11]

        if time is not None and integer:
            time = int(time)

        return time

    @staticmethod
    def get_h5_flist(path, date, prefix=None, suffix=None):
        """Get the .h5 cloud data file path list.

        Parameters
        ----------
        path : str
            Terminal directory containing .h5 files.
        date : datetime.date
            Date of files to look for.
        prefix : str | None
            File prefix to look for in filenames or None
        suffix : str | None
            File suffix to look for in filenames or None

        Returns
        -------
        flist : list
            List of full file paths sorted by timestamp. Empty list if no
            files found.
        """

        fl = os.listdir(path)
        if prefix is not None:
            fl = [fn for fn in fl if fn.startswith(prefix)]
        if suffix is not None:
            fl = [fn for fn in fl if fn.endswith(suffix)]
        flist = [os.path.join(path, f) for f in fl
                 if f.endswith('.h5') and str(date.year) in f]

        if flist:
            # sort by timestep after the last underscore before .level2.h5
            flist = sorted(flist, key=CloudVar.get_timestamp)
        return flist

    @staticmethod
    def get_nc_flist(path, date, prefix=None, suffix=None):
        """Get the .nc cloud data file path list.

        Parameters
        ----------
        path : str
            Terminal directory containing .nc files.
        date : datetime.date
            Date of files to look for.
        prefix : str | None
            File prefix to look for in filenames or None
        suffix : str | None
            File suffix to look for in filenames or None

        Returns
        -------
        flist : list
            List of full file paths sorted by timestamp. Empty list if no
            files found.
        """

        fl = os.listdir(path)
        if prefix is not None:
            fl = [fn for fn in fl if fn.startswith(prefix)]
        if suffix is not None:
            fl = [fn for fn in fl if fn.endswith(suffix)]
        flist = [os.path.join(path, f) for f in fl
                 if f.endswith('.nc') and str(date.year) in f]

        if flist:
            # sort by timestep after the last underscore before .level2.h5
            flist = sorted(flist, key=CloudVar.get_timestamp)
        return flist

    @property
    def flist(self):
        """List of cloud data file paths for one day. Each file is a timestep.

        Note that this is the raw parsed file list, which may not match
        self.file_df DataFrame, which is the final file list based on desired
        timestep frequency

        Returns
        -------
        flist : list
            List of .h5 or .nc full file paths sorted by timestamp. Exception
            raised if no files are found.
        """

        if self._flist is None:
            self._flist = self.get_h5_flist(self.path, self._date,
                                            prefix=self._prefix,
                                            suffix=self._suffix)
            self._ftype = '.h5'

            if not self._flist:
                self._flist = self.get_nc_flist(self.path, self._date,
                                                prefix=self._prefix,
                                                suffix=self._suffix)
                self._ftype = '.nc'

            if not self._flist:
                raise IOError('Could not find .h5 or .nc files for {} '
                              'with prefix "{}" and suffix "{}" in '
                              'directory: {}'
                              .format(self._date, self._prefix, self._suffix,
                                      self.path))

            self._flist = self._remove_bad_files(self._flist)

        return self._flist

    @staticmethod
    def _remove_bad_files(flist):
        """Parse the filelist and remove any filepaths less than 1MB.

        Parameters
        ----------
        flist : list
            List of .h5 or .nc full file paths to clean.

        Returns
        -------
        flist : list
            List of .h5 or .nc full file paths with bad files removed.
        """

        for fp in flist:
            if os.path.getsize(fp) < 1e6:
                msg = ('Cloud data source file is less than 1MB, skipping: {}'
                       .format(fp))
                warn(msg)
                logger.warning(msg)

        flist = [fp for fp in flist if os.path.getsize(fp) > 1e6]

        return flist

    @property
    def inferred_freq(self):
        """Get the inferred frequency from the file list.

        Returns
        -------
        freq : str
            Pandas datetime frequency.
        """
        return self.infer_data_freq(self.flist)

    @property
    def freq(self):
        """Get the file list timeseries frequency.

        Is forced if this object is initialized with freq != None.
        Otherwise, inferred from file list.

        Returns
        -------
        freq : str
            Nominal pandas datetimeindex frequency of the cloud file list.
        """
        if self._freq is None:
            self._freq = self.inferred_freq
        if len(self._freq) == 1:
            self._freq = '1{}'.format(self._freq)
        return self._freq

    @property
    def file_df(self):
        """Get a dataframe with nominal time index and available cloud files.

        Returns
        -------
        _file_df : pd.DataFrame
            Timeseries of available cloud file paths. The datetimeindex is
            created by the infered timestep frequency of the cloud files.
            The data column is the file paths. Timesteps with missing data
            files has NaN file paths.
        """

        if self._file_df is None:
            # actual file list with actual time index from file timestamps
            data_ti = self.infer_data_time_index(self.flist)
            df_actual = pd.DataFrame({'flist': self.flist}, index=data_ti)

            # nominal time index based on inferred or input freq
            nominal_index = self._get_time_index(self._date, freq=self.freq)
            df_nominal = pd.DataFrame(index=nominal_index)
            tolerance = pd.Timedelta(self.freq) / 2

            logger.debug('Using a file to timestep matching tolerance of {}'
                         .format(tolerance))

            self._file_df = pd.merge_asof(df_nominal, df_actual,
                                          left_index=True, right_index=True,
                                          direction='nearest',
                                          tolerance=tolerance)
        return self._file_df

    @staticmethod
    def infer_data_time_index(flist):
        """Get the actual time index of the file set based on the timestamps.

        Parameters
        ----------
        flist : list
            List of strings of cloud files (with or without full file path).

        Returns
        -------
        time_index : pd.datetimeindex
            Pandas datetime index based on the actual file timestamps.
        """

        strtime = [CloudVar.get_timestamp(fstr, integer=False)
                   for fstr in flist]
        time_index = pd.to_datetime(strtime, format='%Y%j%H%M')

        return time_index

    @staticmethod
    def infer_data_freq(flist):
        """Infer the cloud data timestep frequency from the file list.

        Parameters
        ----------
        flist : list
            List of strings of cloud files (with or without full file path).

        Returns
        -------
        freq : str
            Pandas datetime frequency.
        """

        data_ti = CloudVar.infer_data_time_index(flist)

        if len(flist) == 1:
            freq = '1d'
        else:
            ti_deltas = data_ti - np.roll(data_ti, 1)
            ti_deltas_minutes = ti_deltas.seconds / 60

            if len(flist) <= 3:
                ti_delta_minutes = int(ti_deltas_minutes[0])
                freq = '{}T'.format(ti_delta_minutes)
            else:
                ti_delta_minutes = int(statistics.mode(ti_deltas_minutes))
                freq = '{}T'.format(ti_delta_minutes)
                if len(flist) < 5:
                    w = ('File list contains less than 5 files. Inferred '
                         'frequency of "{}", but may not be accurate'
                         .format(freq))
                    logger.warning(w)
                    warn(w)

        return freq

    @property
    def time_index(self):
        """Get the GOES cloud data time index.

        Returns
        -------
        cloud_ti : pd.DatetimeIndex
            Pandas datetime index for the current day at the cloud temporal
            resolution (should match the NSRDB resolution).
        """

        return self.file_df.index
