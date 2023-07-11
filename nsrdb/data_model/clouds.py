# -*- coding: utf-8 -*-
"""A framework for handling UW/GOES source data."""
import datetime
import logging
import os
import re
from warnings import warn

import numpy as np
import pandas as pd
import psutil
from farms import CLOUD_TYPES
from scipy.spatial import cKDTree
from scipy.stats import mode

from nsrdb.data_model.base_handler import AncillaryVarHandler
from nsrdb.file_handlers.file_system import NSRDBFileSystem as NFS

logger = logging.getLogger(__name__)


class CloudCoords:
    """Class to correct cloud coordinates based on parallax correction and
    also solar position / shading."""

    EARTH_RADIUS = 6371

    # default NADIR longitudes for sensor azimuth calculation. GOES West is now
    # at -137 but this should have very minor impact on azimuth angle.
    DEFAULT_NADIR_E = -75
    DEFAULT_NADIR_W = -135

    REQUIRED = ('latitude',
                'longitude',
                'solar_zenith_angle',
                'solar_azimuth_angle',
                'sensor_zenith_angle',
                'cld_height_acha',
                )

    @staticmethod
    def check_file(fp):
        """Check if file has required vars for cloud coord correction"""

        if fp.endswith(('.nc', '.nc4')):
            with NFS(fp) as f:
                dsets = sorted(list(f.variables.keys()))

        elif fp.endswith('.h5'):
            with NFS(fp, use_h5py=True) as f:
                dsets = list(f)

        else:
            e = ('Could not parse cloud file, expecting .h5 or .nc but '
                 'received: {}'.format(os.path.basename(fp)))
            logger.error(e)
            raise OSError(e)

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
        delta_lat = np.degrees(dist / CloudCoords.EARTH_RADIUS)

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
        r = CloudCoords.EARTH_RADIUS * np.cos(np.radians(latitude))
        delta_lon = np.degrees(dist / r)

        return delta_lon

    @classmethod
    def calc_sensor_azimuth(cls, lat, lon, sen_zen):
        """Calculate an array of sensor azimuth angles given observed lat/lon
        arrays and an array of sensor zenith angles. NADIR is calculated based
        on the minimum zenith angle and the corresponding longitude, unless the
        minimum zenith angle is >5deg in which case the NADIR is taken from the
        default east/west class attributes based on the average longitude

        This is necessary because old cloud files dont have the
        sensor_azimuth_angle dataset.

        This is based on the equations in the reference:
        https://keisan.casio.com/exec/system/1224587128#mistake

        Parameters
        ----------
        lat : np.ndarray
            Latitude values in decimal degrees
        lon : np.ndarray
            Longitude values in decimal degrees
        sen_zen : np.ndarray
            Sensor zenith angle for every lat/lon value for one timestep
            in degrees.

        Returns
        -------
        sen_azi : np.ndarray
            Senzor azimuth angle from -180 to +180 in degrees. Array has the
            same shape as input lat/lon/sen_zen.
        """

        nadir_lat = 0.0
        min_zen = np.abs(np.nanmin(sen_zen))
        if min_zen > 5:
            diff_w = np.abs(np.nanmean(lon) - cls.DEFAULT_NADIR_W)
            diff_e = np.abs(np.nanmean(lon) - cls.DEFAULT_NADIR_E)
            if diff_w < diff_e:
                nadir_lon = cls.DEFAULT_NADIR_W
            else:
                nadir_lon = cls.DEFAULT_NADIR_E
        else:
            nadir_lon = lon[np.argmin(sen_zen)]

        lat = np.radians(lat)
        lon = np.radians(lon)
        nadir_lat = np.radians(nadir_lat)
        nadir_lon = np.radians(nadir_lon)

        dx = nadir_lon - lon
        atan_x1 = np.cos(lat) * np.tan(nadir_lat) - np.sin(lat) * np.cos(dx)
        atan_x2 = np.sin(dx)
        sen_azi = np.arctan2(atan_x1, atan_x2)
        sen_azi = 90 - np.degrees(sen_azi)
        sen_azi[(sen_azi > 180)] -= 360

        return sen_azi

    @classmethod
    def correct_coords(cls, lat, lon, zen, azi, cld_height, zen_threshold=85,
                       option='parallax'):
        """Adjust cloud coordinates for parallax correction using the viewing
        geometry from the sensor or for shading geometry based on the sun's
        position. Height data for clearsky pixels should be NaN or zero, which
        will return un-manipulated lat/lon values.

        Parameters
        ----------
        lat : np.ndarray
            Latitude values in decimal degrees
        lon : np.ndarray
            Longitude values in decimal degrees
        zen : np.ndarray
            Sensor or solar zen angle for every lat/lon value for one timestep
            in degrees.
        azi : np.ndarray
            Sensor or solar azimuth angle for every lat/lon value for one
            timestep in degrees.
        cld_height : np.ndarray
            Cloud height in km. Clearsky pixels should have
            (cld_height == np.nan | 0) which will return un-manipulated lat/lon
            values.
        zen_threshold : float | int
            Thresold over which coordinate adjustments are truncated.
            Coordinate solar shading adjustments approach infinity at a solar
            zenith angle of 90.
        option : str
            Either "parallax" or "shading".

        Returns
        -------
        lat : np.ndarray
            Latitude values in decimal degrees adjusted for either A) parallax
            correction based on the viewing geometry from the sensor (option ==
            "parallax") or B) the sun position so that clouds are mapped to the
            coordinates they are shading (option == "shading").
        lon : np.ndarray
            Longitude values in decimal degrees adjusted for either A) parallax
            correction based on the viewing geometry from the sensor (option ==
            "parallax") or B) the sun position so that clouds are mapped to the
            coordinates they are shading (option == "shading").
        """

        shapes = {'lat': lat.shape, 'lon': lon.shape,
                  'zen': zen.shape, 'azi': azi.shape,
                  'cld_height': cld_height.shape}
        shapes_list = list(shapes.values())
        check = [shapes_list[0] == x for x in shapes_list[1:]]
        if not all(check):
            e = ('Cannot run cloud coordinate shading adjustment. '
                 'Input shapes: {}'.format(shapes))
            logger.error(e)
            raise ValueError(e)

        # if the maximum cloud height is 100+
        # assume units are in meters and convert to km
        # cloud heights should never be >100km
        cld_height[(cld_height < 0)] = np.nan
        if np.nanmax(cld_height) > 100:
            cld_height /= 1000

        zen[(zen > zen_threshold)] = zen_threshold

        zen = np.radians(zen)
        azi = np.radians(azi)

        delta_dist = cld_height * np.tan(zen)
        delta_x = delta_dist * np.sin(azi)
        delta_y = delta_dist * np.cos(azi)

        if option.lower() == 'shading':
            delta_x *= -1
            delta_y *= -1

        delta_lon = cls.dist_to_longitude(lat, delta_x)
        delta_lat = cls.dist_to_latitude(delta_y)

        delta_lon[np.isnan(delta_lon)] = 0.0
        delta_lat[np.isnan(delta_lat)] = 0.0

        lon += delta_lon
        lat += delta_lat

        return lat, lon


class CloudVarSingle:
    """Base framework for single-file/single-timestep cloud data extraction."""

    GRID_LABELS = ['latitude', 'longitude']

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
        self._available_dsets = None
        self._grid = None
        self._tree = None

        # attributes for remapping pc data to raw coordinates
        self._raw_grid = None
        self._raw_cloud_mask = None
        self._remap_pc_index = None
        self._remap_pc_index_clouds = None

    def __contains__(self, dset):
        return dset in self.dsets

    def __repr__(self):
        return 'CloudVarSingle handler for filepath: "{}"'.format(self.fpath)

    def __str__(self):
        return 'CloudVarSingle handler for filepath: "{}"'.format(self.fpath)

    def get_dset(self, dset):
        """Abstract placeholder for data retrieval method"""
        raise NotImplementedError('get_dset() must be defined for H5 or NC '
                                  'file types.')

    @property
    def dsets(self):
        """Get a list of the available datasets in the cloud file."""
        return self._available_dsets

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
        for dset in self._dsets:
            # pylint: disable=assignment-from-no-return
            data[dset] = self.get_dset(dset)

        return data

    @property
    def tree(self):
        """Get the KDTree for the cloud data coordinates
        eg. cKDTree(self.grid[['latitude', 'longitude']])

        Returns
        -------
        cKDTree
        """
        if self._tree is None:
            # pylint: disable=not-callable,unsubscriptable-object
            self._tree = cKDTree(self.grid[self.GRID_LABELS])
        return self._tree

    def remap_pc_coords(self):
        """Remap the parallax/shading corrected coordinates back onto the
        original "raw" coordinate system and set internal variables to do the
        same for the cloud data when processed through get_dset() and
        self.source_data
        """

        # reset the self._grid attribute to be the raw non-pc grid
        # (the desired final cloud coordinate system)
        pc_grid = self._grid.copy()
        self._grid = self._raw_grid

        self._raw_cloud_mask = np.isin(self.get_dset('cloud_type'),
                                       CLOUD_TYPES)

        # this fills in all gaps in the original raw coordinate system created
        # by the parallax correction using simple nearest neighbor
        # pylint: disable=not-callable
        pc_tree = cKDTree(pc_grid[self.GRID_LABELS])
        self._remap_pc_index = pc_tree.query(self._grid.values)[1]
        self._remap_pc_index = self._remap_pc_index.astype(np.int32)

        # this applies the parallax corrected cloud coordinates on top of the
        # original raw coordinate system overriding any existing clear
        # conditions
        self._remap_pc_index_clouds = self.tree.query(
            pc_grid.values[self._raw_cloud_mask])[1].astype(np.int32)

        # try to reduce memory usage
        del pc_tree
        del pc_grid

    def remap_pc_data(self, data):
        """Perform remapping of parallax/shading corrected data onto the
        raw/original cloud coordinate system including overlaying cloud shadow
        data over clear data.

        Parameters
        ----------
        data : np.ndarray
            1D array of flattened data based on the original coordinate system
            ordering from the cloud file, possibly with sparsification due to
            pre processing of nan data/coordinates.

        Returns
        -------
        data : np.ndarray
            1D array of flattened data that corresponds to the original
            coordinate system with no parallax/shading corrections but has been
            re-arranged such that it reflects these coordinate adjustments.
        """

        if self._remap_pc_index is not None:
            raw_data = data.copy()

            # this fills in all gaps in the original raw coordinate system
            # created by the parallax correction using simple nearest neighbor
            data = raw_data[self._remap_pc_index]

            # this applies the parallax corrected cloud coordinates on top of
            # the original raw coordinate system overriding any existing clear
            # conditions
            data[self._remap_pc_index_clouds] = raw_data[self._raw_cloud_mask]

        return data

    @staticmethod
    def _clean_dup_coords(grid):
        """Clean a cloud coordinate dataframe (grid) by manipulating
        duplicate coordinate values.

        Parameters
        ----------
        grid : pd.DataFrame
            Grid dataframe with latitude and longitude columns and possibly
            some duplicate coordinates.

        Returns
        -------
        grid : pd.DataFrame
            Grid dataframe with latitude and longitude columns and no duplicate
            coordinates.
        """

        if grid is None:
            return grid

        dup_mask = grid.duplicated() & ~grid['latitude'].isna()
        if any(dup_mask):
            rand_mult = np.random.uniform(0.99, 1.01, dup_mask.sum())
            grid.loc[dup_mask, 'latitude'] *= rand_mult
            rand_mult = np.random.uniform(0.99, 1.01, dup_mask.sum())
            grid.loc[dup_mask, 'longitude'] *= rand_mult

            wmsg = ('Cloud file had {} duplicate coordinates out of {} '
                    '({:.2f}%)'.format(dup_mask.sum(), len(grid),
                                       100 * dup_mask.sum() / len(grid)))
            warn(wmsg)
            logger.warning(wmsg)

        return grid

    def clean_attrs(self):
        """Try to clean unnecessary object attributes to reduce memory usage
        """
        self._tree = None
        self._raw_grid = None
        self._raw_cloud_mask = None
        self._remap_pc_index = None
        self._remap_pc_index_clouds = None


class CloudVarSingleH5(CloudVarSingle):
    """Framework for .h5 single-file/single-timestep cloud data extraction."""

    def __init__(self, fpath, pre_proc_flag=True, index=None,
                 dsets=('cloud_type', 'cld_opd_dcomp', 'cld_reff_dcomp',
                        'cld_press_acha'),
                 parallax_correct=True, solar_shading=True, remap_pc=True):
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
        parallax_correct : bool
            Flag to adjust cloud coordinates so clouds are overhead their
            coordinates and not at the apparent location from the sensor.
        solar_shading : bool
            Flag to adjust cloud coordinates so clouds are assigned to the
            coordiantes they shade.
        remap_pc : bool
            Flag to remap the parallax-corrected and solar-shading-corrected
            data back onto the original semi-regular GOES coordinates
        """

        super().__init__(fpath, pre_proc_flag=pre_proc_flag, index=index,
                         dsets=dsets)

        grids = self._parse_grid(self._fpath,
                                 solar_shading=solar_shading,
                                 parallax_correct=parallax_correct,
                                 pre_proc_flag=pre_proc_flag)
        self._grid, self._raw_grid = grids

        if self.pre_proc_flag:
            self._grid, self._raw_grid, self._sparse_mask = self.make_sparse(
                self._grid, self._raw_grid)

        if remap_pc and (parallax_correct or solar_shading):
            self.remap_pc_coords()

    @property
    def dsets(self):
        """Get a list of the available datasets in the cloud file."""

        if self._available_dsets is None:
            with NFS(self._fpath, use_h5py=True) as f:
                self._available_dsets = list(f)

        return self._available_dsets

    @classmethod
    def _parse_grid(cls, fpath, parallax_correct=True, solar_shading=True,
                    pre_proc_flag=True):
        """Extract the cloud data grid for the current timestep.

        Parameters
        ----------
        fpath : str
            Full file path to netcdf4 file.
        parallax_correct : bool
            Flag to adjust cloud coordinates so clouds are overhead their
            coordinates and not at the apparent location from the sensor.
        solar_shading : bool
            Flag to adjust cloud coordinates so clouds are assigned to the
            coordiantes they shade.
        pre_proc_flag : bool
            Flag to ensure there are no duplicate coordinates in grid.

        Returns
        -------
        grid : pd.DataFrame | None
            GOES source coordinates (labels: ['latitude', 'longitude']).
            None if bad dataset
        raw_grid : pd.DataFrame | None
            If parallax correction or solar shading is active, this object is
            set as the raw/original grid before pc/shading. Otherwise, None.
        """

        raw_grid = None
        grid = {}
        with NFS(fpath, use_h5py=True) as f:
            for dset in cls.GRID_LABELS:

                if dset not in list(f):
                    msg = ('Could not find "{}" in file: "{}"'
                           .format(dset, fpath))
                    logger.error(msg)
                    raise KeyError(msg)

                grid[dset] = cls.pre_process(dset, f[dset][...],
                                             dict(f[dset].attrs))

        if grid and (parallax_correct or solar_shading):
            raw_grid = pd.DataFrame(grid)
            grid = cls.correct_coordinates(fpath, grid,
                                           parallax_correct=parallax_correct,
                                           solar_shading=solar_shading)

        grid = pd.DataFrame(grid)

        if grid.empty:
            raw_grid = None
            grid = None

        elif pre_proc_flag:
            grid = cls._clean_dup_coords(grid)
            raw_grid = cls._clean_dup_coords(raw_grid)

        return grid, raw_grid

    @classmethod
    def correct_coordinates(cls, fpath, grid, parallax_correct=True,
                            solar_shading=True):
        """Adjust grid lat/lon values based on solar position

        Parameters
        ----------
        fpath : str
            Filepath to cloud h5 file containing required datasets for solpo
            coodinate adjustment.
        grid : dict
            Dictionary with latitude and longitude keys and corresponding
            numpy array values.
        parallax_correct : bool
            Flag to adjust cloud coordinates so clouds are overhead their
            coordinates and not at the apparent location from the sensor.
        solar_shading : bool
            Flag to adjust cloud coordinates so clouds are assigned to the
            coordiantes they shade.

        Returns
        -------
        grid : dict
            Dictionary with latitude and longitude keys and corresponding
            numpy array values. Coordinates are adjusted for solar position
            so that clouds are linked to the coordinate that they are shading.
        """
        lat, lon = grid['latitude'], grid['longitude']
        with NFS(fpath, use_h5py=True) as f:
            missing = [d for d in CloudCoords.REQUIRED if d not in f]
            if any(missing):
                msg = ('Could not correct cloud coordinates, missing datasets '
                       '{} from source file: {}'.format(missing, fpath))
                logger.error(msg)
                raise KeyError(msg)

            sen_zen = cls.pre_process(
                'sensor_zenith_angle', f['sensor_zenith_angle'][...],
                dict(f['sensor_zenith_angle'].attrs))
            sol_zen = cls.pre_process(
                'solar_zenith_angle', f['solar_zenith_angle'][...],
                dict(f['solar_zenith_angle'].attrs))
            sol_azi = cls.pre_process(
                'solar_azimuth_angle', f['solar_azimuth_angle'][...],
                dict(f['solar_azimuth_angle'].attrs))
            cld_height = cls.pre_process(
                'cld_height_acha', f['cld_height_acha'][...],
                dict(f['cld_height_acha'].attrs))

            if 'sensor_azimuth_angle' in f:
                sen_azi = cls.pre_process(
                    'sensor_azimuth_angle', f['sensor_azimuth_angle'][...],
                    dict(f['sensor_azimuth_angle'].attrs))
            else:
                sen_azi = CloudCoords.calc_sensor_azimuth(lat, lon, sen_zen)

        try:
            if parallax_correct:
                logger.debug('Running sensor parallax correction.')
                sen_azi = CloudCoords.calc_sensor_azimuth(lat, lon, sen_zen)
                lat, lon = CloudCoords.correct_coords(lat, lon, sen_zen,
                                                      sen_azi, cld_height,
                                                      option='parallax')
            if solar_shading:
                logger.debug('Running cloud shading coordinate correction.')
                lat, lon = CloudCoords.correct_coords(lat, lon, sol_zen,
                                                      sol_azi, cld_height,
                                                      option='shading')
            grid['latitude'], grid['longitude'] = lat, lon

        except Exception as e:
            logger.warning('Could not perform cloud coordinate adjustment '
                           'for: {}, received error: {}'
                           .format(os.path.basename(fpath), e))

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
    def make_sparse(grid, raw_grid):
        """Make the cloud grid sparse by removing NaN coordinates.

        Parameters
        ----------
        grid : pd.DataFrame
            GOES source coordinates (labels: ['latitude', 'longitude']).
        raw_grid : pd.DataFrame | None
            Raw GOES source coordinates before parallax correction / solar
            shading or None if those algorithms are disabled.

        Returns
        -------
        grid : pd.DataFrame
            Sparse GOES source coordinates with all NaN rows removed.
        raw_grid : pd.DataFrame | None
            Raw GOES source coordinates before parallax correction / solar
            shading or None if those algorithms are disabled.
        mask : pd.Series
            Boolean series; the mask to extract sparse data.
        """

        mask = (grid['latitude'] == -90) & (grid['longitude'] == -180)
        grid.loc[mask, :] = np.nan
        mask = ~(pd.isna(grid['latitude']) | pd.isna(grid['longitude']))
        grid = grid[mask]

        if raw_grid is not None:
            raw_grid = raw_grid[mask]

        return grid, raw_grid, mask

    def get_dset(self, dset):
        """Get a single dataset from the source cloud data file.

        Parameters
        ----------
        dset : str
            Variable dataset name to retrieve from the cloud file.

        Returns
        -------
        dset : np.ndarray
            1D array of flattened data that should match the self.grid meta
            data.
        """

        with NFS(self._fpath, use_h5py=True) as f:
            if dset not in list(f):
                raise KeyError('Could not find "{}" in the cloud file: {}'
                               .format(dset, self._fpath))

            if self.pre_proc_flag:
                data = self.pre_process(
                    dset, f[dset][...], dict(f[dset].attrs),
                    sparse_mask=self._sparse_mask, index=self._index)
            else:
                data = f[dset][...].ravel()

        # will not remap if proper internal attributes are not present
        data = self.remap_pc_data(data)

        return data


class CloudVarSingleNC(CloudVarSingle):
    """Framework for .nc single-file/single-timestep cloud data extraction."""

    def __init__(self, fpath, pre_proc_flag=True, index=None,
                 dsets=('cloud_type', 'cld_opd_dcomp', 'cld_reff_dcomp',
                        'cld_press_acha'),
                 parallax_correct=True, solar_shading=True, remap_pc=True):
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
        parallax_correct : bool
            Flag to adjust cloud coordinates so clouds are overhead their
            coordinates and not at the apparent location from the sensor.
        solar_shading : bool
            Flag to adjust cloud coordinates so clouds are assigned to the
            coordiantes they shade.
        remap_pc : bool
            Flag to remap the parallax-corrected and solar-shading-corrected
            data back onto the original semi-regular GOES coordinates
        """

        super().__init__(fpath, pre_proc_flag=pre_proc_flag, index=index,
                         dsets=dsets)

        self._grid, self._raw_grid, self._sparse_mask = self._parse_grid(
            self._fpath, parallax_correct=parallax_correct,
            solar_shading=solar_shading,
            pre_proc_flag=pre_proc_flag)

        if remap_pc and (parallax_correct or solar_shading):
            self.remap_pc_coords()

    @property
    def dsets(self):
        """Get a list of the available datasets in the cloud file."""

        if self._available_dsets is None:
            with NFS(self._fpath) as f:
                self._available_dsets = list(f.variables.keys())

        return self._available_dsets

    @classmethod
    def _parse_grid(cls, fpath, parallax_correct=True, solar_shading=True,
                    pre_proc_flag=True):
        """Extract the cloud data grid for the current timestep.

        Parameters
        ----------
        fpath : str
            Full file path to netcdf4 file.
        parallax_correct : bool
            Flag to adjust cloud coordinates so clouds are overhead their
            coordinates and not at the apparent location from the sensor.
        solar_shading : bool
            Flag to adjust cloud coordinates so clouds are assigned to the
            coordiantes they shade.
        pre_proc_flag : bool
            Flag to ensure there are no duplicate coordinates in grid.

        Returns
        -------
        grid : pd.DataFrame | None
            GOES source coordinates (labels: ['latitude', 'longitude']).
            None if bad dataset
        raw_grid : pd.DataFrame | None
            If parallax correction or solar shading is active, this object is
            set as the raw/original grid before pc/shading. Otherwise, None.
        mask : np.ndarray
            2D boolean array to extract good data.
        """

        sparse_mask = None
        raw_grid = None
        grid = {}
        with NFS(fpath) as f:
            for dset in cls.GRID_LABELS:

                if dset not in f.variables.keys():
                    msg = ('Could not find "{}" in file: "{}"'
                           .format(dset, fpath))
                    logger.error(msg)
                    raise KeyError(msg)

                # use netCDF masked array mask to reduce ~1/4 of the data
                if sparse_mask is None:
                    try:
                        sparse_mask = ~f[dset][:].mask
                    except Exception as e:
                        msg = ('Exception masking {} in {}: {}'
                               .format(dset, fpath, e))
                        logger.error(msg)
                        raise RuntimeError(msg) from e

                if not isinstance(sparse_mask, np.ndarray):
                    sparse_mask = np.full(f[dset][:].data.shape, sparse_mask)

                grid[dset] = f[dset][:].data[sparse_mask]

        if grid and (parallax_correct or solar_shading):
            raw_grid = pd.DataFrame(grid)
            grid = cls.correct_coordinates(fpath, grid, sparse_mask,
                                           parallax_correct=parallax_correct,
                                           solar_shading=solar_shading)

        grid = pd.DataFrame(grid)

        if grid.empty:
            raw_grid = None
            grid = None

        elif pre_proc_flag:
            grid = cls._clean_dup_coords(grid)
            raw_grid = cls._clean_dup_coords(raw_grid)

        if sparse_mask.sum() == 0:
            msg = ('Cloud data handler had a completely empty sparse mask '
                   'for: {}'.format(fpath))
            logger.warning(msg)
            warn(msg)

        return grid, raw_grid, sparse_mask

    @staticmethod
    def correct_coordinates(fpath, grid, sparse_mask, parallax_correct=True,
                            solar_shading=True):
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
        parallax_correct : bool
            Flag to adjust cloud coordinates so clouds are overhead their
            coordinates and not at the apparent location from the sensor.
        solar_shading : bool
            Flag to adjust cloud coordinates so clouds are assigned to the
            coordiantes they shade.

        Returns
        -------
        grid : dict
            Dictionary with latitude and longitude keys and corresponding
            numpy array values. Coordinates are adjusted for solar position
            so that clouds are linked to the coordinate that they are shading.
        """
        lat, lon = grid['latitude'], grid['longitude']
        with NFS(fpath) as f:
            missing = [d for d in CloudCoords.REQUIRED if d not in f.variables]
            if any(missing):
                msg = ('Could not correct cloud coordinates, missing datasets '
                       '{} from source file: {}'.format(missing, fpath))
                logger.error(msg)
                raise KeyError(msg)

            sen_zen = f['sensor_zenith_angle'][:].data[sparse_mask]
            sol_zen = f['solar_zenith_angle'][:].data[sparse_mask]
            sol_azi = f['solar_azimuth_angle'][:].data[sparse_mask]
            cld_height = f['cld_height_acha'][:].data[sparse_mask]

            if 'sensor_azimuth_angle' in f.variables:
                sen_azi = f['sensor_azimuth_angle'][:].data[sparse_mask]
            else:
                sen_azi = CloudCoords.calc_sensor_azimuth(lat, lon, sen_zen)

        try:
            if parallax_correct:
                lat, lon = CloudCoords.correct_coords(lat, lon, sen_zen,
                                                      sen_azi, cld_height,
                                                      option='parallax')
            if solar_shading:
                lat, lon = CloudCoords.correct_coords(lat, lon, sol_zen,
                                                      sol_azi, cld_height,
                                                      option='shading')
            grid['latitude'], grid['longitude'] = lat, lon

        except Exception as e:
            logger.warning('Could not perform cloud coordinate adjustment '
                           'for: {}, received error: {}'
                           .format(os.path.basename(fpath), e))

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

    def get_dset(self, dset):
        """Get a single dataset from the source cloud data file.

        Parameters
        ----------
        dset : str
            Variable dataset name to retrieve from the cloud file.

        Returns
        -------
        dset : np.ndarray
            1D array of flattened data that should match the self.grid meta
            data.
        """
        with NFS(self._fpath) as f:
            if dset not in list(f.variables.keys()):
                raise KeyError('Could not find "{}" in the cloud file: {}'
                               .format(dset, self._fpath))

            if self.pre_proc_flag:
                fill_value = None
                if hasattr(f.variables[dset], '_FillValue'):
                    fill_value = f.variables[dset]._FillValue

                data = self.pre_process(
                    dset, f[dset][:].data, fill_value=fill_value,
                    sparse_mask=self._sparse_mask, index=self._index)
            else:
                data = f[dset][:].data.ravel()

        # will not remap if proper internal attributes are not present
        data = self.remap_pc_data(data)

        return data


class CloudVar(AncillaryVarHandler):
    """Framework for cloud data extraction (GOES data processed by UW)."""

    def __init__(self, name, var_meta, date, freq=None,
                 dsets=('cloud_type', 'cld_opd_dcomp', 'cld_reff_dcomp',
                        'cld_press_acha'),
                 parallax_correct=True, solar_shading=True, remap_pc=True,
                 **kwargs):
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
        freq : str | None
            Optional timeseries frequency to force cloud files to
            (time_index.freqstr). If None, the frequency of the cloud file
            list will be inferred.
        dsets : tuple | list
            Source datasets to extract. It is more efficient to extract all
            required datasets at once from each cloud file, so that only one
            kdtree is built for each unique coordinate set in each cloud file.
        parallax_correct : bool
            Flag to adjust cloud coordinates so clouds are overhead their
            coordinates and not at the apparent location from the sensor.
        solar_shading : bool
            Flag to adjust cloud coordinates so clouds are assigned to the
            coordiantes they shade.
        remap_pc : bool
            Flag to remap the parallax-corrected and solar-shading-corrected
            data back onto the original semi-regular GOES coordinates
        """

        super().__init__(name, var_meta=var_meta, date=date, **kwargs)

        self._freq = freq
        self._flist = None
        self._file_df = None
        self._dsets = dsets
        self._i = None
        self._parallax_correct = parallax_correct
        self._solar_shading = solar_shading
        self._remap_pc = remap_pc
        self._obj_cache = {}

        logger.info('Cloud coordinate parallax correction: {}, solar '
                    'shading adjustment: {}, coordinate remapping: {}'
                    .format(parallax_correct, solar_shading, remap_pc))

        self._check_freq()

        if len(self.file_df) != len(self.flist):
            msg = ('Bad number of cloud data files for {}. Counted {} files '
                   'in {} but expected: {}'
                   .format(self._date, len(self.flist), self.pattern,
                           len(self.file_df)))
            warn(msg)
            logger.warning(msg)

    def __len__(self):
        """Length of this object is the number of source files."""
        return len(self.file_df)

    def __iter__(self):
        """Initialize this instance as an iter object."""
        self._i = 0
        logger.info('Iterating through {} cloud data files located in "{}"'
                    .format(len(self.file_df), self.pattern))

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

            if fpath in self._obj_cache:
                obj = self._obj_cache[fpath]
                logger.debug('Found cached object {}'.format(obj))

            elif not isinstance(fpath, str) or not NFS(fpath).exists():
                msg = ('Cloud data timestep {} is missing its '
                       'source file.'.format(timestamp))
                warn(msg)
                logger.warning(msg)

            else:
                # initialize a single timestep helper object
                obj = self.get_handler(fpath, **self.single_handler_kwargs)

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

    def _check_freq(self):
        """Check the input vs inferred file frequency and warn if !="""
        test_freq_1 = self.freq.lower().replace('T', 'min')
        test_freq_2 = self.inferred_freq.lower().replace('T', 'min')
        if test_freq_1 != test_freq_2:
            w = ('CloudVar handler has an input frequency of "{}" but '
                 'inferred a frequency of "{}" for pattern: {}'
                 .format(self.freq, self.inferred_freq, self.pattern))
            logger.warning(w)
            warn(w)
        else:
            m = ('CloudVar handler has a frequency of "{}" for pattern: {}'
                 .format(self.freq, self.pattern))
            logger.debug(m)

    @property
    def doy(self):
        """Get the day of year string e.g. 001 for jan 1 and 365 for Dec 31

        Returns
        -------
        str
        """
        return str(self._date.timetuple().tm_yday).zfill(3)

    @property
    def pattern(self):
        """Get the source file pattern which is sent to glob().

        Returns
        -------
        str | None
        """
        pat = super().pattern
        if pat is None:
            msg = ('Need "pattern" input kwarg to initialize CloudVar data '
                   'handler. Can have {doy} format key.')
            logger.error(msg)
            raise KeyError(msg)

        if '{doy}' in pat:
            pat = pat.format(doy=self.doy)

        return pat

    def pre_flight(self):
        """Perform pre-flight checks - source pattern check.

        Returns
        -------
        missing : str
            Look for the source pattern and return the string if not found.
            If nothing is missing, return an empty string.
        """

        missing = ''
        if not any(self.flist):
            missing = self.pattern

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

    @property
    def file(self):
        """Alias for cloudvar file list.

        Returns
        -------
        list
        """
        return self.flist

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
            self._flist = NFS(self.pattern).glob()
            if not any(self._flist):
                emsg = ('Could not find or found too many source files '
                        'for dataset "{}" with glob pattern: "{}". '
                        'Found {} files: {}'
                        .format(self.name, self.pattern,
                                len(self._flist), self._flist))
                logger.error(emsg)
                raise FileNotFoundError(emsg)

            # sort by timestep after the last underscore before .level2.h5
            self._flist = sorted(self._flist, key=self.get_timestamp)
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
            if NFS(fp).size() < 1e6:
                msg = ('Cloud data source file is less than 1MB, skipping: {}'
                       .format(fp))
                warn(msg)
                logger.warning(msg)

        flist = [fp for fp in flist if NFS(fp).size() > 1e6]

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

            # make sure that flist still matches
            not_used = [fp for fp in self.flist
                        if fp not in self._file_df['flist'].values.tolist()]
            if any(not_used):
                file_names = self._file_df['flist'].values
                file_names = [os.path.basename(fp) if isinstance(fp, str)
                              else fp for fp in file_names]
                temp_df = self._file_df.copy()
                temp_df['flist'] = file_names
                msg = ('Some available cloud source data files were not used: '
                       '{}\nCloud file mapping table:\n{}'
                       .format(not_used, temp_df))
                logger.warning(msg)

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
                try:
                    ti_delta_minutes = int(mode(ti_deltas_minutes).mode)
                except Exception as e:
                    msg = ('Could not get mode of time index deltas: {}'
                           .format(ti_deltas_minutes))
                    logger.error(msg)
                    raise ValueError(msg) from e

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

    @property
    def single_handler_kwargs(self):
        """Get a kwargs dict to initialize a single cloud timestep data handler

        Returns
        -------
        dict
        """
        kwargs = {'dsets': self._dsets,
                  'parallax_correct': self._parallax_correct,
                  'solar_shading': self._solar_shading,
                  'remap_pc': self._remap_pc,
                  }
        return kwargs

    @staticmethod
    def get_handler(fp_cloud, **kwargs):
        """Get a single cloud timestep data handler for one cloud file.

        Parameters
        ----------
        fp_cloud : str
            Single cloud source file either .nc or .h5
        kwargs : dict
            Kwargs for the initialization of CloudVarSingleH5 or
            CloudVarSingleNC along with fp_cloud

        Returns
        -------
        obj : None | CloudVarSingleNC | CloudVarSingleH5
            Handler for a single cloud data file.
        """

        obj = None

        if str(fp_cloud).endswith('.h5'):
            try:
                obj = CloudVarSingleH5(fp_cloud, **kwargs)
            except ValueError:
                logger.error(f'Error reading file: {fp_cloud}')
        elif str(fp_cloud).endswith('.nc'):
            try:
                obj = CloudVarSingleNC(fp_cloud, **kwargs)
            except ValueError:
                logger.error(f'Error reading file: {fp_cloud}')

        return obj

    def save_obj(self, cloud_var_single):
        """Save a single cloud object to a cache for later use.

        Parameters
        ----------
        cloud_obj_single : CloudVarSingleH5 | CloudVarSingleNC
            Single-timestep cloud variable data handler.
        """
        self._obj_cache[cloud_var_single.fpath] = cloud_var_single
