# -*- coding: utf-8 -*-
"""A collection of frameworks for handling NSRDB data sources."""

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


class CloudVarSingle:
    """Framework for single-file/single-timestep cloud data extraction."""

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
        self._grid = self._parse_grid(self._fpath)
        if self.pre_proc_flag:
            self._grid, self._sparse_mask = self.make_sparse(self._grid)

    @staticmethod
    def _parse_grid(fpath):
        """Extract the cloud data grid for the current timestep.

        Returns
        -------
        self._grid : pd.DataFrame
            GOES source coordinates (labels: ['latitude', 'longitude']).
        """

        if fpath.endswith('.h5'):
            labels = ['latitude', 'longitude']
            grid = pd.DataFrame()
            with h5py.File(fpath, 'r') as f:
                for dset in labels:
                    if dset not in list(f):
                        raise KeyError('Could not find "{}" in the cloud '
                                       'file: {}'
                                       .format(dset, fpath))
                    grid[dset] = CloudVarSingle.pre_process(
                        dset, f[dset][...], dict(f[dset].attrs))
        return grid

    @property
    def grid(self):
        """Return the cloud data grid for the current timestep.

        Returns
        -------
        self._grid : pd.DataFrame
            GOES source coordinates (labels: ['latitude', 'longitude']).
        """
        return self._grid

    @staticmethod
    def pre_process(dset, data, attrs, sparse_mask=None, index=None):
        """Pre-process cloud data by filling missing values and unscaling.

        Pre-processing steps:
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
        logger.debug('Retrieving single timestep cloud source data from {}'
                     .format(os.path.basename(self._fpath)))
        data = {}
        if self._fpath.endswith('.h5'):
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


class CloudVar(AncillaryVar):
    """Framework for cloud data extraction (GOES data processed by UW)."""

    # the number of files for a given day dictates the temporal frequency
    LEN_TO_FREQ = {1: '1d',
                   48: '30min',
                   96: '15min',
                   288: '5min'}

    def __init__(self, var_meta, name, date, extent='east', path=None,
                 parallel=False, dsets=('cloud_type', 'cld_opd_dcomp',
                                        'cld_reff_dcomp', 'cld_press_acha')):
        """
        Parameters
        ----------
        var_meta : str | pd.DataFrame
            CSV file or dataframe containing meta data for all NSRDB variables.
        name : str
            NSRDB var name.
        date : datetime.date
            Single day to extract data for.
        extent : str
            Regional (satellite) extent to process, used to form file paths.
        parallel : bool
            Flag to perform regrid in parallel.
        path : str | NoneType
            Optional path string to force a cloud data directory. If this is
            None, the file path will be infered from the extent, year, and day
            of year.
        dsets : tuple | list
            Source datasets to extract. It is more efficient to extract all
            required datasets at once from each cloud file, so that only one
            kdtree is built for each unique coordinate set in each cloud file.
        """

        self._extent = extent
        self._path = path
        self._parallel = parallel
        self._flist = None
        self._dsets = dsets

        super().__init__(var_meta, name, date)

        if len(self) not in self.LEN_TO_FREQ:
            raise KeyError('Bad number of cloud data files. Counted {} files '
                           'in {} but expected one of the following: {}'
                           .format(len(self), self.path,
                                   list(self.LEN_TO_FREQ.keys())))

        if len(self) != 1 and len(self) != len(self.time_index):
            raise KeyError('Bad number of cloud data files. Counted {} files '
                           'in {} but a time index with length {}.'
                           .format(len(self), self.path, len(self.time_index)))

    def __len__(self):
        """Length of this object is the number of source files."""
        return len(self.flist)

    def __iter__(self):
        """Initialize this instance as an iter object."""
        self._i = 0
        return self

    def __next__(self):
        """Iterate through CloudVarSingle objects for each cloud data file.

        Returns
        -------
        obj : CloudVarSingle
            Single cloud data retrieval object.
        """

        # iterate through all timesteps (one file per timestep)
        if self._i < len(self.flist):
            # initialize a single timestep helper object
            obj = CloudVarSingle(self.flist[self._i], dsets=self._dsets)
            self._i += 1
            return obj
        else:
            raise StopIteration

    @property
    def path(self):
        """Final path containing cloud data files."""
        if self._path is None:
            doy = str(self._date.timetuple().tm_yday).zfill(3)
            self._path = os.path.join(self.source_dir, self._extent,
                                      str(self._date.year), doy, 'level2')
            if not os.path.exists(self._path):
                raise IOError('Looking for cloud data but could not find the '
                              'target path: {}'.format(self._path))
        return self._path

    @property
    def flist(self):
        """List of cloud data files for one day. Each file is a timestep."""
        if self._flist is None:
            fl = os.listdir(self.path)
            self._flist = [os.path.join(self.path, f) for f in fl
                           if f.endswith('.h5') and str(self._date.year) in f]
            logger.debug('Cloud data initialized with the following file '
                         'list of length {}:\n{}'
                         .format(len(self._flist), self._flist))
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
        freq = self.LEN_TO_FREQ[len(self)]
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

        rh = MerraVar.relative_humidity(t, h, p)
        dp = (243.04 * (np.log(rh / 100.) + (17.625 * t / (243.04 + t))) /
              (17.625 - np.log(rh / 100.) - ((17.625 * t) / (243.04 + t))))

        # ensure that temeprature is reconverted to Kelvin
        if convert_t:
            t += 273.15

        return dp


class VarFactory:
    """Factory pattern to retrieve ancillary variable helper objects."""

    # mapping of NSRDB variable names to helper objects
    MAPPING = {'asymmetry': AsymVar,
               'air_temperature': MerraVar,
               'alpha': MerraVar,
               'aod': MerraVar,
               'cloud_type': CloudVar,
               'cld_opd_dcomp': CloudVar,
               'cld_reff_dcomp': CloudVar,
               'cld_press_acha': CloudVar,
               'dew_point': MerraVar.dew_point,
               'ozone': MerraVar,
               'relative_humidity': MerraVar.relative_humidity,
               'specific_humidity': MerraVar,
               'ssa': MerraVar,
               'surface_pressure': MerraVar,
               'total_precipitable_water': MerraVar,
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
            if var_name in ('dew_point', 'relative_humidity'):
                return self.MAPPING[var_name]
            else:
                return self.MAPPING[var_name](*args, **kwargs)

        else:
            raise KeyError('Did not recognize "{}" as an available NSRDB '
                           'variable. The following variables are available: '
                           '{}'.format(var_name, list(self.MAPPING.keys())))
