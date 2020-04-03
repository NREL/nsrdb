# -*- coding: utf-8 -*-
"""A framework for handling UW/GOES source data."""

import numpy as np
import pandas as pd
import h5py
import re
import os
import netCDF4
import logging
from warnings import warn

from nsrdb.data_model.base_handler import AncillaryVarHandler


logger = logging.getLogger(__name__)


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

        super().__init__(fpath, pre_proc_flag=pre_proc_flag, index=index,
                         dsets=dsets)
        self._grid = self._parse_grid(self._fpath)
        if self.pre_proc_flag:
            self._grid, self._sparse_mask = self.make_sparse(self._grid)

    @staticmethod
    def _parse_grid(fpath, dsets=('latitude_pc', 'longitude_pc')):
        """Extract the cloud data grid for the current timestep.

        Parameters
        ----------
        fpath : str
            Full file path to netcdf4 file.
        dsets : tuple
            Latitude, longitude datasets to retrieve from cloud file. H5
            files have parallax corrected datasets (*_pc). Output dataset
            names will always be 'latitude' and 'longitude'.

        Returns
        -------
        grid : pd.DataFrame | None
            GOES source coordinates (labels: ['latitude', 'longitude']).
            None if bad dataset
        """

        grid = pd.DataFrame()
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

                grid[dset_out] = CloudVarSingleH5.pre_process(
                    dset, f[dset][...], dict(f[dset].attrs))

        if grid.empty:
            grid = None

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

        super().__init__(fpath, pre_proc_flag=pre_proc_flag, index=index,
                         dsets=dsets)
        self._grid, self._sparse_mask = self._parse_grid(self._fpath)

    @staticmethod
    def _parse_grid(fpath, dsets=('latitude_pc', 'longitude_pc')):
        """Extract the cloud data grid for the current timestep.

        Parameters
        ----------
        fpath : str
            Full file path to netcdf4 file.
        dsets : tuple
            Latitude, longitude datasets to retrieve from cloud file. NetCDF4
            files have parallax corrected datasets (*_pc). Output dataset
            names will always be 'latitude' and 'longitude'.

        Returns
        -------
        grid : pd.DataFrame | None
            GOES source coordinates (labels: ['latitude', 'longitude']).
            None if bad dataset
        mask : np.ndarray
            2D boolean array to extract good data.
        """

        sparse_mask = None

        grid = pd.DataFrame()
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

        if grid.empty:
            grid = None

        return grid, sparse_mask

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
                        'cld_press_acha')):
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

            timestamp = self.file_df.index[self._i]
            fpath = self.file_df.iloc[self._i, 0]

            if isinstance(fpath, str):
                # initialize a single timestep helper object
                if fpath.endswith('.h5'):
                    obj = CloudVarSingleH5(fpath, dsets=self._dsets)
                elif fpath.endswith('.nc'):
                    obj = CloudVarSingleNC(fpath, dsets=self._dsets)

                logger.info('Cloud data timestep {} has source file: {}'
                            .format(timestamp, os.path.basename(fpath)))

            else:
                obj = None
                msg = ('Cloud data timestep {} is missing its '
                       'source file.'.format(timestamp))
                warn(msg)
                logger.warning(msg)

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
            logger.info(m)

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

            # walk through current directory looking for day directory
            for dirpath, _, _ in os.walk(self._source_dir):
                dirpath = dirpath.replace('\\', '/')
                if not dirpath.endswith('/'):
                    dirpath += '/'
                if dirsearch in dirpath:
                    for fn in os.listdir(dirpath):
                        if fsearch1 in fn or fsearch2 in fn:
                            self._path = dirpath
                            break
                if self._path is not None:
                    break

            if self._path is None:
                msg = ('Could not find cloud data dir for date {} in '
                       'source_dir {}. Looked for {}, {}, and {}'
                       .format(self._date, self._source_dir, dirsearch,
                               fsearch1, fsearch2))
                logger.exception(msg)
                raise IOError(msg)
            else:
                logger.info('Cloud data dir for date {} found at: {}'
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
    def get_timestamp(fstr):
        """Extract the cloud file timestamp.

        Parameters
        ----------
        fstr : str
            File path or file name with timestamp.

        Returns
        -------
        time : int | None
            Integer timestamp of format:
                YYYYDDDHHMMSSS (YYYY DDD HH MM SSS) (for .nc files)
                YYYYDDDHHMM (YYYY DDD HH MM) (for .h5 files)
            None if not found
        """

        match_nc = re.match(r".*s([1-2][0-9]{13})", fstr)
        match_h5 = re.match(r".*([1-2][0-9]{3}_[0-9]{3}_[0-9]{4})", fstr)

        if match_nc:
            time = int(match_nc.group(1))
        elif match_h5:
            time = int(match_h5.group(1).replace('_', ''))
        else:
            time = None

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
                 if f.endswith('.h5')
                 and str(date.year) in str(CloudVar.get_timestamp(f))]

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
                 if f.endswith('.nc')
                 and str(date.year) in str(CloudVar.get_timestamp(f))]

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
                raise IOError('Could not find .h5 or .nc files for {} in '
                              'directory: {}'.format(self._date, self.path))
        return self._flist

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

        strtime = [str(CloudVar.get_timestamp(fstr))[:11] for fstr in flist]
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
            for i in range(0, len(data_ti), 10):
                freq = pd.infer_freq(data_ti[i:i + 5])

                if freq is not None:
                    break

        if freq is None:
            raise ValueError('Could not infer cloud data timestep frequency.')

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
