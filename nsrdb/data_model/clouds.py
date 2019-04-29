# -*- coding: utf-8 -*-
"""A framework for handling UW/GOES source data."""

import numpy as np
import pandas as pd
import h5py
import os
import netCDF4
import logging

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
        self._grid : pd.DataFrame
            GOES source coordinates (labels: ['latitude', 'longitude']).
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
    def _parse_grid(fpath):
        """Extract the cloud data grid for the current timestep.

        Returns
        -------
        self._grid : pd.DataFrame
            GOES source coordinates (labels: ['latitude', 'longitude']).
        """

        labels = ['latitude', 'longitude']
        grid = pd.DataFrame()
        with h5py.File(fpath, 'r') as f:
            for dset in labels:
                if dset not in list(f):
                    raise KeyError('Could not find "{}" in the cloud '
                                   'file: {}'
                                   .format(dset, fpath))
                grid[dset] = CloudVarSingleH5.pre_process(
                    dset, f[dset][...], dict(f[dset].attrs))
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
        logger.debug('Retrieving single timestep cloud source data from: "{}"'
                     .format(os.path.basename(self._fpath)))
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
    def _parse_grid(fpath):
        """Extract the cloud data grid for the current timestep.

        Returns
        -------
        grid : pd.DataFrame
            GOES source coordinates (labels: ['latitude', 'longitude']).
        mask : np.ndarray
            Boolean array to extract good data.
        """

        sparse_mask = None
        labels = ['latitude', 'longitude']
        grid = pd.DataFrame()
        with netCDF4.Dataset(fpath, 'r') as f:
            for dset in labels:
                if dset not in list(f.variables.keys()):
                    raise KeyError('Could not find "{}" in the cloud '
                                   'file: {}'
                                   .format(dset, fpath))

                # use netCDF masked array mask to reduce ~1/4 of the data
                if sparse_mask is None:
                    sparse_mask = ~f[dset][:].mask
                grid[dset] = f[dset][:].data[sparse_mask]
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
        logger.debug('Retrieving single timestep cloud source data from: "{}"'
                     .format(os.path.basename(self._fpath)))
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

    # the number of files for a given day dictates the temporal frequency
    LEN_TO_FREQ = {1: '1d',
                   48: '30min',
                   96: '15min',
                   288: '5min'}

    def __init__(self, var_meta, name, date, extent='east', path=None,
                 dsets=('cloud_type', 'cld_opd_dcomp', 'cld_reff_dcomp',
                        'cld_press_acha')):
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
        self._flist = None
        self._dsets = dsets
        self._ftype = None

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
        logger.info('Iterating through {} cloud data {} files located in "{}"'
                    .format(len(self._flist), self._ftype, self.path))
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
            if self._ftype == '.h5':
                obj = CloudVarSingleH5(self.flist[self._i], dsets=self._dsets)
            elif self._ftype == '.nc':
                obj = CloudVarSingleNC(self.flist[self._i], dsets=self._dsets)
            else:
                raise TypeError('Did not recognize cloud file type as .nc or '
                                '.h5: {}'.format(self._ftype))

            self._i += 1
            return obj
        else:
            raise StopIteration

    @property
    def path(self):
        """Final path containing cloud data files.

        Path is interpreted as:
            /source_dir/extent/YYYY/DOY/level2/

        Where source_dir is defined in the nsrdb_vars.csv meta/config file.
        """

        if self._path is None:
            doy = str(self._date.timetuple().tm_yday).zfill(3)
            self._path = os.path.join(self.source_dir, self._extent,
                                      str(self._date.year), doy, 'level2')
            if not os.path.exists(self._path):
                raise IOError('Looking for cloud data but could not find the '
                              'target path: {}'.format(self._path))
        return self._path

    def pre_flight(self):
        """Perform pre-flight checks - source dir check.

        Returns
        -------
        missing : str
            Look for the source dir and return the string if not found.
            If nothing is missing, return an empty string.
        """

        missing = ''
        if not os.path.exists(self.path):
            missing = self.path
        return missing

    @staticmethod
    def _h5_flist(path, date):
        """Get the .h5 cloud data file list.

        Parameters
        ----------
        path : str
            Terminal directory containing .h5 files.
        date : datetime.date
            Date of files to look for.

        Returns
        -------
        flist : list
            List of files sorted by timestamp. Empty list if no files found.
        """

        fl = os.listdir(path)
        flist = [os.path.join(path, f) for f in fl
                 if f.endswith('.h5') and str(date.year) in f]

        if flist:
            # sort by timestep after the last underscore before .level2.h5
            flist = sorted(flist, key=lambda x: os.path.basename(x)
                           .split('.')[0].split('_')[-1])
        return flist

    @staticmethod
    def _nc_flist(path, date):
        """Get the .nc cloud data file list.

        Parameters
        ----------
        path : str
            Terminal directory containing .nc files.
        date : datetime.date
            Date of files to look for.

        Returns
        -------
        flist : list
            List of files sorted by timestamp. Empty list if no files found.
        """

        fl = os.listdir(path)
        flist = [os.path.join(path, f) for f in fl
                 if f.endswith('.nc') and str(date.year) in f]

        if flist:
            # sort by timestep after the last underscore before .level2.h5
            flist = sorted(flist, key=lambda x: os.path.basename(x)
                           .split('.')[0].split('_')[-1].strip('s'))
        return flist

    @property
    def flist(self):
        """List of cloud data files for one day. Each file is a timestep.

        Returns
        -------
        flist : list
            List of .h5 or .nc files sorted by timestamp. Exception raised
            if no files are found.
        """

        if self._flist is None:
            self._flist = self._h5_flist(self.path, self._date)
            self._ftype = '.h5'
            if not self._flist:
                self._flist = self._nc_flist(self.path, self._date)
                self._ftype = '.nc'
            if not self._flist:
                raise IOError('Could not find .h5 or .nc files for {} in '
                              'directory: {}'.format(self._date, self.path))
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
