# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 20:35:24 2014

@author: alopez

"""

import h5py
import numpy as np
import os
import logging
from scipy.spatial import cKDTree
import pandas as pd
import calendar

from nsrdb_utilities import __version__
from nsrdb_utilities.loggers import init_logger


logger = logging.getLogger(__name__)


class Blender:
    """East-west NSRDB data blending utility."""

    def __init__(self, source_dir, fout, year,
                 chunk_shape=[None, 100], var=None,
                 f_loc='/projects/PXS/reference_grids/location_info.h5'):
        """Initialize a blender utility object.

        Parameters
        ----------
        source_dir : str
            Directory containing east/west source h5 files to be blended.
        fout : str
            Target output file (with path) to save blended data.
        year : int | str
            Year to be blended. Only files with this year in the filename are
            blended.
        chunk_shape : list
            Two-entry chunk shape (y chunk, x chunk). If an entry is None,
            this value will be replaced by the corresponding dimension in the
            final dataset.
        var : str | NoneType
            Optional variable flag to only blend a certain variable. If this
            is set, only files in source_dir with var in the name will be
            blended.
        f_loc : str
            h5 file (with path) containing reference grid location info.
        """

        self.year = int(year)
        self.i_start = 0
        self.i_end = 0
        self.lat = []
        self.lon = []
        self.elev = []
        self.chunk_shape = chunk_shape
        self.source_dir = source_dir
        self.fout = fout
        self.f_loc = f_loc
        self.irradiance_dsets = ['ghi', 'clearsky_ghi', 'dni',
                                 'clearsky_dni', 'dhi', 'clearsky_dhi']
        self.daysinyear = 366 if calendar.isleap(self.year) else 365

        self._getFiles(var=var)
        self._defineIndicesMeta()
        self._dataShape()
        self._set_chunk_shape()
        self._createHDF()

    def _getFiles(self, var=None):
        """Get the target file list in dataframe format."""
        logger.debug('Getting files...')

        if var is not None:
            logger.debug('Only blending files containing "{}"'.format(var))

        file_list = os.listdir(self.source_dir)
        east = [f for f in file_list if
                self.blend_f(f, 'east', self.year, flag=var)]
        west = [f for f in file_list if
                self.blend_f(f, 'west', self.year, flag=var)]

        # Sort all of the HDFs so that we get them in ascending index order
        east_files = sorted(east, key=lambda x: int(x.split('_')[2]))
        west_files = sorted(west, key=lambda x: int(x.split('_')[2]))

        logger.debug('Blending the following east files:\n{}'
                     .format(east_files))
        logger.debug('Blending the following west files:\n{}'
                     .format(west_files))

        self.files = pd.DataFrame(west_files + east_files)
        self.files.columns = ['files']

        data = pd.DataFrame(self.files.files.str.split('_', expand=True))
        self.files = pd.concat([self.files, data], axis=1)

        self.files.columns = ['files', 'region', 'year', 'start', 'end']
        self.files['end'] = self.files.end.str.split('.', expand=True)[0]

        self.files.start = pd.to_numeric(self.files.start)
        self.files.end = pd.to_numeric(self.files.end)

    def _defineIndicesMeta(self):
        """Construct meta and determine indicies given overlap"""
        logger.debug('Defining Indices...')

        all_indices = []

        for _, row in self.files.iterrows():
            with h5py.File(self.source_dir + row.files, 'r') as curHDF:
                lat = curHDF['latitude'][:]
                lon = curHDF['longitude'][:]
                elev = curHDF['elevation'][:]
            if row.region == 'west' and np.any(lon > -105):
                indices = np.where(lon <= -105)[0]
                all_indices.append(indices)
            else:
                indices = np.arange(len(lon))
                all_indices.append(indices)

            self.lat.append(lat[indices])
            self.lon.append(lon[indices])
            self.elev.append(elev[indices])

        self.lat = np.concatenate(self.lat)
        self.lon = np.concatenate(self.lon)
        self.elev = np.concatenate(self.elev)

        self.files['indices'] = np.array(all_indices)

    def _dataShape(self):
        """Define the shape of datasets to be created in final output"""
        logger.debug('determining shape...')

        self.y = 17568 if calendar.isleap(self.year) else 17520
        self.x = len(self.lon)
        self.shape = (self.y, self.x)
        logger.debug('Output data shape is: {}'.format(self.shape))

    def _set_chunk_shape(self):
        """Set chunk shape values that are "None" based on data shape."""
        if hasattr(self, 'shape') and hasattr(self, 'chunk_shape'):
            for i in range(2):
                if self.chunk_shape[i] is None:
                    self.chunk_shape[i] = self.shape[i]

    def _createHDF(self):
        """Create the output HDF with datasets and attrs"""

        logger.debug('creating output HDF...')

        self.hfile = h5py.File(self.fout, 'w')

        # create summary stats group/datasets
        self.hfile.create_group('stats')

        with h5py.File(self.source_dir + self.files.files[0], 'r') as hdf:

            datasets, dtypes = self.get_dsets(
                self.source_dir + self.files.files[0])

            for i, dataset in enumerate(datasets):
                self.hfile.create_dataset(dataset, shape=self.shape,
                                          dtype=dtypes[i],
                                          chunks=self.chunk_shape)

                for stat in ['min', 'max', 'avg', 'std']:
                    self.hfile.create_dataset(
                        'stats/{d}_{s}'.format(d=dataset, s=stat),
                        shape=(self.x, ), dtype=np.float)

                for attr in hdf[dataset].attrs.keys():
                    self.hfile[dataset].attrs[attr] = hdf[dataset].attrs[attr]

        # add meta
        meta = np.dtype([('latitude', np.float), ('longitude', np.float),
                         ('elevation', np.float), ('timezone', np.int),
                         ('country', 'S30'), ('state', 'S30'),
                         ('county', 'S30'), ('urban', 'S30'),
                         ('population', np.int), ('landcover', np.int)])

        self.hfile.create_dataset('meta', shape=(self.x,), dtype=meta)

        self.hfile['meta']['latitude'] = self.lat
        self.hfile['meta']['longitude'] = self.lon
        self.hfile['meta']['elevation'] = self.elev

        self.hfile['meta'].attrs['lat_units'] = 'decimal degrees'
        self.hfile['meta'].attrs['lng_units'] = 'decimal degrees'
        self.hfile['meta'].attrs['elevation_units'] = 'meters'
        self.hfile['meta'].attrs['timezone_units'] = 'UTC Offset'

        # add time stamp
        time_index = pd.date_range('1-1-{y}'.format(y=self.year),
                                   '1-1-{y}'.format(y=self.year + 1),
                                   freq='30Min')[:-1]
        self.hfile.create_dataset('time_index',
                                  data=time_index.values.astype('S30'),
                                  dtype='S30')

        # # add location info
        with h5py.File(self.f_loc, 'r') as htz:
            target_ll = np.dstack((self.lon.ravel(), self.lat.ravel()))[0]
            # Build lat/lng pairs by unravel
            ref_ll = np.dstack((np.ravel(htz['meta']['longitude'][...]),
                                np.ravel(htz['meta']['latitude'][...])))[0]
            # Build NN tree
            tree = cKDTree(ref_ll)
            # Get the distance and index of NN
            _, index = tree.query(target_ll, k=1, distance_upper_bound=0.05)
            # iterate each dataset
            for key in htz.keys():
                if not key == 'meta':
                    data = htz[key][...]
                    self.hfile['meta'][key] = data[index]
                    attr = '{key}_fill_value'.format(key=key)

                    self.hfile['meta'].attrs[attr] = \
                        htz[key].attrs['fill_value']

                    if key == 'landcover':
                        self.hfile['meta'].attrs['landcover_types'] = \
                            htz[key].attrs['landcover_types']

        # Add NSRDB Version
        self.hfile.attrs['version'] = __version__

    @staticmethod
    def blend_f(fname, region, year, flag=None, ext='.h5'):
        """Check whether to blend a file based on various flags.

        Parameters
        ----------
        fname : str
            Target filename that may have to be blended.
        region : str
            Region being blended (east or west).
        year : int | str
            Year being blended.
        flag : str | NoneType
            Optional additional flag to look for in fname. Could be an NSRDB
            variable.
        ext : str
            Requested file extension.

        Returns
        -------
        blend : bool
            Whether or not to blend the target file.
        """

        blend = False
        if region in fname and str(year) in fname and fname.endswith(ext):
            blend = True
        if flag is not None:
            if flag not in fname:
                blend = False
        return blend

    @staticmethod
    def get_dsets(h5, ignore=('latitude', 'longitude', 'elevation', 'meta')):
        """Get a list of datasets and dtypes to operate on from the file.

        Parameters
        ----------
        h5 : str
            Target h5 file to retrieve datasets from. Must include path.
        ignore : tuple | list
            Datasets to ignore.

        Returns
        -------
        datasets : list
            List of datasets.
        dtypes : list
            List of dataset dtypes.
        """

        with h5py.File(h5, 'r') as hdf:
            datasets = [i for i in hdf if i not in ignore]
            dtypes = [hdf[i].dtype for i in hdf if i not in ignore]
        return datasets, dtypes

    def blend_file(self, source_h5, indices, out_i_start, out_i_end):
        """Blend a single source h5 file to the final output blended h5.

        Parameters
        ----------
        source_h5 : str
            Source h5 file (either east or west) found in self.source_dir to
            be blended.
        indices : np.array
            Indices in source_h5 to retrieve and pass to the final output h5.
        out_i_start : int
            INCLUSIVE starting site (column) index in the output h5file to
            write the data to.
        out_i_end : int
            EXCLUSIVE final site (column) index in the output h5file to write
            the data to.
        """

        with h5py.File(self.source_dir + source_h5, 'r') as hdf:
            logger.debug('Blending file: {}'.format(source_h5))

            datasets, _ = self.get_dsets(self.source_dir + source_h5)

            for dataset in datasets:

                logger.debug('Performing blend on dset: {}'.format(dataset))

                data = hdf[dataset][:]
                data = data[:, indices]
                self.hfile[dataset][:, out_i_start:out_i_end] = data

                self.write_stats(data, dataset, out_i_start, out_i_end)

    def write_stats(self, data, dataset, out_i_start, out_i_end):
        """Calculate statistics on data and write summary to output h5.

        Parameters
        ----------
        data : np.ndarray
            Data array to calculate stats on
        dataset : str
            Dataset name.
        out_i_start : int
            INCLUSIVE starting site (column) index in the output h5file to
            write the data to.
        out_i_end : int
            EXCLUSIVE final site (column) index in the output h5file to write
            the data to.
        """

        if 'psm_scale_factor' in self.hfile[dataset].attrs:
            data /= self.hfile[dataset].attrs['psm_scale_factor']

        # create and save stats
        if dataset in self.irradiance_dsets:
            davg = np.sum(data, axis=0) / self.daysinyear / 1000. / 2.
        else:
            davg = np.mean(data, axis=0)

        dmin = np.min(data, axis=0)
        dmax = np.max(data, axis=0)
        dstd = np.std(data, axis=0)

        k1 = 'stats/{d}_{s}'.format(d=dataset, s='avg')
        k2 = 'stats/{d}_{s}'.format(d=dataset, s='min')
        k3 = 'stats/{d}_{s}'.format(d=dataset, s='max')
        k4 = 'stats/{d}_{s}'.format(d=dataset, s='std')

        self.hfile[k1][out_i_start:out_i_end] = davg
        self.hfile[k2][out_i_start:out_i_end] = dmin
        self.hfile[k3][out_i_start:out_i_end] = dmax
        self.hfile[k4][out_i_start:out_i_end] = dstd

    def process_all(self):
        """Process and blend all files in the file list."""

        logger.debug('Processing files... count: {}'.format(len(self.files)))

        for _, row in self.files.iterrows():

            self.i_start = self.i_end
            self.i_end = self.i_start + len(row.indices)

            self.blend_file(row.files, row.indices, self.i_start, self.i_end)

    @classmethod
    def blend_var(cls, var, year,
                  fout='/scratch/gbuster/blended/test_blend.h5'):
        """Blend a single variable."""

        init_logger(__name__, {'log_level': 'DEBUG', 'log_file': 'blend.log'})
        source_dir = '/projects/PXS/ancillary/gridded/striped'
        blend = Blender(source_dir, fout, year, var=var)
        blend.process_all()
