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
from warnings import warn

from nsrdb_utilities import __version__


logger = logging.getLogger(__name__)


class Blender:
    """East-west NSRDB data blending utility."""

    def __init__(self, source_dir, fout, year,
                 f_loc='/projects/PXS/reference_grids/location_info.h5'):
        self.year = int(year)
        self.i_start = 0
        self.i_end = 0
        self.lat = []
        self.lon = []
        self.elev = []
        self.chunk_size = 100
        self.source_dir = source_dir
        self.fout = fout
        self.f_loc = f_loc
        self.irradiance_dsets = ['ghi', 'clearsky_ghi', 'dni',
                                 'clearsky_dni', 'dhi', 'clearsky_dhi']
        self.daysinyear = 366 if calendar.isleap(self.year) else 365

        self._getFiles()
        self._defineIndicesMeta()
        self._dataShape()
        self._createHDF()

    def _getFiles(self):
        '''
        collect all of the files to pull together
        '''
        logger.debug('Getting files...')

        files = os.listdir(self.source_dir)
        east = [i for i in files if 'east' in i]
        west = [i for i in files if 'west' in i]

        # Sort all of the HDFs so that we get them in ascending index order
        east_files = sorted(east, key=lambda x: int(x.split('_')[2]))
        west_files = sorted(west, key=lambda x: int(x.split('_')[2]))

        self.files = pd.DataFrame(west_files + east_files)
        self.files.columns = ['files']

        data = pd.DataFrame(self.files.files.str.split('_', expand=True))
        self.files = pd.concat([self.files, data], axis=1)

        self.files.columns = ['files', 'region', 'year', 'start', 'end']
        self.files['end'] = self.files.end.str.split('.', expand=True)[0]

        self.files.start = pd.to_numeric(self.files.start)
        self.files.end = pd.to_numeric(self.files.end)

    def _defineIndicesMeta(self):
        '''
        using this to contruct meta and determine indicies given overlap
        '''
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
        '''
        defines shape of datasets to be created in final output
        '''
        logger.debug('determining shape...')

        self.y = 17568 if calendar.isleap(self.year) else 17520
        self.x = len(self.lon)

    def _createHDF(self):
        '''
        create the output HDF with datasets and attrs
        '''
        logger.debug('creating output HDF...')

        self.hfile = h5py.File(self.fout, 'w')

        # create summary stats group/datasets
        self.hfile.create_group('stats')

        with h5py.File(self.source_dir + self.files.files[0], 'r') as hdf:
            self.datasets = [i for i in hdf if i not in
                             ['latitude', 'longitude', 'elevation', 'meta']]

            for dataset in self.datasets:
                self.hfile.create_dataset(dataset, shape=(self.y, self.x),
                                          dtype=hdf[dataset].dtype,
                                          chunks=(self.y, self.chunk_size))

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

    def processData(self):
        '''
        main processing function
        '''
        logger.debug('Processing files... count: ', len(self.files))

        for index, row in self.files.iterrows():

            self.i_start = self.i_end
            self.i_end = self.i_start + len(row.indices)

            with h5py.File(self.source_dir + row.files, 'r') as hdf:
                logger.debug(index, row.files)

                for dataset in self.datasets:

                    data = hdf[dataset][:]
                    data = data[:, row.indices]
                    self.hfile[dataset][:, self.i_start:self.i_end] = data

                    try:
                        data /= self.hfile[dataset].attrs['psm_scale_factor']
                    except Exception as e:
                        warn(e)
                    # create and save stats
                    if dataset in self.irradiance_dsets:
                        davg = (np.sum(data, axis=0) / self.daysinyear /
                                1000. / 2.)
                    else:
                        davg = np.mean(data, axis=0)

                    dmin = np.min(data, axis=0)
                    dmax = np.max(data, axis=0)
                    dstd = np.std(data, axis=0)

                    k1 = 'stats/{d}_{s}'.format(d=dataset, s='avg')
                    k2 = 'stats/{d}_{s}'.format(d=dataset, s='min')
                    k3 = 'stats/{d}_{s}'.format(d=dataset, s='max')
                    k4 = 'stats/{d}_{s}'.format(d=dataset, s='std')

                    self.hfile[k1][self.i_start:self.i_end] = davg
                    self.hfile[k2][self.i_start:self.i_end] = dmin
                    self.hfile[k3][self.i_start:self.i_end] = dmax
                    self.hfile[k4][self.i_start:self.i_end] = dstd


if __name__ == '__main__':
    source_dir = '/projects/PXS/ancillary/gridded/striped'
    fout = '/scratch/gbuster/blended/test_blend.h5'
    year = 2016

    blend = Blender(source_dir, fout, year)
    blend.processData()
