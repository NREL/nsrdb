'''
#@author: Anthony Lopez
'''
import numpy as np
import h5py
import time
import os
import pandas as pd
import sys
from scipy.spatial import cKDTree
from configobj import ConfigObj
import logging

from nsrdb_utilities.loggers import init_logger


logger = logging.getLogger(__name__)


class ReGrid:
    """Class to manage re-gridding of cloud data from U-Wisc."""

    def __init__(self, config, region, year):
        """
        Parameters
        ----------
        config
        region
        year
        """
        self.config = config
        self.year = int(year)
        self.region = region
        self.pxs_attrs = config['regrid']['pxs_attrs']
        self.pxs_attrs_info = None
        self.extent = None
        self.time_index = None
        self.existing_files = None
        self.missing_index = None
        self.time_steps = None
        self._setExtent()
        self._setTimeIndex()
        self._getFiles()
        self._getPXSInfo()
        self._defineTimeSteps()
        self._checkComplete()

    def _fillData(self, attr):
        '''
        Return Fill Data for missing time-step
        '''
        fill = np.empty(shape=(len(self.extent), ),
                        dtype=self.pxs_attrs_info[attr]['dtype'])
        if self.pxs_attrs_info[attr]['fill_value']:
            fill[:] = self.pxs_attrs_info[attr]['fill_value']
        else:
            fill[:] = np.array([-999])\
                .astype(self.pxs_attrs_info[attr]['dtype'])
        return fill

    def _setExtent(self):
        '''
        Returns indices of the fixed grid
        '''
        grid_file = ('/projects/PXS/reference_grids/{region}_psm_extent.csv'
                     .format(region=self.region))
        self.extent = pd.read_csv(grid_file)

    def _setTimeIndex(self):
        """Set the reference time index."""
        rng = pd.date_range('1-1-%s' % self.year,
                            '1-1-%s' % (int(self.year) + 1),
                            freq='30Min')
        self.time_index = rng[:-1]

    def _getFiles(self):
        file_path, day, hour, minute = [], [], [], []
        fdir = os.path.join(self.config['regrid']['in_dir'],
                            self.region, str(self.year))
        for folder, _, files in os.walk(fdir):
            for file in files:
                try:
                    file_path.append(os.path.join(folder, file))
                    day.append(int(os.path.join(folder, file).split('/')[6]))
                    hour.append(int(file.split('_')[-1].split('.')[0][0:2]))
                    minute.append(int(file.split('_')[-1].split('.')[0][2:]))
                except Exception as e:
                    logger.exception(e)

        # create the dataframe
        df = pd.DataFrame({'file_path': file_path,
                           'year': self.year,
                           'day': day,
                           'hour': hour,
                           'minute': minute})
        # sort on time
        try:
            df.sort(['year', 'day', 'hour', 'minute'], inplace=True)
        except Exception as e:
            logger.exception(e)
            df.sort_values(['year', 'day', 'hour', 'minute'], inplace=True)

        # create time index of existing data
        strtime = df[['year', 'day', 'hour', 'minute']]\
            .apply(lambda x: '{},{},{},{}'.format(x[0], x[1], x[2], x[3]),
                   axis=1)

        index = pd.to_datetime(strtime, format='%Y,%j,%H,%M')
        df = df.set_index(index)
        self.existing_files = df

    def _defineTimeSteps(self):
        '''
        Find indices of missing files so that they can be filled
        '''
        df = pd.DataFrame(np.arange(len(self.time_index)),
                          index=self.time_index)

        df_existing = self.existing_files.copy(deep=True)
        if self.region == 'east':
            df_existing = df_existing.set_index(
                df_existing.index.shift(-15, freq='min'))

        df = df.merge(df_existing, how='left',
                      left_index=True, right_index=True)
        df.columns = ['write_index', 'day', 'file_path',
                      'hour', 'minute', 'year']

        self.missing_index = np.isnan(df['year'])
        self.time_steps = df

    def _getPXSInfo(self):
        with h5py.File(self.existing_files['file_path'].iloc[0]) as hfile:
            store = {}
            for attr in self.pxs_attrs:
                store.update({attr: {'scale_factor': 1, 'add_offset': 0,
                                     'fill_value': 0, 'long_name': 'N/A',
                                     'units': 'N/A'}})
                for k, _ in store[attr].iteritems():
                    try:
                        store[attr][k] = hfile[attr].attrs[k]
                    except Exception as e:
                        logger.exception(e)

                        if k == 'fill_value':
                            try:
                                store[attr][k] = hfile[attr]\
                                    .attrs['_FillValue']
                            except Exception as e:
                                logger.exception(e)

                store[attr].update({'dtype': np.float})
                try:
                    store[attr]['dtype'] = hfile[attr].dtype
                except Exception as e:
                    logger.exception(e)

        self.pxs_attrs_info = store

    def _checkComplete(self):
        '''
        check to see if files were already created
        '''
        tstoprocess = []
        for _, row in self.time_steps.iterrows():
            fh5 = '%s_%s_%s.h5' % (row.write_index, self.region, self.year)
            h_name = os.path.join(self.config['regrid']['tmp_dir'], fh5)
            try:
                with h5py.File(h_name, 'r'):
                    pass
            except Exception as e:
                logger.exception(e)
                tstoprocess.append(row.write_index)

        self.tstoprocess = tstoprocess

    def createHDFSingle(self, h_name):
        """Create single HDF files"""
        fixedHDF = h5py.File(h_name, 'w')

        for attr in self.pxs_attrs:
            # create the dataset and set chunk size to be time wise -
            # compression is not yet implemented in MPI driver
            fixedHDF.create_dataset(attr, shape=(len(self.extent), ),
                                    dtype=self.pxs_attrs_info[attr]['dtype'])

            fixedHDF[attr].attrs['scale_factor'] = \
                self.pxs_attrs_info[attr]['scale_factor']

            fixedHDF[attr].attrs['add_offset'] = \
                self.pxs_attrs_info[attr]['add_offset']

            fixedHDF[attr].attrs['fill_value'] = \
                self.pxs_attrs_info[attr]['fill_value']

            fixedHDF[attr].attrs['long_name'] = \
                self.pxs_attrs_info[attr]['long_name']

            fixedHDF[attr].attrs['units'] = \
                self.pxs_attrs_info[attr]['units']

        # lat/lng dataset
        meta = np.dtype([('longitude', np.float), ('latitude', np.float)])
        dset = fixedHDF.create_dataset('meta', shape=(len(self.extent),),
                                       dtype=meta)

        dset['latitude'] = self.extent['latitude']
        dset['longitude'] = self.extent['longitude']

        fixedHDF.create_dataset('time_index',
                                data=self.time_index.values.astype('S30'),
                                dtype='S30')
        fixedHDF['time_index'].attrs['Time Zone'] = 'UTC'

        # not all data per region are in the same time arrangement.
        # this data set keeps track of this
        fixedHDF.create_dataset('time_step_actual',
                                shape=fixedHDF['time_index'].shape,
                                dtype='S4')
        msg = ('The actual time-step of each index as read from the source '
               'HDF file.')
        fixedHDF['time_step_actual'].attrs['long_name'] = msg

        return fixedHDF

    def processDays(self, core):
        """Process all days"""
        logger.info(core, 'processing...')
        extent2d = np.dstack((self.extent['longitude'],
                              self.extent['latitude']))[0]

        # select out the days to process for this core
        df = self.time_steps[np.in1d(self.time_steps.write_index,
                                     self.tstoprocess)]
        for time_index, row in df.iterrows():
            # create the fixedHDF for a single time-step
            fh5 = '%s_%s_%s.h5' % (row.write_index, self.region, self.year)
            h_name = os.path.join(self.config['regrid']['tmp_dir'], fh5)
            logger.info(h_name, row.file_path)
            fixedHDF = self.createHDFSingle(h_name)

            if pd.isnull(row.file_path):
                logger.info('file is missing')
                logger.info('Missing File Catch: {t}'.format(t=time_index))

                for attr in self.pxs_attrs:
                    data = self._fillData(attr)
                    fixedHDF[attr][:] = data
            else:
                try:
                    with h5py.File(row.file_path, 'r') as hfile:
                        logger.info('{f}'.format(f=row.file_path))
                        # Get the lng
                        lat = (hfile['latitude'][...] *
                               hfile['latitude'].attrs['scale_factor'])
                        lng = (hfile['longitude'][...] *
                               hfile['longitude'].attrs['scale_factor'])

                        # Get fill values
                        lng_fill = (hfile['longitude'].attrs['_FillValue'] *
                                    hfile['longitude'].attrs['scale_factor'])
                        lat_fill = (hfile['latitude'].attrs['_FillValue'] *
                                    hfile['latitude'].attrs['scale_factor'])

                        # Build lat/lng pairs by unravel
                        # where not == fill value
                        ll = np.dstack((np.ravel(lng), np.ravel(lat)))[0]

                        # good (not missing) spatial index
                        sgood = np.where((ll[:, 0] != lng_fill) &
                                         (ll[:, 1] != lat_fill))[0]

                        # valid coords
                        valid_coords = ll[sgood]

                        # Build NN tree
                        tree = cKDTree(valid_coords)

                        # Get the distance and index of NN
                        _, index = tree.query(extent2d, k=1,
                                              distance_upper_bound=0.5)

                        for attr in self.pxs_attrs:
                            logger.info('writing: ', attr)
                            # pull data and subset by good indices
                            data = hfile[attr][...].ravel()[sgood]

                            if self.pxs_attrs_info[attr]['fill_value']:
                                data = np.append(
                                    data,
                                    self.pxs_attrs_info[attr]['fill_value'])
                            else:
                                data = np.append(data, -999)
                            # Get the data out according to the NN index
                            fixedHDF[attr][:] = data[index]

                except Exception as e:
                    logger.exception('source exists, but file is corrupt: {}'
                                     .format(e))
                    # should only trigger if the source file exists
                    # but something is wrong with it
                    logger.info('Existing File Error Catch: {f} : {e}'
                                .format(f=row.file_path, e=e))
                    for attr in self.pxs_attrs:
                        data = self._fillData(attr)
                        fixedHDF[attr][:] = data
            fixedHDF.close()


if __name__ == '__main__':
    from mpi4py import MPI

    init_logger(__name__, log_level='DEBUG', log_file=None)

    # Config input file specified as input
    config = ConfigObj('../config/{config_file}.ini'
                       .format(config_file=sys.argv[1]), unrepr=True)
    region = sys.argv[2]
    year = sys.argv[3]
    cores = int(sys.argv[4])

    time_start = time.time()

    regrid = ReGrid(config, region, year)

    jobs = np.array_split(regrid.tstoprocess, cores)
    logger.info('processing: ', len(regrid.tstoprocess), ' files...')
    for core, ts in enumerate(jobs):
        if core == MPI.COMM_WORLD.rank:
            regrid.processDays(core, ts)

    logger.info('Finished {} in {} minutes'
                .format(year, (time.time() - time_start) / 60))
