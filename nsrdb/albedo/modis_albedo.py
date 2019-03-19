# -*- coding: utf-8 -*-
"""NSRDB MODIS Albedo processing tools.

Adapted from Galen and Nick Gilroy's original code:
    https://github.nrel.gov/dav-gis/pv_task/tree/dev/pv_task

@authors: gbuster & ngilroy
"""

import os
import h5py
from pyhdf.SD import SD, SDC
import numpy as np
import pandas as pd
import sys
import time
# import warnings
# from collections import OrderedDict
import psutil
from scipy.spatial import cKDTree
from configobj import ConfigObj
from scoop import futures
# import linecache
import copy
# from scoop import shared
import logging
from memory_profiler import memory_usage

from nsrdb.utilities.loggers import init_logger
# from nsrdb.utilities.execution import PBS
from nsrdb.utilities.file_utils import url_download


logger = logging.getLogger(__name__)


def mem_str():
    """Get a string to log memory status."""
    mem = psutil.virtual_memory()
    msg = ('{0:.3f} GB used of {1:.3f} GB total ({2:.1f}% used) '
           '({3:.3f} GB free) ({4:.3f} GB available).'
           .format(mem.used / 1e9,
                   mem.total / 1e9,
                   100 * mem.used / mem.total,
                   mem.free / 1e9,
                   mem.available / 1e9))
    return msg


class RetrieveMODIS:
    """Class to manage MODIS data retrieval"""

    def __init__(self, year, target_path):
        """
        Parameters
        ----------
        year : int | str
            Year to download modis data (last year is currently 2015).
        target_path : str
            Target path to download files to.
        """

        self._year = year
        self._target_path = target_path

    def retrieve_data(self):
        """Retrieve MODIS albedo source data for a single year.

        Parameters
        ----------
        year : int | str
            Year to download modis data (last year is currently 2015).
        target_path : str
            Target path to download files to.
        """

        f_dir = 'ftp://rsftp.eeos.umb.edu/data02/Gapfilled/{year}/'
        fname_base = 'MCD43GF_wsa_shortwave_{day}_{year}.hdf'
        flink_base = os.path.join(f_dir, fname_base)
        days = [str(d).zfill(3) for d in range(1, 362, 8)]
        flinks = [flink_base.format(year=self._year, day=day) for day in days]
        dl_files = os.listdir(self._target_path)

        if not os.path.exists(self._target_path):
            os.makedirs(self._target_path)

        failed = []
        for url in flinks:
            dl_fname = url.split('/')[-1]

            if dl_fname in dl_files:
                logger.info('Skipping (already exists): {}'.format(dl_fname))
            else:
                logger.info('Downloading {}'.format(dl_fname))
                dfname = os.path.join(self._target_path, dl_fname)
                fail = url_download(url, dfname)
                if failed:
                    failed.append(fail)

        return failed

    @classmethod
    def run(cls, year, target_path, log_level='INFO'):
        """Retrieve MODIS albedo source data for a single year.

        Parameters
        ----------
        year : int | str
            Year to download modis data (last year is currently 2015).
        target_path : str
            Directory to save the downloaded files.
        Returns
        -------
        failed : list
            List of files that failed to download.
        """
        # initialize logger output file for method in modis directory.
        init_logger(__name__, log_file=None, log_level=log_level)
        init_logger('nsrdb_utilities.file_utils', log_file=None,
                    log_level=log_level)

        modis = cls(year, target_path)
        failed = modis.retrieve_data()
        return failed


class Aggregate(object):
    """Class to aggregate MODIS variables to NSRDB grid"""
    def __init__(self, config, years, days):
        """
        Parameters
        ----------
        config : ini
            config file for aggregating MODIS albedo to NSRDB grid.
        years : range(2003, 2015)
            Years to process.
        days : str
            Eight day periods that alebdo data is processed at ('001').
        """
        self.config = config
        self.years = years
        self.days = days

    def sparseToSpatial(self):
        """Sparse to Spatial."""
        with h5py.File(
            os.path.join(self.config['dir']['nsrdb_dir'],
                         self.config['fileNames']['nsrdb']
                         ), 'r') as hfile:
            self.meta = hfile['meta'][...]
        coords = pd.DataFrame(
            {'lon': np.round(self.meta['longitude'] * 100, 0).astype(np.int16),
             'lat': np.round(self.meta['latitude'] * 100, 0).astype(np.int16)})
        coords.loc[(
            coords['lon'] > 0), 'lon'] = coords.loc[(
                coords['lon'] > 0), 'lon'] - 36000
        coords['h5_order'] = np.array(coords.index)
        cs5x = np.abs(
            np.unique(coords['lon'])[0] - np.unique(coords['lon'])[1])
        cs5y = np.abs(
            np.unique(coords['lat'])[0] - np.unique(coords['lat'])[1])
        self.h5gtform = (
            coords['lon'].min() / 100.0,
            coords['lon'].max() / 100.0,
            cs5x / 100.0, coords['lat'].min() / 100.0,
            coords['lat'].max() / 100.0, cs5y / 100.0)
        lon5, lat5 = [x.astype(np.int16) for x in np.meshgrid(
            np.arange(
                coords['lon'].min(),
                coords['lon'].max() + cs5x,
                cs5x, dtype=np.int16),
            np.arange(
                coords['lat'].min(),
                coords['lat'].max() + cs5y,
                cs5y, dtype=np.int16))]
        lon_ind, lat_ind = [x.astype(np.int32) for x in np.meshgrid(
            np.arange(
                np.unique(lon5).size, dtype=np.int32),
            np.arange(
                np.unique(lat5).size, dtype=np.int32))]
        self.grid_shape = lon5.shape
        self.grid = pd.DataFrame(
            {'lon': lon5.flatten(),
             'lat': lat5.flatten(),
             'lon_ind': lon_ind.flatten(),
             'lat_ind': lat_ind.flatten()})
        self.grid = pd.merge(self.grid, coords, how='left', on=['lon', 'lat'])
        self.grid['lon'] = (self.grid.loc[:, 'lon'].astype(np.float32)) / 100
        self.grid['lat'] = (self.grid.loc[:, 'lat'].astype(np.float32)) / 100
        self.grid['h5_order'] = self.grid.loc[:, 'h5_order'].astype(np.float32)
        self.grid.loc[np.isnan(self.grid.loc[:, 'h5_order']), 'lon_ind'] = -99
        self.grid.loc[np.isnan(self.grid.loc[:, 'h5_order']), 'lat_ind'] = -99

    def getHDF4meta(self):
        """Get a meta for the HDF4."""
        try:
            clipping_frame = (
                self.h5gtform[0] - self.h5gtform[2] / 2,
                self.h5gtform[1] + self.h5gtform[2] / 2,
                self.h5gtform[3] - self.h5gtform[5] / 2,
                self.h5gtform[4] + self.h5gtform[5] / 2)
            fname = os.path.join(
                self.config['dir']['in_dir'],
                self.config['fileNames']['inName'].format(
                    var=self.config['variables']['variable_names'][0],
                    day='001', year=self.years[0]))
            hf = SD(fname, SDC.READ)
            lon4 = hf.select('Longitude')[:]
            lon4[lon4 > 0] = lon4[lon4 > 0] - 360
            lat4 = hf.select('Latitude')[:]
            hf.end()
            self.lon4mask = (
                lon4 >= clipping_frame[0]) & (lon4 <= clipping_frame[1])
            self.lon4 = lon4[self.lon4mask]
            self.lat4mask = (
                lat4 >= clipping_frame[2]) & (lat4 <= clipping_frame[3])
            self.lat4 = lat4[self.lat4mask]
        except Exception as e:
            # logger = logging.getLogger('model_run.getHDF4data')
            # logger.info(PrintException())
            logger.info(e)
            logger.info(fname)

    def getAggregationIndices(self):
        """Retrieve indicies for aggregation."""
        tree = cKDTree(
            np.unique(self.grid['lon']).reshape(
                np.unique(self.grid['lon']).size, 1))
        xind = tree.query(self.lon4.reshape(self.lon4.size, 1))[1]
        tree = cKDTree(
            np.unique(self.grid['lat']).reshape(
                np.unique(self.grid['lat']).size, 1))
        yind = tree.query(self.lat4.reshape(self.lat4.size, 1))[1]
        xm, ym = np.meshgrid(xind, yind)
        agg_inds = pd.DataFrame({'xm': xm.flatten(), 'ym': ym.flatten()})
        agg_inds = pd.merge(
            agg_inds,
            self.grid,
            how='left',
            left_on=['xm', 'ym'],
            right_on=['lon_ind', 'lat_ind'])
        del self.grid
        agg_inds.drop(
            ['xm',
             'ym',
             'lat',
             'lat_ind',
             'lon',
             'lon_ind'], axis=1, inplace=True)
        self.h5_mask = ~np.isnan(agg_inds['h5_order']).values
        agg_inds = agg_inds.dropna(axis=0, subset=['h5_order'])
        agg_inds.h5_order = agg_inds.h5_order.astype(np.int32)
        self.agg_inds = agg_inds

    def getHDF4data(self, fname):
        """Retrieve HDF4 data."""
        try:
            hf = SD(fname, SDC.READ)
            key = [x for x in hf.datasets().keys() if (
                'Albedo' in x) or ('CMG' in x)][0]
            h4data = hf.select(key)[:, :][(
                self.lon4mask[np.newaxis, :] * self.lat4mask[
                    :, np.newaxis]) == 1][self.h5_mask]
            hf.end()
            return h4data
        except Exception as e:
            # logger = logging.getLogger('model_run.getHDF4data')
            # logger.info(PrintException())
            logger.info(e)
            logger.info(fname)

    def write_results(self, year, var, results):
        """Write results to HDF."""
        with h5py.File(
            os.path.join(
                self.config['dir']['out_dir'],
                self.config['fileNames']['outName'].format(
                    var=var, year=year)), 'w') as hfile:
            for key, values in results.iteritems():
                hfile.create_dataset(key, data=values, dtype=np.float32)
            hfile.create_dataset('meta', data=self.meta)

    def processYear(self, fname):
        """Process a year."""
        try:
            days = [str(d).zfill(3) for d in range(1, 362, 8)]
            year = fname[-8:-4]
            splt = fname.split('_')
            var = '{var}_{band}'.format(var=splt[1], band=splt[2])
            fnames = (
                [os.path.join(self.config['dir']['in_dir'],
                 fname.format(day=day)) for day in days])
            results = {'means': np.zeros(
                shape=(len(days), self.meta.shape[0]), dtype=np.float32)}
            t0 = time.time()
            mem = np.max(memory_usage(-1, interval=.2, timeout=1))
            for i, fpath in enumerate(fnames):
                # t1 = time.time()
                agg_inds = copy.deepcopy(self.agg_inds)
                # agg_inds = shared.getConst('inds_df')
                agg_inds['h4_data'] = self.getHDF4data(
                    fpath).astype(np.float32)
                agg_inds.loc[
                    agg_inds.loc[:, 'h4_data'] == agg_inds.loc[
                        :, 'h4_data'].max(), 'h4_data'] = np.nan
                results['means'][i, :] = agg_inds.groupby(
                    ['h5_order']).mean().values[:, 0]
                if np.max(memory_usage(-1, interval=.2, timeout=1)) > mem:
                    mem = np.max(
                        memory_usage(-1, interval=.2, timeout=1))
            self.write_results(year, var, results)
            return {'year': year, 'time': (
                time.time() - t0) / 60.0, 'memory': mem}
        except Exception as e:
            # logger = logging.getLogger('model_run.processYear')
            logger.info('exception')
            # logger.info(PrintException())
            logger.info(e)

    @classmethod
    def main(cls, log_level='INFO'):
        """Run processing with test filepaths.

        Parameters
        ----------
        year : int
            Year to process IMS snow data for.
        hpc : bool
            Flag for HPC parallel computation.
        log_level : str
            Logging level for this module (DEBUG or INFO).
        """
        init_logger(__name__, log_file='ims.log', log_level=log_level)
        init_logger('nsrdb.utilities.file_utils', log_file=None,
                    log_level=log_level)
        try:
            t0 = time.time()
            config = ConfigObj(
                '{config_file}.ini'.format(
                    config_file=sys.argv[1]), unrepr=True)
            years = [str(y) for y in range(
                config['variables']['years'][0],
                config['variables']['years'][1] + 1)]
            days = [str(d).zfill(3) for d in range(1, 362, 8)]

            out_vars = [config['fileNames']['outName'].format(
                var=var,
                year=year)
                for year in years
                for var in config['variables']['variable_names']]
            files = os.listdir(config['dir']['out_dir'])
            out_vars = [var for var in out_vars if var not in files]
            out_vars = [(fname[-7:-3], '{var}_{band}'.format(
                var=fname.split('_')[1],
                band=fname.split('_')[2])) for fname in out_vars]

            vars = [config['fileNames']['inName'].format(
                var=pair[1],
                day='{day}',
                year=pair[0]) for pair in out_vars]

            logger.info(vars)
            ag = Aggregate(config, years, days)
            ag.sparseToSpatial()
            logger.info('sparse to spatial')
            ag.getHDF4meta()
            ag.getAggregationIndices()
            logger.info('aggregation indices')
            logger.debug(
                'Memory: %s', str(np.max(memory_usage(
                    -1, interval=.2, timeout=1))))
            # list(futures.map(ag.processYear, vars))
            for result in futures.map_as_completed(ag.processYear, vars):
                logger.info('%s complete, time: %s, memory: %s',
                            result['year'], result['time'], result['memory'])
                logger.debug(
                    'Memory: %s', str(
                        np.max(memory_usage(-1, interval=.2, timeout=1))))
            logger.info('process complete, time: %s',
                        (time.time() - t0) / 60.0)
        except Exception as e:
            # logger.info(PrintException())
            logger.info(e)
