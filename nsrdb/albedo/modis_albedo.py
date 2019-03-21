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
import time
import psutil
from scipy.spatial import cKDTree
from configobj import ConfigObj
import copy
import logging
import calendar
from memory_profiler import memory_usage

from nsrdb.utilities.loggers import init_logger
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


class AggregateMODIS(object):
    """Class to aggregate MODIS variables to NSRDB grid"""
    def __init__(self, config, years):
        """
        Parameters
        ----------
        config : ini
            config file for aggregating MODIS albedo to NSRDB grid.
        years : int
            Year to process.
        days : str
            Eight day periods that alebdo data is processed at ('001').
        """
        self.config = ConfigObj(config, unrepr=True)
        logger.debug(self.config)

        self.years = years
        self.days = [str(d).zfill(3) for d in range(1, 362, 8)]

    def sparseToSpatial(self):
        """Sparse to Spatial: Taking meta from HDF5 nsrdb 1998

        !!!Eventually: Take the meta from the extent files!!!

        """
        fname = os.path.join(
            self.config['dir']['nsrdb_dir'], self.config['fileNames']['nsrdb'])
        logger.debug('reading{}'.format(fname))

        with h5py.File(fname, 'r') as hfile:
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
        """
        Get a meta for the HDF4.
        Function grabs the albedo metadata.
        """
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
                    day='001', year=self.years))
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
            logger.exception(e)
            logger.info(fname)

    def getAggregationIndices(self):
        """
        Retrieve indicies for aggregation. (Really Assiging nearest pixel).
        This function does a cKDTree mapping the albedo data to the nsrdb data.
        """
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
            logger.exception(fname)

    def write_results(self, year, var, results):
        """Write results to HDF."""
        with h5py.File(
            os.path.join(
                self.config['dir']['out_dir'],
                self.config['fileNames']['outName'].format(
                    var=var, year=year)), 'w') as hfile:
            for key, values in results.items():
                hfile.create_dataset(key, data=values, dtype=np.float32)
            hfile.create_dataset('meta', data=self.meta)

    def processYear(self, fname):
        """Process a year."""
        try:
            year = fname[-8:-4]
            splt = fname.split('_')
            var = '{var}_{band}'.format(var=splt[1], band=splt[2])
            fnames = (
                [os.path.join(self.config['dir']['in_dir'],
                 fname.format(day=day)) for day in self.days])
            results = {'means': np.zeros(
                shape=(len(self.days), self.meta.shape[0]), dtype=np.float32)}
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
            logger.exception(e)

    def main(self, log_level='DEBUG'):
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
        init_logger(__name__, log_file='modis.log', log_level=log_level)
        init_logger('nsrdb.utilities.file_utils', log_file=None,
                    log_level=log_level)
        try:
            t0 = time.time()
            out_vars = [self.config['fileNames']['outName'].format(
                var=var,
                year=self.years)
                for var in self.config['variables']['variable_names']]
            files = os.listdir(self.config['dir']['out_dir'])
            out_vars = [var for var in out_vars if var not in files]
            out_vars = [(fname[-7:-3], '{var}_{band}'.format(
                var=fname.split('_')[1],
                band=fname.split('_')[2])) for fname in out_vars]
            logger.debug(out_vars)

            vars = [self.config['fileNames']['inName'].format(
                var=pair[1],
                day='{day}',
                year=pair[0]) for pair in out_vars]

            logger.info(vars)

            logger.info('sparse to spatial')
            self.sparseToSpatial()

            logger.info('get HDF4 meta')
            self.getHDF4meta()

            logger.info('aggregation indices')
            self.getAggregationIndices()

            logger.debug(
                'Memory: %s', str(np.max(memory_usage(
                    -1, interval=.2, timeout=1))))

            for var in vars:
                logger.debug('processing{}'.format(var))
                self.processYear(var)

            logger.info('process complete, time: %s',
                        (time.time() - t0) / 60.0)

            logger.info(mem_str())
        except Exception as e:
            # logger.info(PrintException())
            logger.exception(e)

    @classmethod
    def run(cls, config, years):
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
        init_logger(__name__, log_file=None, log_level='DEBUG')

        modis = cls(config, years)
        agg = modis.main()
        return agg


class Eightday_to_dailyMODIS(object):
    """Class to move MODIS variables to daily from 8 day time steps."""
    def __init__(self, config, years):
        """
        Parameters
        ----------
        config : ini
            config file for aggregating MODIS albedo to NSRDB grid.
        years : int
            Year to process.
        """
        self.config = ConfigObj(config, unrepr=True)
        self.years = years

    def main(self):
        """Convert MODIS variables to daily from 8 day time steps."""
        t0 = time.time()
        try:
            years = self.years
            rep_years = [str(y) for y in range(
                self.config['variables']['replicate_years'][0],
                self.config['variables']['replicate_years'][1])]

            hdfRows = [(366 if calendar.isleap(
                int(year)) else 365) for year in rep_years + years]
            rowInds = np.cumsum([0] + hdfRows)
            modisDays = [d for d in range(0, 361, 8)] + [365]

            with h5py.File(
                os.path.join(
                    self.config['dir']['out_dir'],
                    self.config['fileNames']['outName'].format(
                        year=2015)), 'r') as hfile:
                meta = hfile['meta'][...]

            with h5py.File(
                os.path.join(
                    self.config['dir']['out_dir'],
                    self.config['fileNames']['dailyFileName']), 'w') as hfile:
                hfile.create_dataset('meta', data=meta)
                hfile.create_dataset(
                    'albedo', shape=(
                        (sum(hdfRows), meta.shape[0])), dtype=np.float32)
            logger.info('hdf created')
            # replicate 8-day to daily
            for i, year in enumerate(rep_years + years):
                days = (366 if calendar.isleap(int(year)) else 365)
                modisDays[-1] = days
                rname = os.path.join(
                    self.config['dir']['out_dir'],
                    self.config['fileNames']['outName'].format(year=year))

                with h5py.File(rname, 'r') as hfile:
                    data = hfile['means'][...]
                replicated = np.empty((days, meta.shape[0]))
                for j, day in enumerate(modisDays[: - 1]):
                    replicated[day:modisDays[j + 1], :] = data[j, :]
                with h5py.File(
                    os.path.join(
                        self.config['dir']['out_dir'],
                        self.config['fileNames']['dailyFileName']),
                        'r+') as hfile:
                    hfile['albedo'][rowInds[i]:rowInds[i + 1], :] = replicated
                logger.info('year: %s complete', year)
        except Exception as e:
            # logger.info(PrintException())
            logger.exception(e)

        logger.info('main done, time: %s', (time.time() - t0) / 60.0)

    @classmethod
    def run(cls, config, years):
        """Class to move MODIS variables to daily from 8 day time steps.
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
        init_logger(__name__, log_file=None, log_level='DEBUG')

        modis = cls(config, years)
        eightday = modis.main()
        return eightday
