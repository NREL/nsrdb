# -*- coding: utf-8 -*-
"""Entry point for NSRDB data pipeline execution.


Created on Thu Apr 25 15:47:53 2019

@author: gbuster
"""

import datetime
import pandas as pd
import numpy as np
import os
import logging
import sys
import shutil
import time

from nsrdb.all_sky.all_sky import all_sky_h5, all_sky_h5_parallel
from nsrdb.data_model import DataModel, VarFactory
from nsrdb.gap_fill.cloud_fill import CloudGapFill
from nsrdb.file_handlers.outputs import Outputs
from nsrdb.file_handlers.collection import Collector
from nsrdb.utilities.loggers import init_logger
from nsrdb.pipeline.status import Status


logger = logging.getLogger(__name__)


class NSRDB:
    """Entry point for NSRDB data pipeline execution."""

    OUTS = {'nsrdb_ancillary_{y}.h5': ('alpha',
                                       'aod',
                                       'asymmetry',
                                       'ozone',
                                       'ssa',
                                       'surface_albedo',
                                       'surface_pressure',
                                       'total_precipitable_water'),
            'nsrdb_sam_{y}.h5': ('dew_point',
                                 'relative_humidity',
                                 'air_temperature',
                                 'surface_pressure',
                                 'wind_direction',
                                 'wind_speed'),
            'nsrdb_clouds_{y}.h5': ('cloud_type',
                                    'cld_opd_dcomp',
                                    'cld_reff_dcomp',
                                    'cld_press_acha',
                                    'fill_flag',
                                    'solar_zenith_angle'),
            'nsrdb_irradiance_{y}.h5': ('dhi',
                                        'dni',
                                        'ghi',
                                        'clearsky_dhi',
                                        'clearsky_dni',
                                        'clearsky_ghi',
                                        'fill_flag')}

    def __init__(self, out_dir, year, grid, freq='5min', var_meta=None):
        """
        Parameters
        ----------
        out_dir : str
            Project directory.
        year : int | str
            Processing year.
        grid : str | pd.DataFrame
            CSV file containing the NSRDB reference grid to interpolate to,
            or a pre-extracted (and reduced) dataframe. The first csv column
            must be the NSRDB site gid's.
        freq : str
            Final desired NSRDB temporal frequency.
        var_meta : str | None
            File path to NSRDB variables meta data. None will use the default
            file from the github repo.
        """

        self._out_dir = out_dir
        self._log_dir = os.path.join(out_dir, 'logs/')
        self._data_dir = os.path.join(out_dir, 'data/')
        self._final_dir = os.path.join(out_dir, 'final/')
        self.make_out_dir()
        self._year = int(year)
        self._grid = grid
        self._freq = freq
        self._var_meta = var_meta
        self._ti = None

    def make_out_dir(self):
        """Ensure that out_dir exists."""
        for d in [self._out_dir, self._log_dir, self._data_dir]:
            if not os.path.exists(d):
                os.makedirs(d)

    @property
    def time_index_year(self):
        """Get the NSRDB full-year time index.

        Returns
        -------
        nsrdb_ti : pd.DatetimeIndex
            Pandas datetime index for the current year at the NSRDB resolution.
        """
        if self._ti is None:
            self._ti = pd.date_range('1-1-{y}'.format(y=self._year),
                                     '1-1-{y}'.format(y=self._year + 1),
                                     freq=self._freq)[:-1]
        return self._ti

    @property
    def meta(self):
        """Get the NSRDB meta dataframe from the grid file.

        Returns
        -------
        meta : pd.DataFrame
            DataFrame of meta data from grid file csv.
            The first column must be the NSRDB site gid's.
        """

        if isinstance(self._grid, str):
            self._grid = pd.read_csv(self._grid, index_col=0)
        return self._grid

    @staticmethod
    def _log_py_version():
        """Check python version and 64-bit and print to logger."""

        logger.info('Running python version: {}'.format(sys.version_info))

        is_64bits = sys.maxsize > 2 ** 32
        if is_64bits:
            logger.info('Running on 64-bit python, sys.maxsize: {}'
                        .format(sys.maxsize))
        else:
            logger.warning('Running 32-bit python, sys.maxsize: {}'
                           .format(sys.maxsize))

    def _exe_daily_data_model(self, month, day, cloud_dir, var_list=None,
                              parallel=True, fpath_out=None):
        """Execute the data model for a single day.

        Parameters
        ----------
        month : int | str
            Month to run data model for.
        day : int | str
            Day to run data model for.
        cloud_dir : str
            Cloud data directory containing nested daily directories with
            h5 or nc files from UW.
        var_list : list | tuple | None
            Variables to process with the data model. None will default to all
            variables.
        parallel : bool
            Flag to perform data model processing in parallel.
        fpath_out : str | None
            File path to dump results. If no file path is given, results will
            be returned as an object.

        Returns
        -------
        data_model : nsrdb.data_model.DataModel
            Daily data model object.
        """

        date = datetime.date(year=self._year, month=int(month), day=int(day))

        if var_list is None:
            var_list = DataModel.ALL_VARS

        logger.info('Starting daily data model execution for {}-{}-{}'
                    .format(month, day, self._year))

        # run data model
        data_model = DataModel.run_multiple(
            var_list, date, cloud_dir, self._grid, nsrdb_freq=self._freq,
            var_meta=self._var_meta, parallel=parallel,
            return_obj=True, fpath_out=fpath_out)

        logger.info('Finished daily data model execution for {}-{}-{}'
                    .format(month, day, self._year))

        return data_model

    def _exe_fout(self, data_model):
        """Send the single-day data model results to output files.

        Parameters
        ----------
        data_model : nsrdb.data_model.DataModel
            Daily data model object.
        """

        logger.info('Starting file export of daily data model results to: {}'
                    .format(self._out_dir))

        # output handling for each entry in data model
        for var, arr in data_model._processed.items():
            if var not in ['time_index', 'meta']:

                fpath_out = self._get_fpath_out(data_model.date)
                fpath_out = fpath_out.format(var=var, i=self.meta.index[0])

                logger.debug('\tWriting file: {}'
                             .format(os.path.basename(fpath_out)))

                # make file for each var
                with Outputs(fpath_out, mode='w') as fout:
                    fout.time_index = data_model.nsrdb_ti
                    fout.meta = data_model.nsrdb_grid

                    var_obj = VarFactory.get_base_handler(
                        var, var_meta=self._var_meta, date=data_model.date)
                    attrs = var_obj.attrs

                    fout._add_dset(dset_name=var, data=arr,
                                   dtype=var_obj.final_dtype,
                                   chunks=var_obj.chunks, attrs=attrs)

        logger.info('Finished file export of daily data model results to: {}'
                    .format(self._out_dir))

    def _get_fpath_out(self, date):
        """Get the data model file output path based on a date.
        Will have {var} and {i}.

        Parameters
        ----------
        date : datetime.date
            Single day for the output file.

        Returns
        -------
        fpath_out : str
            Full file path with directory. format is /dir/YYYYMMDD_{var}_{i}.h5
        """

        fname = ('{}{}{}'.format(date.year,
                                 str(date.month).zfill(2),
                                 str(date.day).zfill(2)))
        fname += '_{var}_{i}.h5'
        fpath_out = os.path.join(self._data_dir, fname)
        return fpath_out

    def _init_loggers(self, loggers=None, log_file='nsrdb.log',
                      log_level='DEBUG', date=None, log_version=True):
        """Initialize nsrdb loggers.

        Parameters
        ----------
        loggers : None | list | tuple
            List of logger names to initialize. None defaults to all NSRDB
            loggers.
        log_file : str
            Log file name. Will be placed in the nsrdb out dir.
        log_level : str | None
            Logging level (DEBUG, INFO). If None, no logging will be
            initialized.
        date : None | datetime
            Optional date to put in the log file name.
        """

        if log_level in ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'):

            if loggers is None:
                loggers = ('nsrdb.nsrdb', 'nsrdb.data_model',
                           'nsrdb.file_handlers', 'nsrdb.all_sky',
                           'nsrdb.gap_fill')

            log_file = os.path.join(self._log_dir, log_file)

            if isinstance(date, datetime.date):
                date_str = ('{}{}{}'.format(date.year,
                                            str(date.month).zfill(2),
                                            str(date.day).zfill(2)))
                log_file = log_file.replace('.log', '_{}.log'.format(date_str))

            for name in loggers:
                init_logger(name, log_level=log_level, log_file=log_file)

        if log_version:
            self._log_py_version()

    @staticmethod
    def _init_final_out(f_out, dsets, time_index, meta):
        """Initialize the final output file.

        Parameters
        ----------
        f_out : str
            File path to final .h5 file.
        dsets : list
            List of dataset / variable names that are to be contained in f_out.
        time_index : pd.datetimeindex
            Time index to init to file.
        meta : pd.DataFrame
            Meta data to init to file.
        """

        if not os.path.isfile(f_out):
            logger.info('Initializing {} for the following datasets: {}'
                        .format(f_out, dsets))

            attrs, chunks, dtypes = NSRDB.get_dset_attrs(dsets)

            Outputs.init_h5(f_out, dsets, attrs, chunks, dtypes,
                            time_index, meta)

    @staticmethod
    def doy_to_datestr(year, doy):
        """Convert day of year to YYYYMMDD string format

        Parameters
        ----------
        year : int
            Year of interest
        doy : int
            Enumerated day of year.

        Returns
        -------
        date : str
            Single day to extract ancillary data for.
            str in YYYYMMDD format.
        """
        date = (datetime.datetime(int(year), 1, 1)
                + datetime.timedelta(int(doy) - 1))
        datestr = '{}{}{}'.format(date.year,
                                  str(date.month).zfill(2),
                                  str(date.day).zfill(2))
        return datestr

    @staticmethod
    def date_to_doy(date):
        """Convert a date to a day of year integer.

        Parameters
        ----------
        date : datetime.date | str | int
            Single day to extract ancillary data for.
            Can be str or int in YYYYMMDD format.

        Returns
        -------
        doy : int
            Day of year.
        """
        return NSRDB.to_datetime(date).timetuple().tm_yday

    @staticmethod
    def to_datetime(date):
        """Convert a date string or integer to datetime object.

        Parameters
        ----------
        date : datetime.date | str | int
            Single day to extract ancillary data for.
            Can be str or int in YYYYMMDD format.

        Returns
        -------
        date : datetime.date
            Date object.
        """

        if isinstance(date, (int, float)):
            date = str(int(date))
        if isinstance(date, str):
            if len(date) == 8:
                date = datetime.date(year=int(date[0:4]),
                                     month=int(date[4:6]),
                                     day=int(date[6:]))
            else:
                raise ValueError('Could not parse date: {}'.format(date))

        return date

    @staticmethod
    def get_dset_attrs(dsets):
        """Get output file dataset attributes for a set of datasets.

        Parameters
        ----------
        dsets : list
            List of dataset / variable names.

        Returns
        -------
        attrs : dict
            Dictionary of dataset attributes keyed by dset name.
        chunks : dict
            Dictionary of chunk tuples keyed by dset name.
        dtypes : dict
            dictionary of numpy datatypes keyed by dset name.
        """

        attrs = {}
        chunks = {}
        dtypes = {}

        for dset in dsets:
            var_obj = VarFactory.get_base_handler(dset)
            attrs[dset] = var_obj.attrs
            chunks[dset] = var_obj.chunks
            dtypes[dset] = var_obj.final_dtype

        return attrs, chunks, dtypes

    @classmethod
    def run_data_model(cls, out_dir, date, cloud_dir, grid, freq='5min',
                       parallel=True, log_level='DEBUG',
                       log_file='data_model.log', job_name=None):
        """Run daily data model, and save output files.

        Parameters
        ----------
        out_dir : str
            Project directory.
        date : datetime.date | str | int
            Single day to extract ancillary data for.
            Can be str or int in YYYYMMDD format.
        cloud_dir : str
            Cloud data directory containing nested daily directories with
            h5 or nc files from UW.
        grid : str | pd.DataFrame
            CSV file containing the NSRDB reference grid to interpolate to,
            or a pre-extracted (and reduced) dataframe. The first csv column
            must be the NSRDB site gid's.
        freq : str
            Final desired NSRDB temporal frequency.
        parallel : bool
            Flag to perform data model processing in parallel.
        log_level : str | None
            Logging level (DEBUG, INFO). If None, no logging will be
            initialized.
        log_file : str
            File to log to. Will be put in output directory.
        job_name : str
            Optional name for pipeline and status identification.
        """

        t0 = time.time()
        date = cls.to_datetime(date)

        nsrdb = cls(out_dir, date.year, grid, freq=freq)
        nsrdb._init_loggers(date=date, log_file=log_file, log_level=log_level)

        fpath_out = nsrdb._get_fpath_out(date)

        data_model = nsrdb._exe_daily_data_model(date.month, date.day,
                                                 cloud_dir, parallel=parallel,
                                                 fpath_out=fpath_out)

        if fpath_out is None:
            nsrdb._exe_fout(data_model)

        runtime = (time.time() - t0) / 60
        status = {'out_dir': nsrdb._out_dir,
                  'fout': fpath_out,
                  'job_status': 'successful',
                  'runtime': runtime,
                  'grid': grid,
                  'freq': freq,
                  'cloud_dir': cloud_dir,
                  'data_model_date': date}
        Status.make_job_file(nsrdb._out_dir, 'data-model', job_name, status)

    @classmethod
    def collect_data_model(cls, daily_dir, out_dir, year, grid, freq='5min',
                           log_level='DEBUG', log_file='collect_dm.log'):
        """Init output file and collect daily data model output files.

        Parameters
        ----------
        daily_dir : str
            Directory with daily files to be collected.
        out_dir : str
            Project directory.
        year : int | str
            Year of analysis
        grid : str
            Final/full NSRDB grid file. The first column must be the NSRDB
            site gid's.
        freq : str
            Final desired NSRDB temporal frequency.
        log_level : str | None
            Logging level (DEBUG, INFO). If None, no logging will be
            initialized.
        log_file : str
            File to log to. Will be put in output directory.
        """

        nsrdb = cls(out_dir, year, grid, freq=freq)
        nsrdb._init_loggers(log_file=log_file, log_level=log_level)

        for fname, dsets in cls.OUTS.items():
            if 'irradiance' not in fname:
                f_out = os.path.join(nsrdb._data_dir, fname.format(y=year))
                nsrdb._init_final_out(f_out, dsets, nsrdb.time_index_year,
                                      nsrdb.meta)
                Collector.collect(daily_dir, f_out, dsets)
        logger.info('Finished file collection to: {}'.format(out_dir))

    @classmethod
    def collect_data_model_chunk(cls, daily_dir, out_dir, year, grid,
                                 n_chunks, i_chunk, i_fname,
                                 freq='5min', log_level='DEBUG',
                                 log_file='collect_dm.log',
                                 parallel=True):
        """Collect daily data model files to a single site-chunked output file.

        Parameters
        ----------
        daily_dir : str
            Directory with daily files to be collected.
        out_dir : str
            Project directory.
        year : int | str
            Year of analysis
        grid : str
            Final/full NSRDB grid file. The first column must be the NSRDB
            site gid's.
        n_chunks : int
            Number of chunks (site-wise) to collect to.
        i_chunks : int
            Chunk index (indexing n_chunks) to run.
        i_fname : int
            File name index from NSRDB.OUTS to run collection for:
                0 - ancillary
                1 - clouds
                2 - sam vars
        freq : str
            Final desired NSRDB temporal frequency.
        log_level : str | None
            Logging level (DEBUG, INFO). If None, no logging will be
            initialized.
        log_file : str
            File to log to. Will be put in output directory.
        parallel : bool | int
            Flag to do chunk collection in parallel. Can be integer number of
            workers to use (number of parallel reads).
        """

        nsrdb = cls(out_dir, year, grid, freq=freq)
        nsrdb._init_loggers(log_file=log_file, log_level=log_level)

        chunks = np.array_split(range(len(nsrdb.meta)), n_chunks)

        fnames = sorted(list(cls.OUTS.keys()))
        fnames = [fn for fn in fnames if 'irradiance' not in fn]

        chunk = chunks[i_chunk]
        fname = fnames[i_fname]
        dsets = cls.OUTS[fname]

        if '{y}' in fname:
            fname = fname.format(y=year)

        f_out = os.path.join(out_dir, fname)
        f_out = f_out.replace('.h5', '_{}.h5'.format(i_chunk))

        meta_chunk = nsrdb.meta.iloc[chunk, :]
        if 'gid' not in meta_chunk:
            meta_chunk['gid'] = meta_chunk.index

        logger.info('Running data model collection for chunk {} out of {} '
                    'with meta gid {} to {} and target file: {}'
                    .format(i_chunk, n_chunks, meta_chunk['gid'].values[0],
                            meta_chunk['gid'].values[-1], f_out))

        NSRDB._init_final_out(f_out, dsets, nsrdb.time_index_year, meta_chunk)
        Collector.collect(daily_dir, f_out, dsets, sites=chunk,
                          parallel=parallel)
        logger.info('Finished file collection to: {}'.format(f_out))

    @classmethod
    def collect_final(cls, collect_dir, out_dir, year, grid, freq='5min',
                      i_fname=None, tmp=False,
                      log_level='DEBUG', log_file='final_collection.log'):
        """Collect chunked files to single final output files.

        Parameters
        ----------
        collect_dir : str
            Directory with chunked files to be collected.
        out_dir : str
            Project directory.
        year : int | str
            Year of analysis
        grid : str
            Final/full NSRDB grid file. The first column must be the NSRDB
            site gid's.
        freq : str
            Final desired NSRDB temporal frequency.
        log_level : str | None
            Logging level (DEBUG, INFO). If None, no logging will be
            initialized.
        log_file : str
            File to log to. Will be put in output directory.
        i_fname : int | None
            Optional index to collect just a single output file. Indexes the
            sorted OUTS class attribute keys.
        tmp : bool
            Flag to use temporary scratch storage, then move to out_dir when
            finished. Doesn't seem to be faster than collecting to normal
            scratch on eagle.
        """

        nsrdb = cls(out_dir, year, grid, freq=freq)
        nsrdb._init_loggers(log_file=log_file, log_level=log_level)

        fnames = sorted(list(cls.OUTS.keys()))
        if i_fname is not None:
            fnames = [fnames[i_fname]]

        for fname in fnames:
            dsets = cls.OUTS[fname]
            fname = fname.format(y=year)

            if tmp:
                f_out = os.path.join('/tmp/scratch/', fname)
            else:
                f_out = os.path.join(nsrdb._final_dir, fname)

            flist = [fn for fn in os.listdir(collect_dir)
                     if fn.endswith('.h5')
                     and fname.replace('.h5', '') in fn]
            flist = sorted(flist, key=lambda x: float(
                x.replace('.h5', '').split('_')[-1]))
            fids = [int(fn.replace('.h5', '').split('_')[-1]) for fn in flist]
            if fids != list(range(np.min(fids), 1 + np.max(fids))):
                emsg = ('File list appears to be missing files. '
                        '{} files from {} to {}.'
                        .format(len(flist), np.min(fids), np.max(fids)))
                raise FileNotFoundError(emsg)

            if any(flist):
                nsrdb._init_final_out(f_out, dsets, nsrdb.time_index_year,
                                      nsrdb.meta)
                logger.info('Collecting {} files in list: {}'
                            .format(len(flist), flist))

                for dset in dsets:
                    logger.info('Collecting dataset "{}".'.format(dset))
                    Collector.collect_flist_lowmem(flist, collect_dir, f_out,
                                                   dset)

            else:
                emsg = ('Could not find files to collect for {} in the '
                        'collect dir: {}'
                        .format(fname, collect_dir))
                raise FileNotFoundError(emsg)

            if tmp:
                logger.info('Moving temp file to final output directory.')
                shutil.move(f_out, os.path.join(out_dir, fname))

        logger.info('Finished final file collection to: {}'.format(out_dir))

    @classmethod
    def gap_fill_clouds(cls, f_cloud, rows=slice(None), cols=slice(None),
                        col_chunk=1000, log_level='DEBUG',
                        log_file='cloud_fill.log'):
        """Gap fill cloud properties in a collected data model output file.

        Parameters
        ----------
        f_cloud : str
            File path to a cloud property file generated by the
            collect_data_model() method.
        rows : slice
            Subset of rows to gap fill.
        cols : slice
            Subset of columns to gap fill.
        col_chunks : None
            Optional chunking method to gap fill a few chunks at a time
            to reduce memory requirements.
        log_level : str | None
            Logging level (DEBUG, INFO). If None, no logging will be
            initialized.
        log_file : str
            File to log to. Will be put in output directory.
        """
        nsrdb = cls(os.path.dirname(f_cloud), 2000, None)
        nsrdb._init_loggers(log_file=log_file, log_level=log_level)
        CloudGapFill.fill_file(f_cloud, rows=rows, cols=cols,
                               col_chunk=col_chunk)
        logger.info('Finished cloud gap fill.')

    @classmethod
    def run_all_sky(cls, out_dir, year, grid, freq='5min',
                    rows=slice(None), cols=slice(None), parallel=True,
                    log_level='DEBUG', log_file='all_sky.log',
                    i_chunk=None):
        """Run the all-sky physics model from .h5 files.

        Parameters
        ----------
        out_dir : str
            Project directory.
        year : int | str
            Year of analysis
        grid : str
            Final/full NSRDB grid file. The first column must be the NSRDB
            site gid's.
        freq : str
            Final desired NSRDB temporal frequency.
        rows : slice
            Subset of rows to run.
        cols : slice
            Subset of columns to run.
        log_level : str | None
            Logging level (DEBUG, INFO). If None, no logging will be
            initialized.
        log_file : str
            File to log to. Will be put in output directory.
        i_chunk : None | int
            Enumerated file index if running on site chunk.
        """

        nsrdb = cls(out_dir, year, grid, freq=freq)
        nsrdb._init_loggers(log_file=log_file, log_level=log_level)

        for fname, dsets in cls.OUTS.items():
            if 'irradiance' in fname:
                f_out = os.path.join(nsrdb._data_dir,
                                     fname.format(y=year))
                irrad_dsets = dsets
            elif 'ancil' in fname:
                f_ancillary = os.path.join(nsrdb._data_dir,
                                           fname.format(y=year))
            elif 'cloud' in fname:
                f_cloud = os.path.join(nsrdb._data_dir,
                                       fname.format(y=year))

        if i_chunk is not None:
            f_out = f_out.replace('.h5', '_{}.h5'.format(i_chunk))
            f_ancillary = f_ancillary.replace('.h5', '_{}.h5'.format(i_chunk))
            f_cloud = f_cloud.replace('.h5', '_{}.h5'.format(i_chunk))

        with Outputs(f_ancillary) as out:
            meta = out.meta
            time_index = out.time_index

        nsrdb._init_final_out(f_out, irrad_dsets, time_index, meta)

        if parallel:
            out = all_sky_h5_parallel(f_ancillary, f_cloud, rows=rows,
                                      cols=cols)
        else:
            out = all_sky_h5(f_ancillary, f_cloud, rows=rows, cols=cols)

        logger.info('Finished all-sky. Writing results to: {}'.format(f_out))

        with Outputs(f_out, mode='a') as f:
            for dset, arr in out.items():
                f[dset, rows, cols] = arr

        logger.info('Finished writing results to: {}'.format(f_out))
