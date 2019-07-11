# -*- coding: utf-8 -*-
"""Entry point for NSRDB data pipeline execution.


Created on Thu Apr 25 15:47:53 2019

@author: gbuster
"""

from concurrent.futures import ProcessPoolExecutor
import datetime
import pandas as pd
import numpy as np
import os
import logging
import sys

from nsrdb.all_sky.all_sky import all_sky_h5, all_sky_h5_parallel
from nsrdb.data_model import DataModel, VarFactory
from nsrdb.gap_fill.cloud_fill import CloudGapFill
from nsrdb.file_handlers.outputs import Outputs
from nsrdb.file_handlers.collection import Collector
from nsrdb.utilities.execution import SLURM
from nsrdb.utilities.loggers import init_logger


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

    def __init__(self, out_dir, year, grid, freq='5min', cloud_extent='east',
                 var_meta=None):
        """
        Parameters
        ----------
        out_dir : str
            Target directory to dump all-sky-ready data files.
        year : int | str
            Processing year.
        grid : str | pd.DataFrame
            CSV file containing the NSRDB reference grid to interpolate to,
            or a pre-extracted (and reduced) dataframe. The first csv column
            must be the NSRDB site gid's.
        freq : str
            Final desired NSRDB temporal frequency.
        cloud_extent : str
            Regional (satellite) extent to process for cloud data processing,
            used to form file paths to cloud data files.
        var_meta : str | None
            File path to NSRDB variables meta data. None will use the default
            file from the github repo.
        """

        self._out_dir = out_dir
        self.make_out_dir(self._out_dir)
        self._year = int(year)
        self._grid = grid
        self._freq = freq
        self._cloud_extent = cloud_extent
        self._var_meta = var_meta
        self._ti = None

    @staticmethod
    def make_out_dir(out_dir):
        """Ensure that out_dir exists.

        Parameters
        ----------
        out_dir : str
            Directory to put output files
        """
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

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

    def _exe_daily_data_model(self, month, day, var_list=None, parallel=True,
                              fpath_out=None):
        """Execute the data model for a single day.

        Parameters
        ----------
        month : int | str
            Month to run data model for.
        day : int | str
            Day to run data model for.
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
            var_list, date, self._grid, nsrdb_freq=self._freq,
            var_meta=self._var_meta, parallel=parallel,
            cloud_extent=self._cloud_extent, return_obj=True,
            fpath_out=fpath_out)

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
        """Get the file output path based on a date. Will have {var} and {i}.

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
        fpath_out = os.path.join(self._out_dir, fname)
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
                           'nsrdb.file_handlers', 'nsrdb.all_sky')

            log_file = os.path.join(self._out_dir, log_file)

            if isinstance(date, datetime.date):
                date_str = ('{}{}{}'.format(date.year,
                                            str(date.month).zfill(2),
                                            str(date.day).zfill(2)))
                log_file = log_file.replace('.log', '_{}.log'.format(date_str))

            for name in loggers:
                init_logger(name, log_level=log_level, log_file=log_file)

        if log_version:
            self._log_py_version()

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
        date = (datetime.datetime(int(year), 1, 1) +
                datetime.timedelta(int(doy) - 1))
        datestr = '{}{}{}'.format(date.year,
                                  str(date.month).zfill(2),
                                  str(date.day).zfill(2))
        return datestr

    @staticmethod
    def date_to_doy(date):
        """Convert a date to a day of year integer.

        Parameters
        ----------
        date : datetime.date
            Date object.

        Returns
        -------
        doy : int
            Day of year.
        """
        return date.timetuple().tm_yday

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
    def run_data_model(cls, out_dir, date, grid, freq='5min',
                       cloud_extent='east', parallel=True, log_level='DEBUG'):
        """Run daily data model, and save output files.

        Parameters
        ----------
        out_dir : str
            Target directory to dump data model output files.
        date : datetime.date | str | int
            Single day to extract ancillary data for.
            Can be str or int in YYYYMMDD format.
        grid : str | pd.DataFrame
            CSV file containing the NSRDB reference grid to interpolate to,
            or a pre-extracted (and reduced) dataframe. The first csv column
            must be the NSRDB site gid's.
        freq : str
            Final desired NSRDB temporal frequency.
        cloud_extent : str
            Regional (satellite) extent to process for cloud data processing,
            used to form file paths to cloud data files.
        parallel : bool
            Flag to perform data model processing in parallel.
        log_level : str | None
            Logging level (DEBUG, INFO). If None, no logging will be
            initialized.
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

        nsrdb = cls(out_dir, date.year, grid, freq=freq,
                    cloud_extent=cloud_extent)
        nsrdb._init_loggers(date=date, log_file='nsrdb_data_model.log',
                            log_level=log_level)

        fpath_out = nsrdb._get_fpath_out(date)

        data_model = nsrdb._exe_daily_data_model(date.month, date.day,
                                                 parallel=parallel,
                                                 fpath_out=fpath_out)

        if fpath_out is None:
            nsrdb._exe_fout(data_model)

    @classmethod
    def collect_data_model(cls, daily_dir, out_dir, year, grid, freq='5min',
                           log_level='DEBUG'):
        """Init output file and collect daily data model output files.

        Parameters
        ----------
        daily_dir : str
            Directory with daily files to be collected.
        out_dir : str
            Directory to put final output files.
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
        """

        nsrdb = cls(out_dir, year, grid, freq=freq, cloud_extent=None)
        nsrdb._init_loggers(log_file='nsrdb_collect_dm.log',
                            log_level=log_level)

        for fname, dsets in cls.OUTS.items():
            if 'irradiance' not in fname:
                f_out = os.path.join(out_dir, fname.format(y=year))
                nsrdb._init_final_out(f_out, dsets, nsrdb.time_index_year,
                                      nsrdb.meta)
                Collector.collect(daily_dir, f_out, dsets)

    @classmethod
    def collect_data_model_chunks(cls, daily_dir, out_dir, year, grid,
                                  n_chunks=1, freq='5min', log_level='DEBUG',
                                  parallel=True):
        """Init site-chunked output file and collect daily data model outputs.

        Parameters
        ----------
        daily_dir : str
            Directory with daily files to be collected.
        out_dir : str
            Directory to put final output files.
        year : int | str
            Year of analysis
        grid : str
            Final/full NSRDB grid file. The first column must be the NSRDB
            site gid's.
        n_chunks : int
            Number of chunks (site-wise) to collect to.
        freq : str
            Final desired NSRDB temporal frequency.
        log_level : str | None
            Logging level (DEBUG, INFO). If None, no logging will be
            initialized.
        parallel : bool
            Flag to do chunk collection in parallel.
        """

        nsrdb = cls(out_dir, year, grid, freq=freq, cloud_extent=None)
        nsrdb._init_loggers(log_file='nsrdb_collect_dm.log',
                            log_level=log_level)

        chunks = np.array_split(range(len(nsrdb.meta)), n_chunks)

        for fname, dsets in cls.OUTS.items():
            if 'irradiance' not in fname:
                f_out = os.path.join(out_dir, fname.format(y=year))

                if not parallel:
                    for i, chunk in enumerate(chunks):
                        nsrdb.collect_chunk(daily_dir, i, chunk, f_out, dsets,
                                            nsrdb.time_index_year, nsrdb.meta)
                else:
                    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
                        for i, chunk in enumerate(chunks):
                            ex.submit(nsrdb.collect_chunk, daily_dir, i,
                                      chunk, f_out, dsets,
                                      nsrdb.time_index_year,
                                      nsrdb.meta)

    @staticmethod
    def collect_chunk(daily_dir, i, chunk, f_out, dsets, time_index, meta):
        """Collect a single chunk

        Parameters
        ----------
        daily_dir : str
            Directory with daily files to be collected.
        i : int
            Chunk number/enumeration.
        chunk : np.ndarray
            Array of site indices in meta that make up this chunk.
        f_out : str
            Output file path (with dir).
        dsets : list
            Datasets to collect
        time_index : pd.datetimeindex
            Full annual time index being collected
        meta : pd.DataFrame
            Full meta being collected (will be indexed with chunk).
        """

        f_out_i = f_out.replace('.h5', '_{}.h5'.format(i))
        NSRDB._init_final_out(f_out_i, dsets, time_index, meta.iloc[chunk, :])
        Collector.collect(daily_dir, f_out_i, dsets, sites=chunk)

    @staticmethod
    def gap_fill_clouds(f_cloud, rows=slice(None), cols=slice(None),
                        col_chunk=1000):
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
        """

        CloudGapFill.fill_file(f_cloud, rows=rows, cols=cols,
                               col_chunk=col_chunk)

    @classmethod
    def run_all_sky(cls, out_dir, year, grid, freq='5min',
                    rows=slice(None), cols=slice(None), parallel=True,
                    log_level='DEBUG'):
        """Run the all-sky physics model from .h5 files.

        Parameters
        ----------
        out_dir : str
            Directory containing ancillary and cloud output files, also where
            all-sky output files will go.
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
        """

        nsrdb = cls(out_dir, year, grid, freq=freq, cloud_extent=None)
        nsrdb._init_loggers(log_file='nsrdb_all_sky.log', log_level=log_level)

        for fname, dsets in cls.OUTS.items():
            if 'irradiance' in fname:
                f_out = os.path.join(out_dir, fname.format(y=year))
                nsrdb._init_final_out(f_out, dsets, nsrdb.time_index_year,
                                      nsrdb.meta)
            elif 'ancil' in fname:
                f_ancillary = os.path.join(out_dir, fname.format(y=year))
            elif 'cloud' in fname:
                f_cloud = os.path.join(out_dir, fname.format(y=year))

        if parallel:
            out = all_sky_h5_parallel(f_ancillary, f_cloud, rows=rows,
                                      cols=cols)
        else:
            out = all_sky_h5(f_ancillary, f_cloud, rows=rows, cols=cols)

        logger.info('Finished all-sky. Writing results to: {}'.format(f_out))

        with Outputs(f_out, mode='a') as f:
            for dset, arr in out.items():
                f[dset, rows, cols] = arr

    @staticmethod
    def eagle(fun_str, arg_str, alloc='pxs', memory=90, walltime=2,
              feature='--qos=normal', node_name='nsrdb', stdout_path=None):
        """Run an NSRDB class or static method on an Eagle node.

        Format: NSRDB.fun_str(arg_str)

        Parameters
        ----------
        fun_str : str
            Name of the class or static method belonging to the NSRDB class
            to execute in the SLURM job.
        arg_str : str
            Arguments passed to the target method.
        alloc : str
            SLURM project allocation.
        memory : int
            Node memory request in GB.
        walltime : int
            Node walltime request in hours.
        feature : str
            Additional flags for SLURM job. Format is "--qos=high"
            or "--depend=[state:job_id]". Default is None.
        node_name : str
            Name for the SLURM job.
        stdout_path : str
            Path to dump the stdout/stderr files.
        """

        if stdout_path is None:
            stdout_path = os.getcwd()

        cmd = "python -c 'from nsrdb.nsrdb import NSRDB;NSRDB.{f}({a})'"

        cmd = cmd.format(f=fun_str, a=arg_str)

        slurm = SLURM(cmd, alloc=alloc, memory=memory, walltime=walltime,
                      feature=feature, name=node_name, stdout_path=stdout_path)

        print('\ncmd:\n{}\n'.format(cmd))

        if slurm.id:
            msg = ('Kicked off job "{}" (SLURM jobid #{}) on '
                   'Eagle.'.format(node_name, slurm.id))
        else:
            msg = ('Was unable to kick off job "{}". '
                   'Please see the stdout error messages'
                   .format(node_name))
        print(msg)
