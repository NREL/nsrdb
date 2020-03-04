# -*- coding: utf-8 -*-
"""NSRDB chunked file collection tools.
"""
import time
import datetime
import json
import numpy as np
import pandas as pd
import os
import logging
import psutil
from concurrent.futures import ProcessPoolExecutor, as_completed

from nsrdb.data_model import VarFactory
from nsrdb.file_handlers.outputs import Outputs
from nsrdb.pipeline.status import Status
from nsrdb.utilities.loggers import init_logger


logger = logging.getLogger(__name__)


class Collector:
    """NSRDB file collection framework"""

    def __init__(self, collect_dir, dset):
        """
        Parameters
        ----------
        collect_dir : str
            Directory that files are being collected from
        dset : str
            Dataset/var name that is searched for in file names in collect_dir.
        """

        self.flist = self.get_flist(collect_dir, dset)

        if not any(self.flist):
            raise FileNotFoundError('No "{}" files found in {}'
                                    .format(dset, collect_dir))

        self.verify_flist(self.flist, collect_dir, dset)

    @staticmethod
    def verify_flist(flist, d, var):
        """Verify the correct number of files in d for var. Raise if bad flist.

        Filename requirements:
         - Expects file names with leading "YYYYMMDD_".
         - Must have var in the file name.
         - Should end with ".h5"

        Parameters
        ----------
        flist : list
            List of .h5 files in directory d that contain the var string.
            Sorted by integer before the first underscore in the filename.
        d : str
            Directory to get file list from.
        var : str
            Variable name that is searched for in files in d.
        """

        date_str_list = [f.split('_')[0] for f in flist]
        date_str_list = sorted(date_str_list, key=int)
        date_str_start = date_str_list[0]
        date_str_end = date_str_list[-1]

        if len(date_str_start) == 8 and len(date_str_end) == 8:
            date_start = datetime.date(year=int(date_str_start[0:4]),
                                       month=int(date_str_start[4:6]),
                                       day=int(date_str_start[6:]))
            date_end = datetime.date(year=int(date_str_end[0:4]),
                                     month=int(date_str_end[4:6]),
                                     day=int(date_str_end[6:]))
        else:
            raise ValueError('Could not parse date: {}'.format(date_str_start))

        date_end += datetime.timedelta(days=1)
        ti = pd.date_range(start=date_start, end=date_end,
                           freq='1D', closed='left')

        missing = []
        for date in ti:
            date_str = ('{}{}{}'.format(date.year, str(date.month).zfill(2),
                                        str(date.day).zfill(2)))
            if date_str not in date_str_list:
                missing.append(date_str)

        if missing:
            raise FileNotFoundError('Missing the following date files for '
                                    '"{}":\n{}'.format(var, missing))

        logger.info('Good file count of {} for "{}" in year {} in dir: {}'
                    .format(len(flist), var, date_start.year, d))

    @staticmethod
    def get_flist(d, var):
        """Get a date-sorted .h5 file list for a given var.

        Filename requirements:
         - Expects file names with leading "YYYYMMDD_".
         - Must have var in the file name.
         - Should end with ".h5"

        Parameters
        ----------
        d : str
            Directory to get file list from.
        var : str
            Variable name that is searched for in files in d.

        Returns
        -------
        flist : list
            List of .h5 files in directory d that contain the var string.
            Sorted by integer before the first underscore in the filename.
        """

        flist = []
        temp = os.listdir(d)
        flist = [f for f in flist if f.endswith('.h5')]

        for fn in temp:
            fp = os.path.join(d, fn)
            with Outputs(fp, mode='r') as fobj:
                if var in fobj.dsets:
                    flist.append(fn)

        flist = sorted(flist, key=lambda x: int(x.split('_')[0]))

        return flist

    @staticmethod
    def get_slices(final_time_index, final_meta,
                   new_time_index, new_meta):
        """Get index slices where the new ti/meta belong in the final ti/meta.

        Parameters
        ----------
        final_time_index : pd.Datetimeindex
            Time index of the final file that new_time_index is being written
            to.
        final_meta : pd.DataFrame
            Meta data of the final file that new_meta is being written to.
        new_time_index : pd.Datetimeindex
            Chunk time index that is a subset of the final_time_index.
        new_meta : pd.DataFrame
            Chunk meta data that is a subset of the final_meta.

        Returns
        -------
        row_slice : slice
            final_time_index[row_slice] = new_time_index
        col_slice : slice
            final_meta[col_slice] = new_meta
        """

        # pylint: disable-msg=C0121
        row_loc = np.where(
            final_time_index.isin(new_time_index) == True)[0]  # noqa: E712
        col_loc = np.where(
            final_meta.index.isin(new_meta.index) == True)[0]  # noqa: E712

        row_slice = slice(np.min(row_loc), np.max(row_loc) + 1)
        col_slice = slice(np.min(col_loc), np.max(col_loc) + 1)

        return row_slice, col_slice

    @staticmethod
    def get_data(fpath, dset, time_index, meta, sites=None):
        """Retreive a data array from a chunked file.

        Parameters
        ----------
        fpath : str
            h5 file to get data from
        dset : str
            dataset to retrieve data from fpath.
        time_index : pd.Datetimeindex
            Time index of the final file.
        final_meta : pd.DataFrame
            Meta data of the final file.
        sites : None | np.ndarray
            Subset of site indices to collect. None collects all sites.

        Returns
        -------
        f_data : np.ndarray
            Data array from the fpath.
        row_slice : slice
            final_time_index[row_slice] = new_time_index
        col_slice : slice
            final_meta[col_slice] = new_meta
        """

        with Outputs(fpath, unscale=False, mode='r') as f:
            f_ti = f.time_index
            f_meta = f.meta

            if dset not in f.dsets:
                e = ('Trying to collect dataset "{}" but cannot find in '
                     'available: {}'.format(dset, f.dsets))
                logger.error(e)
                raise KeyError(e)

            if sites is None:
                f_data = f[dset][...]
            else:
                f_data = f[dset][:, sites]

        # use gid in chunked file in case results are chunked by site.
        if 'gid' in f_meta:
            f_meta.index = f_meta['gid']

        row_slice, col_slice = Collector.get_slices(time_index, meta,
                                                    f_ti, f_meta)
        return f_data, row_slice, col_slice

    @staticmethod
    def _get_collection_attrs(flist, collect_dir, f_out, dset, sites=None,
                              sort=True, sort_key=None):
        """Get important dataset attributes from a file list to be collected.

        Assumes the file list is chunked in time (row chunked).

        Parameters
        ----------
        flist : list
            List of chunked filenames in collect_dir to collect.
        collect_dir : str
            Directory of chunked files (flist).
        f_out : str
            File path of final output file.
        dset : str
            Dataset name to collect.
        sites : None | np.ndarray
            Subset of site indices to collect. None collects all sites.
        sort : bool
            flag to sort flist to determine meta data order.
        sort_key : None | fun
            Optional sort key to sort flist by (determines how meta is built
            if f_out does not exist).

        Returns
        -------
        time_index : pd.datetimeindex
            Concatenated datetime index from flist
        meta : pd.DataFrame
            Concatenated meta data from flist
        shape : tuple
            Output (collected) dataset shape
        dtype : str
            Dataset output (collected on disk) dataset data type.
        """

        if os.path.exists(f_out):
            with Outputs(f_out, mode='r') as f:
                time_index = f.time_index
                meta = f.meta
            shape = (len(time_index), len(meta))

        else:
            if sort:
                flist = sorted(flist, key=sort_key)
            logger.info('Collection output file does not exist, collecting '
                        'files in this order: {}'.format(flist))
            time_index = None
            meta = []
            for fn in flist:
                fp = os.path.join(collect_dir, fn)
                with Outputs(fp, mode='r') as f:
                    meta.append(f.meta)

                    if time_index is None:
                        time_index = f.time_index
                    else:
                        time_index = time_index.append(f.time_index)

            time_index = time_index.sort_values()
            time_index = time_index.drop_duplicates()
            meta = pd.concat(meta)
            meta = meta.drop_duplicates(subset=['latitude', 'longitude'])
            if sites is not None:
                meta = meta.iloc[sites, :]

            shape = (len(time_index), len(meta))

        fp0 = os.path.join(collect_dir, flist[0])
        with Outputs(fp0, mode='r') as fin:
            dtype = fin.get_dset_properties(dset)[1]

        return time_index, meta, shape, dtype

    @staticmethod
    def _init_collected_h5(f_out, time_index, meta):
        """Initialize the output h5 file to save collected data to.

        Parameters
        ----------
        f_out : str
            Output file path - must not yet exist.
        time_index : pd.datetimeindex
            Full datetime index of collected data.
        meta : pd.DataFrame
            Full meta dataframe collected data.
        """

        with Outputs(f_out, mode='w-') as f:
            logger.info('Initializing collection output file: {}'
                        .format(f_out))
            logger.info('Initializing collection output file with shape {} '
                        'and meta data:\n{}'
                        .format((len(time_index), len(meta)), meta))
            f['time_index'] = time_index
            f['meta'] = meta

    @staticmethod
    def _ensure_dset_in_output(f_out, dset, var_meta=None, data=None):
        """Ensure that dset is initialized in f_out and initialize if not.

        Parameters
        ----------
        f_out : str
            Pre-existing H5 file output path
        dset : str
            Dataset name
        var_meta : str | pd.DataFrame | None
            CSV file or dataframe containing meta data for all NSRDB variables.
            Defaults to the NSRDB var meta csv in git repo.
        data : np.ndarray | None
            Optional data to write to dataset if initializing.
        """

        with Outputs(f_out, mode='a') as f:
            if dset not in f.dsets:
                attrs, chunks, dtype = VarFactory.get_dset_attrs(
                    dset, var_meta=var_meta)
                logger.info('Initializing dataset "{}" with shape {} and '
                            'dtype {}'.format(dset, f.shape, dtype))
                f._create_dset(dset, f.shape, dtype, chunks=chunks,
                               attrs=attrs, data=data)

    @staticmethod
    def collect_flist(flist, collect_dir, f_out, dset, sites=None,
                      var_meta=None, parallel=True):
        """Collect a dataset from a file list with data pre-init.

        Collects data that can be chunked in both space and time.

        Parameters
        ----------
        flist : list
            List of chunked filenames in collect_dir to collect.
        collect_dir : str
            Directory of chunked files (flist).
        f_out : str
            File path of final output file.
        dset : str
            Dataset name to collect.
        sites : None | np.ndarray
            Subset of site indices to collect. None collects all sites.
        var_meta : str | pd.DataFrame | None
            CSV file or dataframe containing meta data for all NSRDB variables.
            Defaults to the NSRDB var meta csv in git repo.
        parallel : bool | int
            Flag to do chunk collection in parallel. Can be integer number of
            workers to use (number of parallel reads).
        """

        time_index, meta, shape, dtype = Collector._get_collection_attrs(
            flist, collect_dir, f_out, dset, sites=sites)

        data = np.zeros(shape, dtype=dtype)
        mem = psutil.virtual_memory()
        logger.info('Initializing output dataset "{0}" with shape {1} and '
                    'dtype {2}. Current memory usage is '
                    '{3:.3f} GB out of {4:.3f} GB total.'
                    .format(dset, shape, dtype,
                            mem.used / 1e9, mem.total / 1e9))

        if not parallel:
            for i, fname in enumerate(flist):
                logger.debug('Collecting data from file {} out of {}.'
                             .format(i + 1, len(flist)))
                fpath = os.path.join(collect_dir, fname)
                f_data, row_slice, col_slice = Collector.get_data(fpath, dset,
                                                                  time_index,
                                                                  meta,
                                                                  sites=sites)
                data[row_slice, col_slice] = f_data
        else:
            if parallel is True:
                max_workers = os.cpu_count()
            else:
                max_workers = parallel
            logger.info('Running parallel collection on {} workers.'
                        .format(max_workers))

            futures = []
            completed = 0
            with ProcessPoolExecutor(max_workers=max_workers) as exe:
                for fname in flist:
                    fpath = os.path.join(collect_dir, fname)
                    futures.append(exe.submit(Collector.get_data, fpath, dset,
                                              time_index, meta, sites=sites))
                for future in as_completed(futures):
                    completed += 1
                    mem = psutil.virtual_memory()
                    logger.debug('Collection futures completed: '
                                 '{0} out of {1}. '
                                 'Current memory usage is '
                                 '{2:.3f} GB out of {3:.3f} GB total.'
                                 .format(completed, len(futures),
                                         mem.used / 1e9, mem.total / 1e9))
                    f_data, row_slice, col_slice = future.result()
                    data[row_slice, col_slice] = f_data

        if not os.path.exists(f_out):
            Collector._init_collected_h5(f_out, time_index, meta)

        Collector._ensure_dset_in_output(f_out, dset, var_meta=var_meta)

        with Outputs(f_out, mode='a') as f:
            f[dset] = data

        logger.info('Finished writing dataset "{}"'.format(dset))

    @staticmethod
    def collect_flist_lowmem(flist, collect_dir, f_out, dset,
                             sort=False, sort_key=None, var_meta=None,
                             log_level=None, log_file=None, write_status=False,
                             job_name=None):
        """Collect a file list without data pre-init for low memory utilization

        Collects data that can be chunked in both space and time as long as
        f_out is pre-initialized.

        Parameters
        ----------
        flist : list | str
            List of chunked filenames in collect_dir to collect. Can also be a
            json.dumps(flist).
        collect_dir : str
            Directory of chunked files (flist).
        f_out : str
            File path of final output file. Must already be initialized with
            full time index and meta.
        dset : str
            Dataset name to collect.
        sort : bool
            flag to sort flist to determine meta data order.
        sort_key : None | fun
            Optional sort key to sort flist by (determines how meta is built
            if f_out does not exist).
        var_meta : str | pd.DataFrame | None
            CSV file or dataframe containing meta data for all NSRDB variables.
            Defaults to the NSRDB var meta csv in git repo.
        log_level : str | None
            Desired log level, None will not initialize logging.
        log_file : str | None
            Target log file. None logs to stdout.
        write_status : bool
            Flag to write status file once complete if running from pipeline.
        job_name : str
            Job name for status file if running from pipeline.
        """
        t0 = time.time()

        if log_level is not None:
            init_logger('nsrdb.file_handlers', log_file=log_file,
                        log_level=log_level)

        if not os.path.exists(f_out):
            time_index, meta, _, _ = Collector._get_collection_attrs(
                flist, collect_dir, f_out, dset, sort=sort, sort_key=sort_key)

            Collector._init_collected_h5(f_out, time_index, meta)

        Collector._ensure_dset_in_output(f_out, dset, var_meta=var_meta)

        if isinstance(flist, str):
            if '[' in flist and ']' in flist:
                flist = json.loads(flist)

        with Outputs(f_out, mode='a') as f:

            time_index = f.time_index
            meta = f.meta

            for fname in flist:
                logger.debug('Collecting file "{}".'.format(fname))
                fpath = os.path.join(collect_dir, fname)

                data, rows, cols = Collector.get_data(fpath, dset, time_index,
                                                      meta)

                f[dset, rows, cols] = data

        if write_status and job_name is not None:
            status = {'out_dir': os.path.dirname(f_out),
                      'fout': f_out,
                      'collect_dir': collect_dir,
                      'job_status': 'successful',
                      'runtime': (time.time() - t0) / 60,
                      'dset': dset}
            Status.make_job_file(os.path.dirname(f_out), 'collect-flist',
                                 job_name, status)

        logger.info('Finished file list collection.')

    @classmethod
    def collect_daily(cls, collect_dir, f_out, dsets, sites=None,
                      var_meta=None, parallel=True, log_level=None,
                      log_file=None, write_status=False, job_name=None):
        """Collect daily data model files from a dir to one output file.

        Assumes the file list is chunked in time (row chunked).

        Filename requirements:
         - Expects file names with leading "YYYYMMDD_".
         - Must have var in the file name.
         - Should end with ".h5"

        Parameters
        ----------
        collect_dir : str
            Directory of chunked files. Each file should be one variable for
            one day.
        f_out : str
            File path of final output file.
        dsets : list | str
            List of datasets / variable names to collect. Can also be a single
            dataset or json.dumps(dsets).
        sites : None | np.ndarray
            Subset of site indices to collect. None collects all sites.
        var_meta : str | pd.DataFrame | None
            CSV file or dataframe containing meta data for all NSRDB variables.
            Defaults to the NSRDB var meta csv in git repo. This is used if
            f_out has not yet been initialized.
        parallel : bool | int
            Flag to do chunk collection in parallel. Can be integer number of
            workers to use (number of parallel reads).
        log_level : str | None
            Desired log level, None will not initialize logging.
        log_file : str | None
            Target log file. None logs to stdout.
        write_status : bool
            Flag to write status file once complete if running from pipeline.
        job_name : str
            Job name for status file if running from pipeline.
        """
        t0 = time.time()

        if log_level is not None:
            init_logger('nsrdb.file_handlers', log_file=log_file,
                        log_level=log_level)

        if isinstance(dsets, str):
            if '[' in dsets and ']' in dsets:
                dsets = json.loads(dsets)
            else:
                dsets = [dsets]

        logger.info('Collecting data from {} to {}'.format(collect_dir, f_out))

        for dset in dsets:
            logger.debug('Collecting dataset "{}".'.format(dset))
            try:
                collector = cls(collect_dir, dset)
            except FileNotFoundError as e:
                if 'No "{}" files found'.format(dset) in str(e):
                    logger.info('Skipping dataset "{}", no files found in: {}'
                                .format(dset, collect_dir))
                else:
                    logger.exception(e)
                    raise e
            else:
                collector.collect_flist(collector.flist, collect_dir, f_out,
                                        dset, sites=sites, var_meta=var_meta,
                                        parallel=parallel)

        if write_status and job_name is not None:
            status = {'out_dir': os.path.dirname(f_out),
                      'fout': f_out,
                      'collect_dir': collect_dir,
                      'job_status': 'successful',
                      'runtime': (time.time() - t0) / 60,
                      'dsets': dsets}
            Status.make_job_file(os.path.dirname(f_out), 'collect-daily',
                                 job_name, status)

        logger.info('Finished daily file collection.')
