# -*- coding: utf-8 -*-
"""NSRDB chunked file collection tools.
"""
from concurrent.futures import as_completed
import datetime
import json
import logging
import numpy as np
import os
import pandas as pd
import psutil
import time

from rex.utilities.execution import SpawnProcessPool
from rex.utilities.loggers import init_logger

from nsrdb.data_model import VarFactory
from nsrdb.file_handlers.outputs import Outputs
from nsrdb.pipeline import Status
from nsrdb.utilities.file_utils import pd_date_range

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
        ti = pd_date_range(start=date_start, end=date_end,
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
        temp = [f for f in temp if f.endswith('.h5') and var in f]

        for fn in temp:
            fp = os.path.join(d, fn)
            with Outputs(fp, mode='r') as fobj:
                if var in fobj.dsets:
                    flist.append(fn)

        flist = sorted(flist, key=lambda x: int(x.split('_')[0]))
        logger.debug('Found files for "{}": {}'.format(var, flist))

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

        final_index = final_meta.index
        new_index = new_meta.index
        if 'gid' in final_meta:
            final_index = final_meta['gid']
        if 'gid' in new_meta:
            new_index = new_meta['gid']

        row_loc = np.where(final_time_index.isin(new_time_index))[0]
        col_loc = np.where(final_index.isin(new_index))[0]

        if not len(row_loc) > 0:
            msg = ('Could not find row locations in file collection. '
                   'New time index: {} final time index: {}'
                   .format(new_time_index, final_time_index))
            logger.error(msg)
            raise RuntimeError(msg)

        if not len(col_loc) > 0:
            msg = ('Could not find col locations in file collection. '
                   'New gid index: {} final gid index: {}'
                   .format(new_index, final_index))
            logger.error(msg)
            raise RuntimeError(msg)

        row_slice = slice(np.min(row_loc), np.max(row_loc) + 1)
        col_slice = slice(np.min(col_loc), np.max(col_loc) + 1)

        return row_slice, col_slice

    @staticmethod
    def get_data(fpath, dset, time_index, meta, scale_factor, dtype,
                 sites=None):
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
        scale_factor : int | float
            Final destination scale factor after collection. If the data
            retrieval from the files to be collected has a different scale
            factor, the collected data will be rescaled and returned as
            float32.
        dtype : np.dtype
            Final dtype to return data as
        sites : None | np.ndarray
            Subset of site indices to collect. None collects all sites.

        Returns
        -------
        f_data : np.ndarray
            Data array from the fpath cast as input dtype.
        row_slice : slice
            final_time_index[row_slice] = new_time_index
        col_slice : slice
            final_meta[col_slice] = new_meta
        """

        with Outputs(fpath, unscale=False, mode='r') as f:
            f_ti = f.time_index
            f_meta = f.meta
            source_scale_factor = f.attrs[dset].get('scale_factor', 1)

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

        if scale_factor != source_scale_factor:
            f_data = f_data.astype(np.float32)
            f_data *= (scale_factor / source_scale_factor)

        if np.issubdtype(dtype, np.integer):
            f_data = np.round(f_data)

        f_data = f_data.astype(dtype)

        return f_data, row_slice, col_slice

    @staticmethod
    def _get_collection_attrs(flist, collect_dir, dset, sites=None,
                              sort=True, sort_key=None):
        """Get important dataset attributes from a file list to be collected.

        Assumes the file list is chunked in time (row chunked).

        Parameters
        ----------
        flist : list
            List of chunked filenames in collect_dir to collect.
        collect_dir : str
            Directory of chunked files (flist).
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
            Concatenated full size datetime index from the flist that is
            being collected
        meta : pd.DataFrame
            Concatenated full size meta data from the flist that is being
            collected
        shape : tuple
            Output (collected) dataset shape
        dtype : str
            Dataset output (collected on disk) dataset data type.
        """

        if sort:
            flist = sorted(flist, key=sort_key)

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

        if 'gid' in meta:
            meta = meta.drop_duplicates(subset=['gid'])
        elif 'latitude' in meta and 'longitude' in meta:
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
                      sort=False, sort_key=None, var_meta=None,
                      max_workers=None):
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
        sort : bool
            flag to sort flist to determine meta data order.
        sort_key : None | fun
            Optional sort key to sort flist by (determines how meta is built
            if f_out does not exist).
        var_meta : str | pd.DataFrame | None
            CSV file or dataframe containing meta data for all NSRDB variables.
            Defaults to the NSRDB var meta csv in git repo.
        max_workers : int | None
            Number of workers to use in parallel. 1 runs serial,
            None uses all available.
        """

        time_index, meta, shape, _ = \
            Collector._get_collection_attrs(flist, collect_dir, dset,
                                            sites=sites, sort=sort,
                                            sort_key=sort_key)

        attrs, _, final_dtype = VarFactory.get_dset_attrs(
            dset, var_meta=var_meta)
        scale_factor = attrs.get('scale_factor', 1)

        logger.debug('Collecting file list of shape {}: {}'
                     .format(shape, flist))

        data = np.zeros(shape, dtype=final_dtype)
        mem = psutil.virtual_memory()
        logger.debug('Initializing output dataset "{0}" in-memory with shape '
                     '{1} and dtype {2}. Current memory usage is '
                     '{3:.3f} GB out of {4:.3f} GB total.'
                     .format(dset, shape, final_dtype,
                             mem.used / 1e9, mem.total / 1e9))

        if max_workers == 1:
            for i, fname in enumerate(flist):
                logger.debug('Collecting data from file {} out of {}.'
                             .format(i + 1, len(flist)))
                fpath = os.path.join(collect_dir, fname)
                f_data, row_slice, col_slice = Collector.get_data(fpath, dset,
                                                                  time_index,
                                                                  meta,
                                                                  scale_factor,
                                                                  final_dtype,
                                                                  sites=sites)
                data[row_slice, col_slice] = f_data
        else:
            logger.info('Running parallel collection on {} workers.'
                        .format(max_workers))

            futures = []
            completed = 0
            loggers = ['nsrdb']
            with SpawnProcessPool(loggers=loggers,
                                  max_workers=max_workers) as exe:
                for fname in flist:
                    fpath = os.path.join(collect_dir, fname)
                    futures.append(exe.submit(Collector.get_data, fpath, dset,
                                              time_index, meta, scale_factor,
                                              final_dtype, sites=sites))
                for future in as_completed(futures):
                    completed += 1
                    mem = psutil.virtual_memory()
                    logger.info('Collection futures completed: '
                                '{0} out of {1}. '
                                'Current memory usage is '
                                '{2:.3f} GB out of {3:.3f} GB total.'
                                .format(completed, len(futures),
                                        mem.used / 1e9, mem.total / 1e9))
                    f_data, row_slice, col_slice = future.result()
                    data[row_slice, col_slice] = f_data

        if not os.path.exists(f_out):
            Collector._init_collected_h5(f_out, time_index, meta)
            x_write_slice, y_write_slice = slice(None), slice(None)
        else:
            with Outputs(f_out, 'r') as f:
                target_meta = f.meta
                target_ti = f.time_index
            y_write_slice, x_write_slice = Collector.get_slices(target_ti,
                                                                target_meta,
                                                                time_index,
                                                                meta)

        Collector._ensure_dset_in_output(f_out, dset, var_meta=var_meta)
        with Outputs(f_out, mode='a') as f:
            f[dset, y_write_slice, x_write_slice] = data

        logger.debug('Finished writing "{}" for row {} and col {} to: {}'
                     .format(dset, y_write_slice, x_write_slice,
                             os.path.basename(f_out)))

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
                flist, collect_dir, dset, sort=sort, sort_key=sort_key)

            Collector._init_collected_h5(f_out, time_index, meta)

        Collector._ensure_dset_in_output(f_out, dset, var_meta=var_meta)

        if isinstance(flist, str):
            if '[' in flist and ']' in flist:
                flist = json.loads(flist)

        with Outputs(f_out, mode='a') as f:
            time_index = f.time_index
            meta = f.meta
            dtype = f.get_dset_properties(dset)[1]
            scale_factor = f.get_scale_factor(dset)

            for fname in flist:
                logger.debug('Collecting file "{}".'.format(fname))
                fpath = os.path.join(collect_dir, fname)

                data, rows, cols = Collector.get_data(fpath, dset, time_index,
                                                      meta, scale_factor,
                                                      dtype)
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
                      n_writes=1, var_meta=None, max_workers=None,
                      log_level=None, log_file=None, write_status=False,
                      job_name=None):
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
        n_writes : None | int
            Number of file list divisions to write per dataset. For example,
            if ghi and dni are being collected and n_writes is set to 2,
            half of the source ghi files will be collected at once and then
            written, then the second half of ghi files, then dni.
        var_meta : str | pd.DataFrame | None
            CSV file or dataframe containing meta data for all NSRDB variables.
            Defaults to the NSRDB var meta csv in git repo. This is used if
            f_out has not yet been initialized.
        max_workers : int | None
            Number of workers to use in parallel. 1 runs serial,
            None will use all available workers.
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

        for i, dset in enumerate(dsets):
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
                if n_writes > len(collector.flist):
                    e = ('Cannot split file list of length {} into '
                         '{} write chunks!'
                         .format(len(collector.flist), n_writes))
                    logger.error(e)
                    raise ValueError(e)

                if not os.path.exists(f_out):
                    time_index, meta, _, _ = \
                        collector._get_collection_attrs(
                            collector.flist, collect_dir, dset, sites=sites)
                    collector._init_collected_h5(f_out, time_index, meta)

                flist_chunks = np.array_split(np.array(collector.flist),
                                              n_writes)
                flist_chunks = [fl.tolist() for fl in flist_chunks]
                for j, flist in enumerate(flist_chunks):
                    logger.info('Collecting file list chunk {} out of {} '
                                'for "{}" (dataset {} out of {}).'
                                .format(j + 1, len(flist_chunks),
                                        dset, i + 1, len(dsets)))
                    collector.collect_flist(flist, collect_dir, f_out,
                                            dset, sites=sites,
                                            var_meta=var_meta,
                                            max_workers=max_workers)

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
