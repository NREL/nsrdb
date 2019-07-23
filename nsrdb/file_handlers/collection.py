# -*- coding: utf-8 -*-
"""NSRDB chunked file collection tools.
"""
import datetime
import numpy as np
import pandas as pd
import os
import logging
import psutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from nsrdb.file_handlers.outputs import Outputs


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
        date_str = date_str_list[0]

        if len(date_str) == 8:
            date = datetime.date(year=int(date_str[0:4]),
                                 month=int(date_str[4:6]),
                                 day=int(date_str[6:]))
        else:
            raise ValueError('Could not parse date: {}'.format(date))

        ti = pd.date_range('1-1-{y}'.format(y=date.year),
                           '1-1-{y}'.format(y=date.year + 1),
                           freq='1D')[:-1]

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
                    .format(len(flist), var, date.year, d))

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

        flist = os.listdir(d)
        flist = [f for f in flist if '.h5' in f and var in f]
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
    def collect_flist(flist, collect_dir, f_out, dset, sites=None,
                      parallel=True):
        """Collect a dataset from a file list with data pre-init.

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
        parallel : bool | int
            Flag to do chunk collection in parallel. Can be integer number of
            workers to use (number of parallel reads).
        """

        with Outputs(f_out, mode='r') as f:
            time_index = f.time_index
            meta = f.meta
            shape, dtype, _ = f.get_dset_properties(dset)
            if sites is not None:
                shape = (shape[0], len(sites))

        data = np.zeros(shape, dtype=dtype)
        mem = psutil.virtual_memory()
        logger.info('Initializing output dataset "{0}" with shape {1} and '
                    'dtype {2}. Current memory usage is '
                    '{3:.3f} GB out of {4:.3f} GB total.'
                    .format(dset, shape, dtype,
                            mem.used / 1e9, mem.total / 1e9))

        if not parallel:
            for fname in flist:
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

        with Outputs(f_out, mode='a') as f:
            f[dset] = data

        logger.info('Finished writing dataset "{}"'.format(dset))

    @staticmethod
    def collect_flist_lowmem(flist, collect_dir, f_out, dset):
        """Collect a file list without data pre-init for low memory utilization

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
        """

        with Outputs(f_out, mode='a') as f:

            time_index = f.time_index
            meta = f.meta

            for fname in flist:
                logger.debug('Collecting file "{}".'.format(fname))
                fpath = os.path.join(collect_dir, fname)

                data, rows, cols = Collector.get_data(fpath, dset, time_index,
                                                      meta)

                f[dset, rows, cols] = data

    @classmethod
    def collect(cls, collect_dir, f_out, dsets, sites=None, parallel=True):
        """Collect files from a dir to one output file.

        Parameters
        ----------
        collect_dir : str
            Directory of chunked files. Each file should be one variable for
            one day.
        f_out : str
            File path of final output file.
        dsets : list
            List of datasets / variable names to collect.
        sites : None | np.ndarray
            Subset of site indices to collect. None collects all sites.
        parallel : bool | int
            Flag to do chunk collection in parallel. Can be integer number of
            workers to use (number of parallel reads).
        """

        logger.info('Collecting data from {} to {}'.format(collect_dir, f_out))

        for dset in dsets:
            logger.debug('Collecting dataset "{}".'.format(dset))
            try:
                collector = cls(collect_dir, dset)
            except FileNotFoundError as e:
                if 'No "{}" files found'.format(dset) in e:
                    logger.info('Skipping dataset "{}", no files found in: {}'
                                .format(dset, collect_dir))
                else:
                    logger.exception(e)
                    raise e
            else:
                collector.collect_flist(collector.flist, collect_dir, f_out,
                                        dset, sites=sites, parallel=parallel)
