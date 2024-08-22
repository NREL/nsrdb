"""NSRDB chunked file collection tools."""

import datetime
import json
import logging
import os
from concurrent.futures import as_completed

import numpy as np
import pandas as pd
import psutil
from rex import NSRDB as NSRDBHandler
from rex import MultiFileNSRDB
from rex.utilities.execution import SpawnProcessPool
from rex.utilities.loggers import init_logger

from nsrdb.data_model import VarFactory
from nsrdb.file_handlers.outputs import Outputs
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

        self.collect_dir = collect_dir
        self.flist = self.get_flist(collect_dir, dset)

        if not any(self.flist):
            raise FileNotFoundError(
                'No "{}" files found in {}'.format(dset, collect_dir)
            )

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
            date_start = datetime.date(
                year=int(date_str_start[0:4]),
                month=int(date_str_start[4:6]),
                day=int(date_str_start[6:]),
            )
            date_end = datetime.date(
                year=int(date_str_end[0:4]),
                month=int(date_str_end[4:6]),
                day=int(date_str_end[6:]),
            )
        else:
            raise ValueError('Could not parse date: {}'.format(date_str_start))

        date_end += datetime.timedelta(days=1)
        ti = pd_date_range(
            start=date_start, end=date_end, freq='1D', closed='left'
        )

        missing = []
        for date in ti:
            date_str = '{}{}{}'.format(
                date.year, str(date.month).zfill(2), str(date.day).zfill(2)
            )
            if date_str not in date_str_list:
                missing.append(date_str)

        if missing:
            raise FileNotFoundError(
                'Missing the following date files for ' '"{}":\n{}'.format(
                    var, missing
                )
            )

        logger.info(
            'Good file count of {} for "{}" in year {} in dir: {}'.format(
                len(flist), var, date_start.year, d
            )
        )

    @staticmethod
    def filter_flist(flist, collect_dir, dset):
        """Filter file list so that only remaining files have given dset."""
        filt_list = []
        for fn in flist:
            fp = os.path.join(collect_dir, fn)
            with Outputs(fp, mode='r') as fobj:
                if dset in fobj.dsets:
                    filt_list.append(fn)

        logger.debug(f'Found files for "{dset}": {filt_list}')
        return filt_list

    @staticmethod
    def get_flist(d, dset):
        """Get a date-sorted .h5 file list for a given var.

        Filename requirements:
         - Expects file names with leading "YYYYMMDD_".
         - Must have var in the file name.
         - Should end with ".h5"

        Parameters
        ----------
        d : str
            Directory to get file list from.
        dset : str
            Variable name that is searched for in files in d.

        Returns
        -------
        flist : list
            List of .h5 files in directory d that contain the var string.
            Sorted by integer before the first underscore in the filename.
        """

        temp = os.listdir(d)
        temp = [f for f in temp if f.endswith('.h5') and dset in f]
        flist = Collector.filter_flist(temp, collect_dir=d, dset=dset)
        return sorted(flist, key=lambda x: int(x.split('_')[0]))

    @staticmethod
    def get_slices(final_time_index, final_meta, new_time_index, new_meta):
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
            msg = (
                'Could not find row locations in file collection. '
                'New time index: {} final time index: {}'.format(
                    new_time_index, final_time_index
                )
            )
            logger.error(msg)
            raise RuntimeError(msg)

        if not len(col_loc) > 0:
            msg = (
                'Could not find col locations in file collection. '
                'New gid index: {} final gid index: {}'.format(
                    new_index, final_index
                )
            )
            logger.error(msg)
            raise RuntimeError(msg)

        row_slice = slice(np.min(row_loc), np.max(row_loc) + 1)
        col_slice = slice(np.min(col_loc), np.max(col_loc) + 1)

        return row_slice, col_slice

    @staticmethod
    def get_data(
        fpath, dset, time_index, meta, scale_factor, dtype, sites=None
    ):
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
                e = (
                    'Trying to collect dataset "{}" but cannot find in '
                    'available: {}'.format(dset, f.dsets)
                )
                logger.error(e)
                raise KeyError(e)

            f_data = f[dset][...] if sites is None else f[dset][:, sites]

        # use gid in chunked file in case results are chunked by site.
        if 'gid' in f_meta:
            f_meta.index = f_meta['gid']

        row_slice, col_slice = Collector.get_slices(
            time_index, meta, f_ti, f_meta
        )

        if scale_factor != source_scale_factor:
            f_data = f_data.astype(np.float32)
            f_data *= scale_factor / source_scale_factor

        if np.issubdtype(dtype, np.integer):
            f_data = np.round(f_data)

        f_data = f_data.astype(dtype)

        return f_data, row_slice, col_slice

    @staticmethod
    def _special_attrs(dset, dset_attrs):
        """Enforce any special dataset attributes.

        Parameters
        ----------
        dset : str
            Name of dataset
        dset_attrs : dict
            Attribute key-value pair dictionary for dset.

        Returns
        -------
        dset_attrs : dict
            Attributes for dset with any special formatting.
        """

        if 'fill_flag' in dset:
            dset_attrs['units'] = 'percent of filled timesteps'

        if (
            'scale_factor' in dset_attrs
            and 'psm_scale_factor' not in dset_attrs
        ):
            dset_attrs['psm_scale_factor'] = dset_attrs['scale_factor']

        if 'units' in dset_attrs and 'psm_units' not in dset_attrs:
            dset_attrs['psm_units'] = dset_attrs['units']

        return dset_attrs

    @staticmethod
    def _get_collection_attrs(
        flist, collect_dir, sites=None, sort=True, sort_key=None
    ):
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

        return time_index, meta, shape

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
            logger.info(
                'Initializing collection output file: {}'.format(f_out)
            )
            logger.info(
                'Initializing collection output file with shape {} '
                'and meta data:\n{}'.format((len(time_index), len(meta)), meta)
            )
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
                    dset, var_meta=var_meta
                )
                logger.info(
                    'Initializing dataset "{}" with shape {} and '
                    'dtype {}'.format(dset, f.shape, dtype)
                )
                f._create_dset(
                    dset, f.shape, dtype, chunks=chunks, attrs=attrs, data=data
                )

    @classmethod
    def collect_flist(
        cls,
        flist,
        collect_dir,
        f_out,
        dset,
        sites=None,
        sort=False,
        sort_key=None,
        var_meta=None,
        max_workers=None,
    ):
        """Collect a dataset from a file list with data pre-init.

        Note
        ----
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

        flist = cls.filter_flist(
            flist=flist, collect_dir=collect_dir, dset=dset
        )
        time_index, meta, shape = Collector._get_collection_attrs(
            flist, collect_dir, sites=sites, sort=sort, sort_key=sort_key
        )

        attrs, _, final_dtype = VarFactory.get_dset_attrs(
            dset, var_meta=var_meta
        )
        scale_factor = attrs.get('scale_factor', 1)

        logger.debug(
            'Collecting file list of shape {}: {}'.format(shape, flist)
        )

        data = np.zeros(shape, dtype=final_dtype)
        mem = psutil.virtual_memory()
        logger.debug(
            'Initializing output dataset "{}" in-memory with shape '
            '{} and dtype {}. Current memory usage is '
            '{:.3f} GB out of {:.3f} GB total.'.format(
                dset, shape, final_dtype, mem.used / 1e9, mem.total / 1e9
            )
        )

        if max_workers == 1:
            for i, fname in enumerate(flist):
                logger.debug(
                    'Collecting data from file {} out of {}.'.format(
                        i + 1, len(flist)
                    )
                )
                fpath = os.path.join(collect_dir, fname)
                f_data, row_slice, col_slice = Collector.get_data(
                    fpath,
                    dset,
                    time_index,
                    meta,
                    scale_factor,
                    final_dtype,
                    sites=sites,
                )
                data[row_slice, col_slice] = f_data
        else:
            logger.info(
                'Running parallel collection on {} workers.'.format(
                    max_workers
                )
            )

            futures = []
            completed = 0
            loggers = ['nsrdb']
            with SpawnProcessPool(
                loggers=loggers, max_workers=max_workers
            ) as exe:
                for fname in flist:
                    fpath = os.path.join(collect_dir, fname)
                    futures.append(
                        exe.submit(
                            Collector.get_data,
                            fpath,
                            dset,
                            time_index,
                            meta,
                            scale_factor,
                            final_dtype,
                            sites=sites,
                        )
                    )
                for future in as_completed(futures):
                    completed += 1
                    mem = psutil.virtual_memory()
                    logger.info(
                        'Collection futures completed: '
                        '{} out of {}. '
                        'Current memory usage is '
                        '{:.3f} GB out of {:.3f} GB total.'.format(
                            completed,
                            len(futures),
                            mem.used / 1e9,
                            mem.total / 1e9,
                        )
                    )
                    f_data, row_slice, col_slice = future.result()
                    data[row_slice, col_slice] = f_data

        if not os.path.exists(f_out):
            Collector._init_collected_h5(f_out, time_index, meta)
            x_write_slice, y_write_slice = slice(None), slice(None)
        else:
            with Outputs(f_out, 'r') as f:
                target_meta = f.meta
                target_ti = f.time_index
            y_write_slice, x_write_slice = Collector.get_slices(
                target_ti, target_meta, time_index, meta
            )

        Collector._ensure_dset_in_output(f_out, dset, var_meta=var_meta)
        with Outputs(f_out, mode='a') as f:
            f[dset, y_write_slice, x_write_slice] = data

        logger.debug(
            'Finished writing "{}" for row {} and col {} to: {}'.format(
                dset, y_write_slice, x_write_slice, os.path.basename(f_out)
            )
        )

    @staticmethod
    def collect_flist_lowmem(
        flist,
        collect_dir,
        f_out,
        dset,
        sort=False,
        sort_key=None,
        var_meta=None,
        log_file='collect_flist_lowmem.log',
        log_level='DEBUG',
    ):
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
        """
        init_logger(
            'nsrdb.file_handlers', log_file=log_file, log_level=log_level
        )

        if not os.path.exists(f_out):
            time_index, meta, _ = Collector._get_collection_attrs(
                flist, collect_dir, sort=sort, sort_key=sort_key
            )

            Collector._init_collected_h5(f_out, time_index, meta)

        Collector._ensure_dset_in_output(f_out, dset, var_meta=var_meta)

        if isinstance(flist, str) and '[' in flist and ']' in flist:
            flist = json.loads(flist)

        with Outputs(f_out, mode='a') as f:
            time_index = f.time_index
            meta = f.meta
            dtype = f.get_dset_properties(dset)[1]
            scale_factor = f.get_scale_factor(dset)

            for fname in flist:
                logger.debug('Collecting file "{}".'.format(fname))
                fpath = os.path.join(collect_dir, fname)

                data, rows, cols = Collector.get_data(
                    fpath, dset, time_index, meta, scale_factor, dtype
                )
                f[dset, rows, cols] = data

        logger.info('Finished file list collection.')

    @classmethod
    def collect_daily(
        cls,
        collect_dir,
        fn_out,
        dsets,
        sites=None,
        n_writes=1,
        var_meta=None,
        max_workers=None,
        log_file='collect_daily.log',
        log_level='DEBUG',
    ):
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
        fn_out : str
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
            Number of workers to use in parallel. 1 runs serial, None will use
            all available workers.
        log_file : str | None
            Target log file. None logs to stdout.
        log_level : str | None
            Desired log level, None will not initialize logging.
        """
        init_logger(
            'nsrdb.file_handlers', log_file=log_file, log_level=log_level
        )

        if isinstance(dsets, str):
            dsets = (
                json.loads(dsets) if '[' in dsets and ']' in dsets else [dsets]
            )

        logger.info(
            'Collecting data from {} to {}'.format(collect_dir, fn_out)
        )

        for i, dset in enumerate(dsets):
            logger.debug('Collecting dataset "{}".'.format(dset))
            try:
                collector = cls(collect_dir, dset)
            except FileNotFoundError as e:
                if 'No "{}" files found'.format(dset) in str(e):
                    logger.info(
                        'Skipping dataset "{}", no files found in: {}'.format(
                            dset, collect_dir
                        )
                    )
                else:
                    logger.exception(e)
                    raise e
            else:
                if n_writes > len(collector.flist):
                    e = (
                        'Cannot split file list of length {} into '
                        '{} write chunks!'.format(
                            len(collector.flist), n_writes
                        )
                    )
                    logger.error(e)
                    raise ValueError(e)

                if not os.path.exists(fn_out):
                    time_index, meta, _ = collector._get_collection_attrs(
                        collector.flist, collect_dir, sites=sites
                    )
                    collector._init_collected_h5(fn_out, time_index, meta)

                flist_chunks = np.array_split(
                    np.array(collector.flist), n_writes
                )
                flist_chunks = [fl.tolist() for fl in flist_chunks]
                for j, flist in enumerate(flist_chunks):
                    logger.info(
                        'Collecting file list chunk {} out of {} '
                        'for "{}" (dataset {} out of {}).'.format(
                            j + 1, len(flist_chunks), dset, i + 1, len(dsets)
                        )
                    )
                    collector.collect_flist(
                        flist,
                        collect_dir,
                        fn_out,
                        dset,
                        sites=sites,
                        var_meta=var_meta,
                        max_workers=max_workers,
                    )

        logger.info('Finished daily file collection.')

    @staticmethod
    def get_dset_attrs(
        h5dir, ignore_dsets=('coordinates', 'time_index', 'meta')
    ):
        """Get output file dataset attributes for a set of datasets.

        Parameters
        ----------
        h5dir : str
            Path to directory containing multiple h5 files with all available
            dsets. Can also be a single h5 filepath.
        ignore_dsets : tuple | list
            List of datasets to ignore (will not be aggregated).

        Returns
        -------
        dsets : list
            List of datasets.
        attrs : dict
            Dictionary of dataset attributes keyed by dset name.
        chunks : dict
            Dictionary of chunk tuples keyed by dset name.
        dtypes : dict
            dictionary of numpy datatypes keyed by dset name.
        ti : pd.Datetimeindex
            Time index of source files in h5dir.
        """

        dsets = []
        attrs = {}
        chunks = {}
        dtypes = {}

        if h5dir.endswith('.h5') and os.path.isfile(h5dir):
            h5_files = [h5dir]
        elif h5dir.endswith('.h5') and '*' in h5dir:
            with MultiFileNSRDB(h5dir) as res:
                h5_files = res._h5_files
        else:
            h5_files = [fn for fn in os.listdir(h5dir) if fn.endswith('.h5')]

        logger.info(
            'Getting dataset attributes from the following files: {}'.format(
                h5_files
            )
        )

        for fn in h5_files:
            with NSRDBHandler(os.path.join(h5dir, fn)) as out:
                ti = out.time_index
                for d in out.dsets:
                    if d not in ignore_dsets and d not in attrs:
                        attrs[d] = Collector._special_attrs(
                            d, out.get_attrs(dset=d)
                        )

                        try:
                            x = out.get_dset_properties(d)
                        except Exception as e:
                            m = (
                                'Could not get dataset "{}" properties from '
                                'file: {}'.format(d, os.path.join(h5dir, fn))
                            )
                            logger.error(m)
                            logger.exception(m)
                            raise e
                        else:
                            _, dtypes[d], chunks[d] = x

        dsets = list(attrs.keys())
        logger.info('Found the following datasets: {}'.format(dsets))

        return dsets, attrs, chunks, dtypes, ti

    @classmethod
    def collect_dir(
        cls,
        meta_final,
        collect_dir,
        collect_tag,
        fout,
        dsets=None,
        max_workers=None,
        log_file='collect_dir.log',
        log_level='DEBUG',
    ):
        """Perform final collection of dsets for given collect_dir.

        Parameters
        ----------
        meta_final : str | pd.DataFrame
            Final meta data with index = gid.
        collect_dir : str
            Directory path containing chunked h5 files to collect.
        collect_tag : str
            String to be found in files that are being collected
        fout : str
            File path to the output collected file (will be initialized by
            this method).
        dsets : list | tuple
            Select datasets to collect (None will default to all dsets)
        max_workers : int | None
            Number of workers to use in parallel. 1 runs serial, None will use
            all available workers.
        log_file : str | None
            Target log file. None logs to stdout.
        log_level : str | None
            Desired log level, None will not initialize logging.
        """

        init_logger(
            'nsrdb.file_handlers', log_file=log_file, log_level=log_level
        )

        if isinstance(meta_final, str):
            meta_final = pd.read_csv(meta_final, index_col=0)

        fns = os.listdir(collect_dir)
        flist = [
            fn
            for fn in fns
            if fn.endswith('.h5')
            and collect_tag in fn
            and os.path.join(collect_dir, fn) != fout
        ]
        flist = sorted(
            flist,
            key=lambda x: int(x.replace('.h5', '').split('_')[-1])
            if x.replace('.h5', '').split('_')[-1].isdigit()
            else x,
        )

        logger.info(f'Collecting chunks from {len(flist)} files to: {fout}')

        dsets_all, attrs, chunks, dtypes, ti = cls.get_dset_attrs(collect_dir)
        dsets = dsets_all if dsets is None else dsets
        Outputs.init_h5(fout, dsets, attrs, chunks, dtypes, ti, meta_final)

        for dset in dsets:
            cls.collect_flist(
                flist,
                collect_dir,
                fout,
                dset,
                max_workers=max_workers,
            )
