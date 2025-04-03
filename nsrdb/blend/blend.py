# -*- coding: utf-8 -*-
"""NSRDB east/west blending utilities."""

import logging
import os
from warnings import warn

import numpy as np
import pandas as pd
from rex import init_logger
from scipy.spatial import KDTree

from nsrdb.data_model import VarFactory
from nsrdb.file_handlers.outputs import Outputs

logger = logging.getLogger(__name__)


class Blender:
    """Class to blend east and west satellite extents"""

    def __init__(
        self,
        meta_out,
        out_fpath,
        east_fpath,
        west_fpath,
        map_col='gid_full',
        lon_seam=-105.0,
    ):
        """
        Parameters
        ----------
        meta_out : str | pd.DataFrame
            Final output blended meta data (filepath or extracted df).
        out_fpath : str
            Filepath to save output file to.
        east_fpath : str
            NSRDB file for eastern extent.
        west_fpath : str
            NSRDB file for western extent.
        map_col : str, optional
            Column in the east and west meta data that map sites to the full
            meta_out gids.
        lon_seam : int, optional
            Vertical longitude seam at which data transitions from the western
            source to eastern, by default -105 (historical closest to nadir).
            5min conus data (2019 onward) is typically blended at -113.0
            because the conus west satellite extent doesnt go that far east.
        """

        logger.info(
            f'Blender running at longitude seam: {lon_seam}. Blender '
            f'initialized with west source file: {west_fpath}, east '
            f'source file: {east_fpath}, output file: {out_fpath}'
        )

        self._out_fpath = out_fpath
        self._east_fpath = east_fpath
        self._west_fpath = west_fpath
        self._map_col = map_col
        self._lon_seam = lon_seam
        self._meta_out_tree = None

        self._meta_out = self._parse_meta_file(meta_out)
        self._meta_east = self._parse_meta_file(east_fpath)
        self._meta_west = self._parse_meta_file(west_fpath)
        self._parse_blended_meta()

        logger.debug(
            f'Final output meta: \n{self._meta_out.head()}'
            f'\n{self._meta_out.tail()}. Source east meta: '
            f'\n{self._meta_east.head()}\n{self._meta_east.tail()} '
            f'Source west meta: \n{self._meta_west.head()}'
            f'\n{self._meta_west.tail()}'
        )

        self._time_index = self._parse_ti()
        self._dsets = self._parse_dsets()

        attrs, chunks, dtypes = VarFactory.get_dsets_attrs(self._dsets)
        Outputs.init_h5(
            self._out_fpath,
            self._dsets,
            attrs,
            chunks,
            dtypes,
            self._time_index,
            self._meta_out,
        )

    @staticmethod
    def _parse_meta_file(inp):
        """Parse a meta data filepath or dataframe and return extracted meta df

        Parameters
        ----------
        inp : str | pd.DataFrame
            Meta data filepath (csv or h5) or extracted df.

        Returns
        -------
        meta_out : pd.DataFrame
            Meta data extracted df.
        """

        if isinstance(inp, pd.DataFrame):
            meta_out = inp

        elif isinstance(inp, str):
            if inp.endswith('.csv'):
                meta_out = pd.read_csv(inp, index_col=0)
            elif inp.endswith('.h5'):
                with Outputs(inp, mode='r') as out:
                    meta_out = out.meta
            else:
                e = 'Meta filepath must be csv or h5 but received: {}'.format(
                    inp
                )
                logger.error(e)
                raise ValueError(e)

        elif not isinstance(inp, pd.DataFrame):
            e = 'Meta needs to be DataFrame or str but received: {}'.format(
                type(inp)
            )
            logger.error(e)
            raise ValueError(e)

        return meta_out

    @property
    def out_meta_tree(self):
        """Get KDTree for blended output meta"""
        if self._out_meta_tree is None:
            self._out_meta_tree = KDTree(
                self._out_meta[['latitude', 'longitude']].values
            )
        return self._out_meta_tree

    def check_map_col(self, meta):
        """Check if given meta has a column which maps to the full output meta.
        If not add this using queries to a KDTree."""
        if self._map_col not in meta:
            msg = (
                'gid mapping column not found in given meta. Will build '
                'and query a KDTree instead. This might take some time.'
            )
            logger.warning(msg)
            warn(msg)
            _, gids = self.out_meta_tree.query(
                meta[['latitude', 'longitude']].values
            )
            meta[self._map_col] = gids
        return meta

    def _parse_blended_meta(self):
        """Check meta dfs and reduce to active blended source extents."""

        west_mask = self._meta_west['longitude'] < self._lon_seam
        east_mask = self._meta_east['longitude'] >= self._lon_seam

        self._meta_west = self.check_map_col(self._meta_west[west_mask])
        self._meta_east = self.check_map_col(self._meta_east[east_mask])

        west_gid_full = self._meta_west[self._map_col].values.to_list()
        east_gid_full = self._meta_east[self._map_col].values.to_list()

        gid_full_all = list(set(west_gid_full + east_gid_full))

        self._check_sequential(
            self._meta_west.index.values, 'West source gids'
        )
        self._check_sequential(
            self._meta_east.index.values, 'East source gids'
        )
        self._check_sequential(
            self._meta_west[self._map_col].values, 'West destination gids'
        )
        self._check_sequential(
            self._meta_east[self._map_col].values, 'East destination gids'
        )

        if len(gid_full_all) != len(west_gid_full) + len(east_gid_full):
            e = (
                'Western full-extent gids and eastern full-extent gids have '
                'duplicates based on a seam of {}!'.format(self._lon_seam)
            )
            logger.error(e)
            raise RuntimeError(e)

        if len(gid_full_all) != len(self._meta_out):
            e = (
                'Combined western and eastern meta data have {} gids wheras '
                'the final output meta data has {} gids!'.format(
                    len(gid_full_all), len(self._meta_out)
                )
            )
            logger.error(e)
            raise RuntimeError(e)

    @staticmethod
    def _check_sequential(arr, name, warn_flag=False, raise_flag=True):
        """Check if an integer array is sequential. Warn or raise error.

        Parameters
        ----------
        arr : np.ndarray
            1D array of integers.
        name : str
            Name of array to print to warning/error messages
        warn_flag : bool
            Flag to raise a warning if not sequential.
        raise_flag : bool
            Flag to raise Runtimeerror if not sequential.

        Returns
        -------
        sequential : bool
            Whether arr is sequential or not.
        """
        sequential = True
        arr_seq = np.arange(arr.min(), arr.max() + 1)

        if not all(arr == arr_seq):
            sequential = False
            msg = '{} is not sequential!'.format(name)
            if warn_flag:
                logger.warning(msg)
                warn(msg)
            if raise_flag:
                logger.error(msg)
                raise RuntimeError(msg)

        return sequential

    def _parse_ti(self):
        """Parse the time index from the east/west file. Raise if not equal.

        Returns
        -------
        ti : pd.DateTimeIndex
            Pandas datetimeindex for the east and west input files.
        """

        with Outputs(self._east_fpath, mode='r') as out:
            ti_e = out.time_index
        with Outputs(self._west_fpath, mode='r') as out:
            ti_w = out.time_index

        if not all(ti_e == ti_w):
            e = 'Time index for east and west do not match!'
            logger.error(e)
            raise ValueError(e)

        return ti_e

    def _parse_dsets(self, ignore=('meta', 'time_index', 'coordinates')):
        """Parse the datasets from the east/west files.

        Parameters
        ----------
        ignore : list | tuple
            List of datasets to ignore in parsing.

        Returns
        -------
        dsets : list
            List of dataset names common to both files.
        """

        with Outputs(self._east_fpath, mode='r') as out:
            dsets_e = sorted([d for d in out.dsets if d not in ignore])
        with Outputs(self._west_fpath, mode='r') as out:
            dsets_w = sorted([d for d in out.dsets if d not in ignore])

        if dsets_e != dsets_w:
            w = 'Datasets from east and west files do not match!'
            logger.warning(w)
            warn(w)

        dsets = list(set(dsets_e).intersection(set(dsets_w)))
        logger.info(
            'Blending the following {} dsets: {}'.format(len(dsets), dsets)
        )
        return dsets

    def run_blend(self, source_fpath, source_meta, chunk_size=100000):
        """Run blending from one source file to the initialized output file.

        Parameters
        ----------
        source_fpath : str
            Source filepath (h5) to blend.
        source_meta : pd.DataFrame
            Source meta data to be blended - must be reduced to only data that
            is going to be written to final output file from source. Site gids
            must be sequential in source and destination.
        chunk_size : int
            Number of sites to read/write at a time.
        """

        n = np.ceil(len(source_meta) / chunk_size)
        source_indices = source_meta.index.values
        destination_indices = source_meta[self._map_col].values
        source_chunks = np.array_split(source_indices, n)
        destination_chunks = np.array_split(destination_indices, n)

        logger.info('Starting blend from source file: {}'.format(source_fpath))

        with (
            Outputs(source_fpath, mode='r', unscale=False) as source,
            Outputs(self._out_fpath, mode='a') as out,
        ):
            for i_d, dset in enumerate(self._dsets):
                logger.info(
                    'Starting blend of dataset "{}", {} of {}'.format(
                        dset, i_d + 1, len(self._dsets)
                    )
                )

                zipped = zip(source_chunks, destination_chunks)
                for i, (i_source, i_destination) in enumerate(zipped):
                    logger.debug(
                        '\t Blending gid chunk {} out of {}'.format(
                            i + 1, len(source_chunks)
                        )
                    )
                    self._check_sequential(
                        i_source,
                        'Source chunk {}'.format(i),
                        raise_flag=True,
                    )
                    self._check_sequential(
                        i_destination,
                        'Destination chunk {}'.format(i),
                        raise_flag=True,
                    )
                    s = slice(i_source.min(), i_source.max() + 1)
                    d = slice(i_destination.min(), i_destination.max() + 1)
                    out[dset, :, d] = source[dset, :, s]
        logger.info('Finished blend from source file: {}'.format(source_fpath))

    @classmethod
    def blend_file(
        cls,
        meta_out,
        out_fpath,
        east_fpath,
        west_fpath,
        map_col='gid_full',
        lon_seam=-105.0,
        chunk_size=100000,
    ):
        """Initialize and run the blender using explicit source and output
        filepaths. This initializes a "blended" output file with the meta given
        by ``meta_out``, and then imputes east and west data into this output
        file using the indices in ``map_col``. This might be more appropriately
        called "stitching".

        Parameters
        ----------
        meta_out : str | pd.DataFrame
            Final output blended meta data (filepath or extracted df).
        out_fpath : str
            Filepath to save output file to.
        east_fpath : str
            NSRDB file for eastern extent.
        west_fpath : str
            NSRDB file for western extent.
        map_col : str, optional
            Column in the east and west meta data that map sites to the full
            meta_out gids.
        lon_seam : int, optional
            Vertical longitude seam at which data transitions from the western
            source to eastern, by default -105 (historical closest to nadir).
            5min conus data (2019 onward) is typically blended at -113.0
            because the conus west satellite extent doesnt go that far east.
        chunk_size : int
            Number of sites to read/write at a time.
        """

        b = cls(
            meta_out,
            out_fpath,
            east_fpath,
            west_fpath,
            map_col=map_col,
            lon_seam=lon_seam,
        )

        b.run_blend(b._west_fpath, b._meta_west, chunk_size=chunk_size)
        b.run_blend(b._east_fpath, b._meta_east, chunk_size=chunk_size)

        logger.info('Finished blend. Output file is: {}'.format(b._out_fpath))

    @classmethod
    def blend_dir(
        cls,
        meta_out,
        out_dir,
        east_dir,
        west_dir,
        file_tag,
        out_fn=None,
        map_col='gid_full',
        lon_seam=-105.0,
        chunk_size=100000,
    ):
        """Initialize and run the blender on two source directories with a
        file tag to search for. This can only blend one file.

        Parameters
        ----------
        meta_out : str | pd.DataFrame
            Final output blended meta data (filepath or extracted df).
        out_dir : str
            Directory to save output file to.
        east_dir : str
            NSRDB output directory for eastern extent.
        west_dir : str
            NSRDB output directory for western extent.
        file_tag : str
            String to look for in files in east_dir and west_dir to find
            source files.
        out_fn : str
            Optional output filename. Will be inferred from the east file
            (without '_east') if not input.
        map_col : str, optional
            Column in the east and west meta data that map sites to the full
            meta_out gids.
        lon_seam : int, optional
            Vertical longitude seam at which data transitions from the western
            source to eastern, by default -105 (historical closest to nadir).
            5min conus data (2019 onward) is typically blended at -113.0
            because the conus west satellite extent doesnt go that far east.
        chunk_size : int
            Number of sites to read/write at a time.
        """

        fns_east = [
            fn
            for fn in os.listdir(east_dir)
            if fn.endswith('.h5') and file_tag in fn
        ]
        fns_west = [
            fn
            for fn in os.listdir(west_dir)
            if fn.endswith('.h5') and file_tag in fn
        ]

        if len(fns_east) > 1 or len(fns_west) > 1:
            e = (
                f'Found multiple files with tag "{file_tag}" in source dirs: '
                f'{east_dir} and {west_dir}'
            )
            logger.error(e)
            raise RuntimeError(e)

        east_fpath = os.path.join(east_dir, fns_east[0])
        west_fpath = os.path.join(west_dir, fns_west[0])
        if out_fn is None:
            out_fn = fns_east[0].replace('_east', '')
        out_fpath = os.path.join(out_dir, out_fn)

        cls.blend_file(
            meta_out,
            out_fpath,
            east_fpath,
            west_fpath,
            map_col=map_col,
            lon_seam=lon_seam,
            chunk_size=chunk_size,
        )

    @classmethod
    def run_full(
        cls,
        meta_file,
        out_dir,
        east_dir,
        west_dir,
        out_fn=None,
        east_fn=None,
        west_fn=None,
        file_tag=None,
        map_col='gid_full',
        lon_seam=-105.0,
        chunk_size=100000,
        log_file='blend.log',
        log_level='DEBUG',
    ):
        """NSRDB East-West Blend.

        Parameters
        ----------
            meta_file: str
                Filepath to final output blended meta data csv file.
            out_dir: str
                Directory to save blended output.
            east_dir: str
                Source east directory.
            west_dir : str
                Source west directory.
            out_fn: str
                Optional output filename
            east_fn: str
                Optional east filename (found in east_dir)
            west_fn: str
                Optional west filename (found in west_dir)
            file_tag : str
                File tag found in files in east and west source dirs.
            map_col : str
                Column in the east and west meta data that map sites to the
                full meta_out gids.
            lon_seam : float
                Vertical longitude seam at which data transitions from the
                western source to eastern, by default -105.0 (historical
                closest to nadir).  5min conus data (2019 onwards) is typically
                blended at -113.0 because the conus west satellite extent
                doesnt go that far east.
            chunk_size : int
                Number of sites to read/write at a time.
            log_file : str
                File to use for logging
            log_level : str
                Level to use for logging.
        """
        if (
            out_fn is not None
            and east_fn is not None
            and west_fn is not None
            and file_tag is not None
        ):
            logger.info(
                'Filenames and file tags all specified. Using filenames.'
            )
            file_tag = None

        init_logger('nsrdb.blend', log_level=log_level, log_file=log_file)

        if (
            out_fn is None
            and east_fn is None
            and west_fn is None
            and file_tag is None
        ):
            e = 'Filenames or file_tag must be specified for local blend job.'
            logger.error(e)
            raise RuntimeError(e)

        if out_fn is not None and east_fn is not None and west_fn is not None:
            out_fpath = os.path.join(out_dir, out_fn)
            east_fpath = os.path.join(east_dir, east_fn)
            west_fpath = os.path.join(west_dir, west_fn)
            logger.info(f'Running blend_file with file_tag {file_tag}.')
            cls.blend_file(
                meta_file,
                out_fpath,
                east_fpath,
                west_fpath,
                map_col=map_col,
                lon_seam=lon_seam,
                chunk_size=chunk_size,
            )
        else:
            logger.info(f'Running blend_dir with file_tag {file_tag}.')
            cls.blend_dir(
                meta_file,
                out_dir,
                east_dir,
                west_dir,
                file_tag,
                out_fn=out_fn,
                map_col=map_col,
                lon_seam=lon_seam,
                chunk_size=chunk_size,
            )
