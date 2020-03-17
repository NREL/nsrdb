# -*- coding: utf-8 -*-
"""NSRDB east/west blending utilities.
"""
import logging
import numpy as np
import pandas as pd
from warnings import warn

from nsrdb.file_handlers.outputs import Outputs
from nsrdb.main import NSRDB


logger = logging.getLogger(__name__)


class Blender:
    """Class to blend east and west satellite extents"""

    def __init__(self, meta_out, fpath_out, fpath_east, fpath_west,
                 map_col='gid_full', lon_seam=-105.0):
        """
        Parameters
        ----------
        meta_out : str | pd.DataFrame
            Final output blended meta data (filepath or extracted df).
        fpath_out : str
            Filepath to save output file to.
        fpath_east : str
            NSRDB file for eastern extent.
        fpath_west : str
            NSRDB file for western extent.
        map_col : str, optional
            Column in the east and west meta data that map sites to the full
            meta_out gids.
        lon_seam : int, optional
            Vertical longitude seam at which data transitions from the western
            source to eastern, by default -105 (historical closest to nadir).
        """

        self._fpath_out = fpath_out
        self._fpath_east = fpath_east
        self._fpath_west = fpath_west
        self._map_col = map_col
        self._lon_seam = lon_seam

        self._meta_out = self._parse_meta_file(meta_out)
        self._meta_west = self._parse_meta_file(fpath_east)
        self._meta_east = self._parse_meta_file(fpath_west)

        self._parse_blended_meta()

        self._time_index = self._parse_ti()
        self._dsets = self._parse_dsets()

        logger.info('Blender initialized with west source: {}'
                    .format(fpath_west))
        logger.info('Blender initialized with east source: {}'
                    .format(fpath_east))
        logger.info('Blender initialized with output: {}'
                    .format(fpath_out))

        NSRDB._init_output_h5(fpath_out, self._dsets, self._time_index,
                              self._meta_out)

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
        if isinstance(inp, str):
            if inp.endswith('.csv'):
                meta_out = pd.read_csv(inp, index_col=0)
            elif inp.endswith('.h5'):
                with Outputs(inp, mode='r') as out:
                    meta_out = out.meta
            else:
                e = ('Meta filepath must be csv or h5 but received: {}'
                     .format(inp))
                logger.error(e)
                raise ValueError(e)

        elif not isinstance(inp, pd.DataFrame):
            e = ('Meta needs to be DataFrame or str but received: {}'
                 .format(type(inp)))
            logger.error(e)
            raise ValueError(e)

        return meta_out

    def _parse_blended_meta(self):
        """Check meta dfs and reduce to active blended source extents."""

        emsg = 'Source meta data needs mapping col: "{}"'.format(self._map_col)
        assert self._map_col in self._meta_west, emsg
        assert self._map_col in self._meta_east, emsg

        west_mask = self._meta_west['longitude'] < self._lon_seam
        east_mask = self._meta_east['longitude'] >= self._lon_seam

        self._meta_west = self._meta_west[west_mask]
        self._meta_east = self._meta_east[east_mask]

        west_gid_full = self._meta_west[self._map_col].values.tolist()
        east_gid_full = self._meta_east[self._map_col].values.tolist()
        gid_full_all = list(set(west_gid_full + east_gid_full))

        self._check_sequential(self._meta_west.index.values,
                               'West source gids')
        self._check_sequential(self._meta_east.index.values,
                               'East source gids')
        self._check_sequential(self._meta_west[self._map_col].values,
                               'West destination gids')
        self._check_sequential(self._meta_east[self._map_col].values,
                               'East destination gids')

        if len(gid_full_all) != len(west_gid_full) + len(east_gid_full):
            e = ('Western full-extent gids and eastern full-extent gids have '
                 'duplicates based on a seam of {}!'.format(self._lon_seam))
            logger.error(e)
            raise RuntimeError(e)

        if len(gid_full_all) != len(self._meta_out):
            e = ('Combined western and eastern meta data have {} gids wheras '
                 'the final output meta data has {} gids!'
                 .format(len(gid_full_all), len(self._meta_out)))
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
        ------
        sequential : bool
            Whether arr is sequential or not.
        """
        sequential = True
        arr_seq = np.arange(arr.min(), arr.max() + 1)
        if not all(arr == arr_seq):
            sequential = False
            msg = ('{} is not sequential!'.format(name))
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

        with Outputs(self._fpath_east, mode='r') as out:
            ti_e = out.time_index
        with Outputs(self._fpath_west, mode='r') as out:
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

        with Outputs(self._fpath_east, mode='r') as out:
            dsets_e = sorted([d for d in out.dsets if d not in ignore])
        with Outputs(self._fpath_west, mode='r') as out:
            dsets_w = sorted([d for d in out.dsets if d not in ignore])

        if not all(dsets_e == dsets_w):
            w = 'Datasets from east and west files do not match!'
            logger.warning(w)
            warn(w)

        dsets = list(set(dsets_e).intersection(set(dsets_w)))
        logger.info('Blending the following {} dsets: {}'
                    .format(len(dsets), dsets))
        return dsets

    def blend(self, source_fpath, source_meta, chunk_size=1000):
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

        with Outputs(source_fpath, mode='r', unscale=False) as source:
            with Outputs(self._fpath_out, mode='a') as out:

                for dset in self._dsets:
                    logger.info('Blending {}'.format(dset))

                    zipped = zip(source_chunks, destination_chunks)
                    for i, (i_source, i_destination) in enumerate(zipped):
                        logger.info('Blending gid chunk {} out of {}'
                                    .format(i + 1, len(source_chunks)))

                        self._check_sequential(
                            i_source, 'Source chunk {}'.format(i),
                            raise_flag=True)

                        self._check_sequential(
                            i_destination, 'Destination chunk {}'.format(i),
                            raise_flag=True)

                        s = slice(i_source.min(), i_source.max() + 1)
                        d = slice(i_destination.min(), i_destination.max() + 1)

                        out[dset, :, d] = source[dset, :, s]

    @classmethod
    def run(cls, meta_out, fpath_out, fpath_east, fpath_west,
            map_col='gid_full', lon_seam=-105.0, chunk_size=1000):
        """Initialize and run the blender.

        Parameters
        ----------
        meta_out : str | pd.DataFrame
            Final output blended meta data (filepath or extracted df).
        fpath_out : str
            Filepath to save output file to.
        fpath_east : str
            NSRDB file for eastern extent.
        fpath_west : str
            NSRDB file for western extent.
        map_col : str, optional
            Column in the east and west meta data that map sites to the full
            meta_out gids.
        lon_seam : int, optional
            Vertical longitude seam at which data transitions from the western
            source to eastern, by default -105 (historical closest to nadir).
        chunk_size : int
            Number of sites to read/write at a time.
        """

        blndr = cls(meta_out, fpath_out, fpath_east, fpath_west,
                    map_col=map_col, lon_seam=lon_seam)

        blndr.blend(blndr._fpath_east, blndr._meta_east, chunk_size=chunk_size)
        blndr.blend(blndr._fpath_west, blndr._meta_west, chunk_size=chunk_size)
