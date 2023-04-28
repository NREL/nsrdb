# -*- coding: utf-8 -*-
"""NSRDB multi-year mean calculation methods.
@author: gbuster
"""
import logging
import numpy as np
import os
import re
from warnings import warn

from rex.utilities.execution import SpawnProcessPool
from rex.utilities.utilities import split_sites_slice

from nsrdb.file_handlers.file_system import NSRDBFileSystem as NFS
from nsrdb.file_handlers.outputs import Outputs

logger = logging.getLogger(__name__)


class MyMean:
    """Class to calculate multi-year mean data"""

    def __init__(self, flist, fout, dset, process_chunk=10000, parallel=True):
        """
        Parameters
        ----------
        flist : list | tuple
            List of filepaths to NSRDB files to calculate means from.
        fout : str
            Output file path.
        dset : str
            Dataset name to calculate mean values for.
        process_chunk : int
            Number of sites to keep in memory and read at one time.
        parallel : bool
            Flag to run in parallel.
        """

        logger.info('Initializing Multi Year Mean calculation for "{}" with '
                    'output file: {}'.format(dset, fout))

        self._flist = flist
        self._fout = fout
        self._dset = dset
        self._process_chunk = process_chunk
        self._parallel = parallel
        self._meta = None

        self._units, self._shape, self._scale, self._dtype = self._preflight()
        self._years = self._parse_years()

        self._attrs = {'scale_factor': self._scale,
                       'psm_scale_factor': self._scale,
                       'units': self._units,
                       'psm_units': self._units,
                       'years': self._years}

        self._data = np.zeros((len(self),), dtype=np.float32)

        sites = len(self)
        split = int(sites / self._process_chunk)
        self._site_slices = split_sites_slice(slice(None), sites, split)

    def __len__(self):
        """Get the number of sites."""
        return self._shape[1]

    def _parse_years(self):
        """Parse the years from the filepaths.

        Returns
        -------
        years : list
            Sorted list of years.
        """
        years = []
        for f in self._flist:
            fname = os.path.basename(f)
            regex = r".*[^0-9]([1-2][0-9]{3})($|[^0-9])"
            match = re.match(regex, fname)

            if match:
                year = int(match.group(1))
                years.append(year)
            else:
                e = 'Cannot parse year from file: {}'.format(fname)
                logger.error(e)
                raise ValueError(e)
        years = sorted(years)
        logger.info('Running multi year mean over {} years: {}'
                    .format(len(years), years))
        return years

    def _preflight(self):
        """Run pre-flight checks.

        Returns
        -------
        base_units : str
            Units for dset.
        base_shape : tuple
            Full (not mean) dataset shape for dset.
        base_scale : int | float
            Scale factor for dset
        base_dtype : str
            Dataset array dtype
        """

        logger.info('Running preflight on {} files.'.format(len(self._flist)))

        base_units = None
        base_shape = None
        base_scale = None
        base_dtype = None

        for fpath in self._flist:
            logger.debug('\t- Checking file: {}'.format(fpath))
            with NFS(fpath, use_rex=True) as res:
                shape, base_dtype, _ = res.get_dset_properties(self._dset)
                units = res.get_units(self._dset)
                scale = res.get_scale_factor(self._dset)

            if base_shape is None and shape[0] % 8760 == 0:
                base_shape = shape
            elif base_shape is not None:
                # pylint: disable=unsubscriptable-object
                if base_shape[1] != shape[1]:
                    e = ('Dataset "{}" has inconsistent shapes! '
                         'Base shape was {}, but found new shape '
                         'of {} in fpath: {}'
                         .format(self._dset, base_shape, shape, fpath))
                    logger.error(e)
                    raise ValueError(e)

            if base_units is None:
                base_units = units
            else:
                if base_units != units:
                    e = ('Found inconsistent units for dataset "{}": {} and {}'
                         .format(self._dset, base_units, units))
                    logger.error(e)
                    raise ValueError(e)

            if base_scale is None:
                base_scale = scale
            else:
                if base_scale != scale:
                    w = ('Found inconsistent scale factor for dataset '
                         '"{}": {} and {}'
                         .format(self._dset, base_scale, scale))
                    logger.warning(w)
                    warn(w)

        logger.info('Preflight passed.')

        if self._dset in ('dni', 'ghi', 'dhi'):
            base_units = 'kWh/m2/day'
            base_scale = 1.0
            base_dtype = np.float32

        return base_units, base_shape, base_scale, base_dtype

    @staticmethod
    def to_kwh_m2_day(arr):
        """connvert irradiance to mean units (kwh/m2/day).

        Parameters
        ----------
        arr : np.ndarray | pd.Series
            Mean irradiance array in W/m2.

        Returns
        -------
        mean : float | np.ndarray
            Mean irradiance values in kWh/m2/day.
        """

        arr = arr / 1000 * 24
        return arr

    @property
    def meta(self):
        """Get the meta dataframe."""
        if self._meta is None:
            with NFS(self._flist[-1], use_rex=True) as res:
                self._meta = res.meta
        return self._meta

    def _run(self):
        """Run MY Mean calculation in serial or parallel"""
        if self._parallel:
            self._run_parallel()
        else:
            self._run_serial()

    def _run_serial(self):
        """Run the MY Mean calculation in serial."""

        for i, f in enumerate(self._flist):
            logger.info('Processing file {} out of {}: {}'
                        .format(i + 1, len(self._flist), f))
            with NFS(f, use_rex=True) as res:
                for j, site_slice in enumerate(self._site_slices):
                    new_data = res[self._dset, :, site_slice].mean(axis=0)
                    self._data[site_slice] += new_data
                    logger.info('Finished site slice {} out of {}'
                                .format(j + 1, len(self._site_slices)))

        self._data /= len(self._flist)
        self._data = self.to_kwh_m2_day(self._data)

    def _run_parallel(self):
        """Run the MY Mean calculation in a parallel process pool."""

        logger.info('Running MY Mean calculation in parallel.')
        for i, f in enumerate(self._flist):
            logger.info('Processing file {} out of {}: {}'
                        .format(i + 1, len(self._flist), f))
            futures = []
            loggers = ['nsrdb']
            with SpawnProcessPool(loggers=loggers) as exe:
                for site_slice in self._site_slices:
                    future = exe.submit(self._retrieve_data, f, self._dset,
                                        site_slice)
                    futures.append(future)

                for j, future in enumerate(futures):
                    site_slice = self._site_slices[j]
                    new_data = future.result()
                    self._data[site_slice] += new_data
                    if (j + 1) % 10 == 0:
                        logger.info('Finished site slice {} out of {}'
                                    .format(j + 1, len(self._site_slices)))

        self._data /= len(self._flist)
        self._data = self.to_kwh_m2_day(self._data)

    @staticmethod
    def _retrieve_data(fpath, dset, site_slice):
        """Retrieve mean data.

        Parameters
        ----------
        fpath : str
            Filepath to nsrdb data.
        dset : str
            Dataset of interest.
        site_slice : slice
            Sites to retrieve data for.

        Returns
        -------
        data : np.ndarray
            1D data averaged along axis 0 (time axis).
        """
        with NFS(fpath, use_rex=True) as res:
            data = res[dset, :, site_slice].mean(axis=0)
        return data

    def _write(self):
        """Write MY Mean data to disk"""
        logger.info('Writing "{}" data to disk: {}'
                    .format(self._dset, self._fout))

        out_dir = os.path.dirname(self._fout)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        with Outputs(self._fout, mode='a') as out:
            out['meta'] = self.meta
            if self._dset not in out.dsets:
                out._create_dset(self._dset, self._data.shape, self._dtype,
                                 attrs=self._attrs, data=self._data)
            else:
                out[self._dset] = self._data
        logger.info('Finished writing "{}" data to disk: {}'
                    .format(self._dset, self._fout))

    @classmethod
    def run(cls, flist, fout, dset, process_chunk=10000, parallel=True):
        """Run the MY mean calculation and write to disk.

        Parameters
        ----------
        flist : list | tuple
            List of filepaths to NSRDB files to calculate means from.
        fout : str
            Output file path.
        dset : str
            Dataset name to calculate mean values for.
        process_chunk : int
            Number of sites to keep in memory and read at one time.
        parallel : bool
            Flag to run in parallel.
        """
        mymean = cls(flist, fout, dset, process_chunk=process_chunk,
                     parallel=parallel)
        mymean._run()
        mymean._write()
        logger.info('MY Mean compute complete for "{}".'.format(dset))
