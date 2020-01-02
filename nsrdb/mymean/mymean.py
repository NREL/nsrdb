# -*- coding: utf-8 -*-
"""NSRDB multi-year mean calculation methods.
@author: gbuster
"""
from warnings import warn
import numpy as np
import os
import re
import logging
from nsrdb.file_handlers.resource import Resource
from nsrdb.file_handlers.outputs import Outputs


logger = logging.getLogger(__name__)


class MyMean:
    """Class to calculate multi-year mean data"""

    def __init__(self, flist, fout, dset, process_chunk=100000):
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
        """

        self._flist = flist
        self._fout = fout
        self._dset = dset
        self._process_chunk = process_chunk

        self._units, self._shape, self._scale, self._dtype = self._preflight()
        self._years = self._parse_years()

        self._attrs = {'scale_factor': self._scale,
                       'psm_scale_factor': self._scale,
                       'units': self._units,
                       'psm_units': self._units,
                       'years': self._years}

        self._data = np.zeros((len(self),), dtype=np.float32)

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
            with Resource(fpath) as res:
                shape, base_dtype, _ = res.get_dset_properties(self._dset)
                units = res.get_units(self._dset)
                scale = res.get_scale(self._dset)

            if base_shape is None and shape[0] % 8760 == 0:
                base_shape = shape
            elif base_shape is not None:
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

        return base_units, base_shape, base_scale, base_dtype

    def _run(self):
        """Run the MY Mean calculation."""
        sites = np.arange(len(self))
        split = int(len(self) / self._process_chunk)
        site_slices = np.array_split(sites, split)
        site_slices = [slice(a[0], a[-1] + 1) for a in site_slices]

        for i, f in enumerate(self._flist):
            logger.info('Processing file {} out of {}: {}'
                        .format(i + 1, len(self._flist), f))
            with Resource(f) as res:
                for j, site_slice in enumerate(site_slices):
                    logger.info('Processing site slice {} out of {}'
                                .format(j + 1, len(site_slices)))
                    new_data = res[self._dset, :, site_slice].mean(axis=0)
                    self._data[site_slice] += new_data

        self._data /= len(self._flist)

    def _write(self):
        """Write MY Mean data to disk"""
        logger.info('Writing "{}" data to disk: {}'
                    .format(self._dset, self._fout))
        with Outputs(self._fout, mode='a') as out:
            out._create_dset(self._dset, self._data.shape, self._dtype,
                             attrs=self._attrs, data=self._data)
        logger.info('Finished writing "{}" data to disk: {}'
                    .format(self._dset, self._fout))

    @classmethod
    def run(cls, flist, fout, dset, process_chunk=100000):
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
        """
        mymean = cls(flist, fout, dset, process_chunk=process_chunk)
        mymean._run()
        mymean._write()
        logger.info('MY Mean compute complete for "{}".'.format(dset))
