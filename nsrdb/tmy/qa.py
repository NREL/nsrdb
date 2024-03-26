# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 09:11:49 2019

@author: gbuster
"""
import logging
import time
from warnings import warn

import numpy as np

from nsrdb.data_model import VarFactory
from nsrdb.file_handlers.file_system import NSRDBFileSystem as NFS

logger = logging.getLogger(__name__)


def run_checks(fp, i0, iend, interval=1, step=1000):
    """Run various checks on TMY files.

    Checks:
        - attributes (psm_scale_factor and psm_units)
        - data range min/max (compare to physical expected values)
        - Check for full timeseries of 0's

    Parameters
    ----------
    fp : str
        Filepath to .h5 file.
    i0 : int
        Starting site index.
    iend : int
        Ending site index.
    interval : int
        Interval to spot check instead of check everything from i0 to iend
    step : int
        Chunk size to read at once.
    """

    logger.info(
        'Running QA on {} from {} to {} with step size {} and interval {}'
        .format(fp, i0, iend, step, interval))

    n_split = int(np.ceil((iend - i0) / step))
    chunks = np.array_split(np.arange(i0, iend), n_split)
    chunks = [slice(x[0], x[-1] + 1) for x in chunks]
    chunks = chunks[::interval]

    with NFS(fp, use_rex=True) as res:

        ti = res.time_index
        meta = res.meta

        assert all(ti.minute == 30), 'Time_index must be at 30min!'
        assert len(ti) == 8760, 'Time_index must be an 8760!'

        assert 'tmy_year' in res.dsets, 'Could not find "tmy_year"'
        assert 'tmy_year_short' in res.dsets, 'Could not find "tmy_year_short"'

        shape, _, _ = res.get_dset_properties('tmy_year')
        m = 'tmy_year shape is bad: {}'.format(shape)
        assert shape == (len(ti), len(meta)), m

        shape, _, _ = res.get_dset_properties('tmy_year_short')
        m = 'tmy_year_short shape is bad: {}'.format(shape)
        assert shape == (12, len(meta)), m

        dsets = [d for d in res.dsets if d not in ['meta', 'time_index']]

        for dset in dsets:

            logger.info('Starting on "{}"'.format(dset))
            var = VarFactory.get_base_handler(dset)

            if dset in var.var_meta['var'].values:
                logger.info('Expected physical min/max: {} / {}'
                            .format(var.physical_min, var.physical_max))

            attrs = res.get_attrs(dset=dset)
            if 'psm_scale_factor' not in attrs:
                raise KeyError('Could not find psm_scale_factor')
            if 'psm_units' not in attrs:
                raise KeyError('Could not find psm_units')

            for site_slice in chunks:
                data = res[dset, :, site_slice]
                logger.info('\tSite {} min/max: {}/{}'
                            .format(site_slice, data.min(), data.max()))

                if (dset in var.var_meta['var'].values
                        and (data.min() < var.physical_min
                             or data.max() > var.physical_max)):
                    m = ('Out of physical range: {} / {}!'
                         .format(var.physical_min, var.physical_max))
                    logger.info(m)
                    warn(m)
                    time.sleep(.2)

                all_zeros = (data == 0).all(axis=0)

                if any(all_zeros):
                    m = ('Sites have full timeseries of zeros: {} in: {}'
                         .format(np.where(all_zeros)[0], site_slice))
                    logger.info(m)
                    warn(m)
                    time.sleep(.2)
