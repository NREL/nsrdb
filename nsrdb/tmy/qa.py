# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 09:11:49 2019

@author: gbuster
"""
import numpy as np
import time
from warnings import warn

from nsrdb.data_model import VarFactory
from nsrdb.file_handlers.resource import Resource


def run_checks(fp, i0, iend, step=1000):
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
    step : int
        Chunk size to read at once.
    """

    i0_static = i0

    with Resource(fp) as res:

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

            print('Starting on "{}"'.format(dset))
            var = VarFactory.get_base_handler(dset)

            if dset in var.var_meta['var'].values:
                print('Expected physical min/max: ',
                      var.physical_min, var.physical_max)

            attrs = res.get_attrs(dset=dset)
            if 'psm_scale_factor' not in attrs:
                raise KeyError('Could not find psm_scale_factor')
            if 'psm_units' not in attrs:
                raise KeyError('Could not find psm_units')

            i0 = i0_static
            while True:

                site_slice = slice(i0, i0 + step)
                data = res[dset, :, site_slice]
                print('\tSite {} min/max: {}/{}'
                      .format(site_slice, data.min(), data.max()))

                if dset in var.var_meta['var'].values:
                    if (data.min() < var.physical_min
                            or data.max() > var.physical_max):
                        m = 'Out of physical range!'
                        print(m)
                        warn(m)
                        time.sleep(.2)

                sum_data = data.sum(axis=0)

                if any(sum_data == 0):
                    m = ('Sites have full timeseries of zeros: {}'
                         .format(np.where(sum_data == 0)[0]))
                    print(m)
                    warn(m)
                    time.sleep(.2)

                if i0 > iend or (i0 + step) > iend:
                    break
                else:
                    i0 += step
