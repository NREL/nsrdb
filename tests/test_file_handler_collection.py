# pylint: skip-file
"""
PyTest file for PV generation in Rhode Island.

Created on Thu Nov 29 09:54:51 2018

@author: gbuster
"""

import os
import pytest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_index_equal
import datetime

from nsrdb import TESTDATADIR, CONFIGDIR
from nsrdb.data_model import DataModel, VarFactory
from nsrdb.file_handlers.outputs import Outputs
from nsrdb.file_handlers.collection import Collector


RTOL = 0.05
ATOL = .1
PURGE_OUT = True

BASELINE_4DAY = os.path.join(TESTDATADIR, 'collection_baseline/',
                             'collection_baseline_4day.h5')
BASELINE_ANNUAL = os.path.join(TESTDATADIR, 'collection_baseline/',
                               'collection_baseline_annual.h5')


def retrieve_data(fp, dset='air_temperature'):
    """get data from h5."""

    with Outputs(fp, mode='r') as f:
        ti = f.time_index
        meta = f.meta
        data = f[dset]
        attrs = f.get_attrs(dset)

    return ti, meta, data, attrs


@pytest.mark.parametrize(('sites', 'parallel'),
                         ((None, False),
                          (None, True),
                          (np.array([2, 3, 4]), False)))
def test_collect_daily(sites, parallel):

    collect_dir = os.path.join(TESTDATADIR, 'data_model_daily_sample_output/')
    f_out = os.path.join(TESTDATADIR, 'temp_out/collected.h5')
    dsets = ['air_temperature', 'alpha']

    if os.path.exists(f_out):
        os.remove(f_out)

    Collector.collect_daily(collect_dir, f_out, dsets, sites=sites,
                            parallel=parallel)

    for dset in dsets:
        b_ti, b_meta, b_data, b_attrs = retrieve_data(BASELINE_4DAY, dset=dset)
        t_ti, t_meta, t_data, t_attrs = retrieve_data(f_out, dset=dset)

        if sites is not None:
            b_meta = b_meta.iloc[sites, :].reset_index(drop=True)
            b_data = b_data[:, sites]

        assert_index_equal(b_ti, t_ti)
        assert_frame_equal(b_meta, t_meta)
        assert np.allclose(b_data, t_data)
        for k, v in b_attrs.items():
            assert str(v) == str(t_attrs[k])

    if PURGE_OUT:
        os.remove(f_out)


def test_collect_lowmem():
    """Test a low memory file by file collection"""
    flist = ['nsrdb_ancillary_2018_0.h5',
             'nsrdb_ancillary_2018_1.h5']
    collect_dir = os.path.join(TESTDATADIR, 'data_model_annual_sample_output/')
    f_out = os.path.join(TESTDATADIR, 'temp_out/collected.h5')

    dsets = ['surface_albedo', 'alpha', 'aod', 'surface_pressure']
    for dset in dsets:
        Collector.collect_flist_lowmem(flist, collect_dir, f_out, dset)

        ti, meta, data, attrs = retrieve_data(BASELINE_ANNUAL, dset=dset)

        with Outputs(os.path.join(collect_dir, flist[1]), mode='r') as f:
            attrs = f.get_attrs(dset=dset)
            meta_chunk = f.meta
            data_chunk = f[dset]

        assert 'scale_factor' in attrs
        assert 'data_source' in attrs
        assert 'spatial_interp_method' in attrs
        assert_frame_equal(meta.iloc[5:].reset_index(drop=True), meta_chunk)
        assert np.allclose(data[:, 5:], data_chunk)
        assert len(ti) == 105120

    if PURGE_OUT:
        os.remove(f_out)


def execute_pytest(capture='all', flags='-rapP', purge=True):
    """Execute module as pytest with detailed summary report.

    Parameters
    ----------
    capture : str
        Log or stdout/stderr capture option. ex: log (only logger),
        all (includes stdout/stderr)
    flags : str
        Which tests to show logs and results for.
    """

    fname = os.path.basename(__file__)
    pytest.main(['-q', '--show-capture={}'.format(capture), fname, flags])


if __name__ == '__main__':
    execute_pytest()