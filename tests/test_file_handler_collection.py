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
import tempfile

from nsrdb import TESTDATADIR
from nsrdb.data_model import DataModel, VarFactory
from nsrdb.file_handlers.outputs import Outputs
from nsrdb.file_handlers.collection import Collector


RTOL = 0.05
ATOL = .1

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


@pytest.mark.parametrize(('sites', 'max_workers', 'n_writes'),
                         ((None, 1, 1),
                          (None, 2, 1),
                          (None, 2, 2),
                          (np.array([2, 3, 4]), 1, 1)))
def test_collect_daily(sites, max_workers, n_writes):
    with tempfile.TemporaryDirectory() as td:
        collect_dir = os.path.join(TESTDATADIR,
                                   'data_model_daily_sample_output/')
        f_out = os.path.join(td, 'collected.h5')
        dsets = ['air_temperature', 'alpha']

        Collector.collect_daily(collect_dir, f_out, dsets, sites=sites,
                                max_workers=max_workers, n_writes=n_writes)

        for dset in dsets:
            b_ti, b_meta, b_data, b_attrs = retrieve_data(BASELINE_4DAY,
                                                          dset=dset)
            t_ti, t_meta, t_data, t_attrs = retrieve_data(f_out, dset=dset)

            if sites is not None:
                b_meta = b_meta.iloc[sites, :].reset_index(drop=True)
                b_data = b_data[:, sites]

            assert_index_equal(b_ti, t_ti)
            assert_frame_equal(b_meta, t_meta)
            assert np.allclose(b_data, t_data)
            for k, v in b_attrs.items():
                # source dirs have changed during re-orgs
                if k != 'source_dir':
                    assert str(v) == str(t_attrs[k])


def test_collect_lowmem():
    """Test a low memory file by file collection"""
    with tempfile.TemporaryDirectory() as td:
        flist = ['nsrdb_ancillary_2018_0.h5',
                 'nsrdb_ancillary_2018_1.h5']
        collect_dir = os.path.join(TESTDATADIR,
                                   'data_model_annual_sample_output/')
        f_out = os.path.join(td, 'collected.h5')

        dsets = ['surface_albedo', 'alpha', 'aod', 'surface_pressure']
        for dset in dsets:
            Collector.collect_flist_lowmem(flist, collect_dir, f_out, dset)

            ti, meta, data, attrs = retrieve_data(BASELINE_ANNUAL, dset=dset)

            with Outputs(f_out, mode='r') as f:
                attrs = f.get_attrs(dset=dset)
                meta_collected = f.meta
                data_collected = f[dset]

            assert 'scale_factor' in attrs
            assert 'data_source' in attrs
            assert 'spatial_interp_method' in attrs
            assert_frame_equal(meta, meta_collected)
            assert len(ti) == 105120

            # reduced precision for changed dtypes (aod in Jan 2022)
            msg = 'Dset failed: {}'.format(dset)
            assert np.allclose(data, data_collected,
                               rtol=0.001, atol=0.001), msg


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
