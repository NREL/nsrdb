# pylint: skip-file
"""
PyTest file for all sky daily processing after mlclouds daily gap fill

Created on 12/3/2020

@author: gbuster
"""
import h5py
from copy import deepcopy
import pytest
import numpy as np
import os
import shutil
from nsrdb import TESTDATADIR
from nsrdb.nsrdb import NSRDB
from nsrdb.all_sky import CLOUD_TYPES
from rex import MultiFileNSRDB

pytest.importorskip("mlclouds")
pytest.importorskip("phygnn")
from nsrdb.gap_fill.phygnn_fill import PhygnnCloudFill


PROJECT_DIR = os.path.join(TESTDATADIR, 'mlclouds_pipeline/')
ARCHIVE_DIR = os.path.join(PROJECT_DIR, 'daily_files_archive/')
DAILY_DIR = os.path.join(PROJECT_DIR, 'daily/')
FINAL_DIR = os.path.join(PROJECT_DIR, 'final/')
GRID = os.path.join(PROJECT_DIR, 'surfrad_meta.csv')


def test_collect(dates=('20190102', '20190103', '20190104')):
    """Test the mlclouds gap fill, allsky, and final collection."""
    if os.path.exists(DAILY_DIR):
        shutil.rmtree(DAILY_DIR, ignore_errors=True)
    shutil.copytree(ARCHIVE_DIR, DAILY_DIR)

    for date in dates:
        h5_source = os.path.join(DAILY_DIR, '{}*.h5'.format(date))
        PhygnnCloudFill.run(h5_source)

        NSRDB.run_daily_all_sky(PROJECT_DIR, date[0:4], GRID, date,
                                freq='5min', max_workers=1)

    for i_fname in range(len(NSRDB.OUTS)):
        NSRDB.collect_data_model(PROJECT_DIR, date[0:4], GRID, n_chunks=1,
                                 i_chunk=0, i_fname=i_fname, freq='5min',
                                 max_workers=1, job_name='mlclouds_test',
                                 final=True, final_file_name='mlclouds_test')

    fns = os.listdir(FINAL_DIR)
    assert len(fns) == 7
    assert all([fn.startswith('mlclouds_test_') for fn in fns])
    all_data = {}
    all_attrs = {}
    with MultiFileNSRDB(os.path.join(FINAL_DIR, 'mlclouds_test_*.h5')) as res:
        dsets = res.dsets
        meta = res.meta
        ti = res.time_index
        assert len(dsets) == 28
        assert len(meta) == 9
        assert len(ti) == (288 * 3)
        dsets = [d for d in dsets if d not in ('time_index', 'meta')]
        for dset in dsets:
            data = res[dset]
            assert (data != 0).sum() > 0
            all_data[dset] = data
            all_attrs[dset] = res.get_attrs(dset)

    with MultiFileNSRDB(h5_source) as res:
        L = len(res.time_index)
        for dset in dsets:
            assert np.allclose(res[dset], all_data[dset][-L:, :])

            attrs = res.get_attrs(dset)
            attrs_final = all_attrs[dset]

            assert 'units' in attrs
            assert 'scale_factor' in attrs
            assert 'psm_units' in attrs
            assert 'psm_scale_factor' in attrs

            for k, v in attrs.items():
                assert k in attrs_final
                if ('dcomp' not in dset
                        and k not in ('physical_max', 'source_dir')):
                    msg = '{}: {}: {} vs {}'.format(dset, k, v, attrs_final[k])
                    assert str(v) == str(attrs_final[k]), msg

    p1 = os.path.join(PROJECT_DIR, 'collect/')
    p2 = os.path.join(PROJECT_DIR, 'logs/')
    for path in (p1, p2, FINAL_DIR, DAILY_DIR):
        if os.path.exists(path):
            shutil.rmtree(path, ignore_errors=True)

    fp = os.path.join(PROJECT_DIR, 'jobstatus_mlclouds_test.json')
    if os.path.exists(fp):
        os.remove(fp)


def execute_pytest(capture='all', flags='-rapP'):
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
