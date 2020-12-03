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
GRID = os.path.join(PROJECT_DIR, 'surfrad_meta.csv')


def test_all_sky_daily(date='20190102'):
    """Test the mlclouds fill on daily files then all sky from those files."""
    if os.path.exists(DAILY_DIR):
        shutil.rmtree(DAILY_DIR)
    shutil.copytree(ARCHIVE_DIR, DAILY_DIR)

    h5_source = os.path.join(DAILY_DIR, '{}*.h5'.format(date))
    PhygnnCloudFill.run(h5_source)

    NSRDB.run_daily_all_sky(PROJECT_DIR, date[0:4], GRID, date, freq='5min',
                            max_workers=1)

    dsets = ('dni', 'ghi', 'dhi', 'fill_flag', 'clearsky_dhi',
             'clearsky_dni', 'clearsky_ghi')
    with MultiFileNSRDB(h5_source) as res:
        assert all([d in res.dsets for d in dsets])
        dni = res['dni']
        ghi = res['ghi']
        dhi = res['dhi']
        sza = res['solar_zenith_angle']
        fill_flag = res['fill_flag']
        cloud_fill_flag = res['cloud_fill_flag']
        ti = res.time_index

    assert ~(ghi == 0).all(axis=0).any()
    assert ~(dni == 0).all(axis=0).any()
    assert ~(dhi == 0).all(axis=0).any()
    assert ~(sza == 0).all(axis=0).any()
    assert (fill_flag > 0).sum() > 50
    assert (cloud_fill_flag > 0).sum() > 50
    day_mask = (sza < 89)
    assert day_mask.any()
    assert (ghi[day_mask] > 0).all()
    diffuse_mask = day_mask & (dni == 0)
    not_diffuse_mask = day_mask & (dhi == 0)
    assert (dhi[diffuse_mask] > 0).all()
    assert (dni[not_diffuse_mask] > 0).all()
    assert (fill_flag >= cloud_fill_flag).all()

    p1 = os.path.join(PROJECT_DIR, 'collect/')
    p2 = os.path.join(PROJECT_DIR, 'final/')
    p3 = os.path.join(PROJECT_DIR, 'logs/')
    for path in (p1, p2, p3, DAILY_DIR):
        if os.path.exists(path):
            shutil.rmtree(path)


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
