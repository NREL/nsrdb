# pylint: skip-file
"""
PyTest file for mlclouds gap fill on daily files.

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
from nsrdb.all_sky import CLOUD_TYPES
from rex import MultiFileNSRDB

pytest.importorskip("mlclouds")
pytest.importorskip("phygnn")
from nsrdb.gap_fill.phygnn_fill import PhygnnCloudFill


ARCHIVE_DIR = os.path.join(TESTDATADIR,
                           'mlclouds_pipeline/daily_files_archive/')
DAILY_DIR = os.path.join(TESTDATADIR, 'mlclouds_pipeline/daily/')


def test_missing_file(date='20190102'):
    """Test the mlclouds raise error on missing files."""
    fp_refl = os.path.join(DAILY_DIR, '{}_refl_0_65um_nom_0.h5'.format(date))
    fp_temp = os.path.join(DAILY_DIR, '{}_temp_11_0um_nom_0.h5'.format(date))
    if os.path.exists(DAILY_DIR):
        shutil.rmtree(DAILY_DIR, ignore_errors=True)
    shutil.copytree(ARCHIVE_DIR, DAILY_DIR)
    os.remove(fp_refl)
    os.remove(fp_temp)
    h5_source = os.path.join(DAILY_DIR, '{}*.h5'.format(date))

    with pytest.raises(FileNotFoundError) as e:
        PhygnnCloudFill.run(h5_source)

    shutil.rmtree(DAILY_DIR, ignore_errors=True)


def test_mlclouds_fill(date='20190102'):
    """Test the mlclouds fill process on a daily output from the data model."""
    fp_ctype = os.path.join(DAILY_DIR, '{}_cloud_type_0.h5'.format(date))
    if os.path.exists(DAILY_DIR):
        shutil.rmtree(DAILY_DIR, ignore_errors=True)
    shutil.copytree(ARCHIVE_DIR, DAILY_DIR)

    with h5py.File(fp_ctype, 'a') as res:
        ctype = res['cloud_type']
        ctype[:, -1] = -15
        ctype[slice(None, None, 10), 0] = -15

    h5_source = os.path.join(DAILY_DIR, '{}*.h5'.format(date))
    PhygnnCloudFill.run(h5_source)

    raw_files = os.path.join(DAILY_DIR, 'raw/{}_*_0.h5'.format(date))
    fill_files = os.path.join(DAILY_DIR, '{}_*_0.h5'.format(date))

    with MultiFileNSRDB(raw_files) as res:
        raw_opd = res['cld_opd_dcomp']
        raw_reff = res['cld_reff_dcomp']
        raw_press = res['cld_press_acha']
        raw_ctype = res['cloud_type']

    with MultiFileNSRDB(fill_files) as res:
        fill_opd = res['cld_opd_dcomp']
        fill_reff = res['cld_reff_dcomp']
        fill_press = res['cld_press_acha']
        fill_ctype = res['cloud_type']
        fill_flag = res['cloud_fill_flag']
        sza = res['solar_zenith_angle']

    missing = (sza < 90) & (raw_ctype < 0)
    assert missing.any()
    assert (fill_ctype >= 0).all()

    missing = (sza < 90) & (raw_press <= 0)
    assert missing.any()
    assert (fill_press >= 0).all() & (np.isnan(fill_press).sum() == 0)

    missing = (sza < 90) & np.isin(fill_ctype, CLOUD_TYPES) & (raw_opd <= 0)
    assert missing.any()
    assert (fill_opd[missing] > 0).all() & (np.isnan(fill_opd).sum() == 0)

    missing = (sza < 90) & np.isin(fill_ctype, CLOUD_TYPES) & (raw_reff <= 0)
    assert missing.any()
    assert (fill_reff[missing] > 0).all() & (np.isnan(fill_reff).sum() == 0)

    assert all(np.unique(fill_flag) == np.arange(4))

    shutil.rmtree(DAILY_DIR, ignore_errors=True)


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
