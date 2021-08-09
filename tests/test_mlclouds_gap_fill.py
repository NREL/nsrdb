# pylint: skip-file
"""
PyTest file for mlclouds gap fill on daily files.

Created on 12/3/2020

@author: gbuster
"""
import h5py
import pytest
import numpy as np
import os
import shutil
from nsrdb import TESTDATADIR
from farms import CLOUD_TYPES
from rex import MultiFileNSRDB

pytest.importorskip("mlclouds")
pytest.importorskip("phygnn")
from nsrdb.gap_fill.mlclouds_fill import MLCloudsFill


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

    with pytest.raises(FileNotFoundError):
        MLCloudsFill.run(h5_source)

    shutil.rmtree(DAILY_DIR, ignore_errors=True)


@pytest.mark.parametrize(('col_chunk', 'max_workers', 'date'),
                         ((None, 1, '20190102'),
                          (3, 1, '20190102'),
                          (2, 2, '20190102'),
                          (None, 2, '20190102'),
                          (None, 1, '20190103'),
                          (None, 1, '20190104'),
                          ))
def test_mlclouds_fill(col_chunk, max_workers, date):
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
    MLCloudsFill.run(h5_source, col_chunk=col_chunk,
                     max_workers=max_workers)

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

    day = (sza < 90)

    # make sure some timesteps were missing cloud type and were filled
    missing = day & (raw_ctype < 0)
    assert missing.any()
    assert (fill_ctype >= 0).all()

    # make sure there was at least one case with full timeseries of missing
    # ctype and make sure it got set to clear sky
    missing_full_timeseries = (missing | ~day).all(axis=0)
    assert any(missing_full_timeseries)
    i = np.where(missing_full_timeseries)[0][0]
    assert (2 == fill_flag[:, i]).all()
    assert (fill_ctype[:, i] == 0).all()

    # make sure some timesteps were missing cloud pressure and were filled
    missing = day & (raw_press <= 0)
    assert missing.any()
    assert (fill_press >= 0).all() & (np.isnan(fill_press).sum() == 0)

    # make sure some clouds were missing opd
    missing = day & np.isin(fill_ctype, CLOUD_TYPES) & (raw_opd <= 0)
    assert missing.any()

    # make sure that the missing opd was filled with appropriate fill flag
    assert (fill_opd[missing] > 0).all() & (np.isnan(fill_opd).sum() == 0)
    assert ((fill_flag[missing] == 7) | (fill_flag[missing] == 1)).all()

    # make sure that some clouds were missing reff and were filled
    missing = day & np.isin(fill_ctype, CLOUD_TYPES) & (raw_reff <= 0)
    assert missing.any()
    assert (fill_reff[missing] > 0).all() & (np.isnan(fill_reff).sum() == 0)
    assert ((fill_flag[missing] == 7) | (fill_flag[missing] == 1)).all()

    # make sure that all nightime opd and reff is 0
    assert fill_opd[~day].sum() == 0
    assert fill_reff[~day].sum() == 0

    # make sure that all fill flags are in the expected values
    assert np.isin(fill_flag, (0, 1, 2, 7)).all()

    shutil.rmtree(DAILY_DIR, ignore_errors=True)


def test_mlclouds_fill_all(col_chunk=None, max_workers=1, date='20190102'):
    """Test the mlclouds fill process on ALL cloudy timesteps."""
    fp_ctype = os.path.join(DAILY_DIR, '{}_cloud_type_0.h5'.format(date))
    if os.path.exists(DAILY_DIR):
        shutil.rmtree(DAILY_DIR, ignore_errors=True)
    shutil.copytree(ARCHIVE_DIR, DAILY_DIR)

    with h5py.File(fp_ctype, 'a') as res:
        ctype = res['cloud_type']
        ctype[:, -1] = -15
        ctype[slice(None, None, 10), 0] = -15

    h5_source = os.path.join(DAILY_DIR, '{}*.h5'.format(date))
    MLCloudsFill.run(h5_source, col_chunk=col_chunk,
                     max_workers=max_workers, fill_all=True)

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
        sza = res['solar_zenith_angle']

    day = (sza < 90)
    cloudy = day & np.isin(fill_ctype, CLOUD_TYPES)

    missing = day & (raw_ctype < 0)
    assert missing.any()
    assert (fill_ctype >= 0).all()

    missing = day & (raw_press <= 0)
    assert missing.any()
    assert (fill_press >= 0).all() & (np.isnan(fill_press).sum() == 0)

    missing = cloudy & (raw_opd <= 0)
    assert missing.any()
    assert (fill_opd[missing] > 0).all() & (np.isnan(fill_opd).sum() == 0)
    frac_same = (fill_opd[cloudy] == raw_opd[cloudy]).sum() / cloudy.sum()
    assert frac_same < 0.01

    missing = cloudy & (raw_reff <= 0)
    assert missing.any()
    assert (fill_reff[missing] > 0).all() & (np.isnan(fill_reff).sum() == 0)
    frac_same = (fill_reff[cloudy] == raw_reff[cloudy]).sum() / cloudy.sum()
    assert frac_same < 0.01

    assert fill_opd[~day].sum() == 0
    assert fill_reff[~day].sum() == 0

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
