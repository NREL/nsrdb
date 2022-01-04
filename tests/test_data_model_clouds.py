# pylint: skip-file
"""
PyTest file for PV generation in Rhode Island.

Created on Thu Nov 29 09:54:51 2018

@author: gbuster
"""
import datetime
import os
import shutil
import pytest
import numpy as np
import pandas as pd
import datetime
import h5py
import tempfile

from nsrdb import DEFAULT_VAR_META, TESTDATADIR, CONFIGDIR
from nsrdb.data_model import DataModel, VarFactory
from nsrdb.data_model.clouds import (CloudVar, CloudVarSingleH5,
                                     CloudVarSingleNC, CloudCoords)

RTOL = 0.001
ATOL = 0.001


DSET_RANGES = {'cld_opd_dcomp': (0, 160),
               'cld_reff_dcomp': (0, 160),
               'cloud_type': (-128, 15),
               'cld_press_acha': (0, 500)}

COORD_RANGES = {'latitude': (-90, 90),
                'longitude': (-360, 360)}


@pytest.fixture
def cloud_data():
    """Return cloud data for a single timestep."""

    kwargs = {'parallax_correct': False,
              'solar_shading': False,
              'remap_pc': False}
    fpath = os.path.join(TESTDATADIR, 'uw_test_cloud_data_h5/016/',
                         'goes12_2007_016_0000.level2.h5')
    c = CloudVarSingleH5(fpath, **kwargs)
    grid = c.grid
    data = c.source_data
    cloud_data = {'grid': grid, 'data': data}
    return cloud_data


def test_cloud_dir():
    """Test the basic CloudVar handler functions with an incomplete cloud data
    directory."""
    date = datetime.date(2022, 1, 4)

    fn = 'clavrx_OR_ABI-L1b-RadC-M6C01_G16_s*.level2.nc'
    cdir = os.path.join(TESTDATADIR, 'uw_test_cloud_data_nc/2022/004/')
    pattern = os.path.join(cdir, fn)

    msg = 'Bad number of cloud data files for 2022-01-04. Counted 6 files'
    with pytest.warns(UserWarning, match=msg):
        cv = CloudVar('cloud_type', var_meta=None, date=date, freq='5min',
                      pattern=pattern)
    assert (~pd.isna(cv.file_df['flist'])).sum() == 6
    assert len(cv.flist) == 6


@pytest.mark.parametrize('dset',
                         ('cloud_type',
                          'cld_opd_dcomp',
                          'cld_reff_dcomp',
                          'cld_press_acha',
                          ))
def test_single_dset_ranges(cloud_data, dset):
    """Test single timestep cloud data import"""

    mi = np.nanmin(cloud_data['data'][dset])
    ma = np.nanmax(cloud_data['data'][dset])
    msg = ('Bad range for "{}". Min/max are {}/{}, expected range is {}'
           .format(dset, mi, ma, DSET_RANGES[dset]))
    assert ma > DSET_RANGES[dset][0], msg
    assert mi < DSET_RANGES[dset][1], msg


@pytest.mark.parametrize('dset', ('latitude', 'longitude'))
def test_single_coords(cloud_data, dset):
    """Test cloud data coordinate ranges and nan values."""

    mi = np.nanmin(cloud_data['grid'][dset])
    ma = np.nanmax(cloud_data['grid'][dset])
    msg = ('Bad range for "{}". Min/max are {}/{}, expected range is {}'
           .format(dset, mi, ma, COORD_RANGES[dset]))
    assert np.sum(np.isnan(cloud_data['grid'][dset])) == 0, 'NaN coords exist!'
    assert ma > COORD_RANGES[dset][0], msg
    assert mi < COORD_RANGES[dset][1], msg


def test_regrid():
    """Test the cloud regrid algorithm."""

    cloud_vars = DataModel.CLOUD_VARS
    var_meta = DEFAULT_VAR_META
    date = datetime.date(year=2007, month=1, day=16)
    pattern = os.path.join(TESTDATADIR, 'uw_test_cloud_data_h5/{doy}/*.h5')
    kwargs = {'pattern': pattern,
              'parallax_correct': False,
              'solar_shading': False,
              'remap_pc': False}
    factory_kwargs = {v: kwargs for v in cloud_vars}
    nsrdb_grid = os.path.join(TESTDATADIR, 'reference_grids',
                              'east_psm_extent.csv')
    out_dir = os.path.join(TESTDATADIR, 'processed_ancillary')

    data = DataModel.run_clouds(cloud_vars, date, nsrdb_grid,
                                nsrdb_freq='1d', var_meta=var_meta,
                                factory_kwargs=factory_kwargs,
                                max_workers_regrid=1,
                                max_workers_cloud_io=1)
    for k in data.keys():
        data[k] = data[k][0, :].ravel()

    for key, value in data.items():
        mask = np.isnan(value)
        data[key][mask] = -15
        baseline_path = os.path.join(out_dir, key + '.h5')
        if not os.path.exists(baseline_path):
            with h5py.File(baseline_path, 'w') as f:
                f.create_dataset(key, data=value)
            msg = 'Output file for "{}" did not exist, created'.format(key)
            assert False, msg
        else:
            with h5py.File(baseline_path, 'r') as f:
                data_baseline = f[key][...]
                var_obj = VarFactory.get_base_handler(
                    key, var_meta=var_meta, date=date)
                data_baseline = var_obj.scale_data(data_baseline)

            assert np.allclose(data_baseline, value,
                               atol=ATOL, rtol=RTOL)


def test_regrid_duplicates():
    """Test the cloud regrid algorithm with duplicate coordinates."""

    cloud_vars = DataModel.CLOUD_VARS
    var_meta = DEFAULT_VAR_META
    date = datetime.date(year=2007, month=1, day=16)

    kwargs = {'parallax_correct': False,
              'solar_shading': False,
              'remap_pc': False}

    with tempfile.TemporaryDirectory() as td:
        source_clouds = os.path.join(TESTDATADIR, 'uw_test_cloud_data_h5')
        temp_clouds = os.path.join(td, 'clouds/')
        temp_cloud_fp = os.path.join(
            temp_clouds, '016/goes12_2007_016_0000.level2.h5')
        shutil.copytree(source_clouds, temp_clouds)

        cv = CloudVarSingleH5(temp_cloud_fp, pre_proc_flag=True, **kwargs)
        assert cv.grid.duplicated().sum() == 0

        with h5py.File(temp_cloud_fp, 'a') as res:
            lat_copy = res['latitude'][1000, 1000]
            lon_copy = res['longitude'][1000, 1000]
            res['latitude'][1000:1025, 1000:1025] = lat_copy
            res['longitude'][1000:1025, 1000:1025] = lon_copy

        cv = CloudVarSingleH5(temp_cloud_fp, pre_proc_flag=False, **kwargs)
        assert cv.grid.duplicated().sum() > 100

        cv = CloudVarSingleH5(temp_cloud_fp, pre_proc_flag=True, **kwargs)
        assert cv.grid.duplicated().sum() == 0


def test_regrid_big_dist():
    """Test the data model regrid process mapping cloud data to a bad meta data
    that is very far from the cloud data coordinates.
    """
    cloud_vars = DataModel.CLOUD_VARS
    var_meta = os.path.join(CONFIGDIR, 'nsrdb_vars.csv')
    date = datetime.date(year=2007, month=1, day=16)
    pattern = os.path.join(TESTDATADIR, 'uw_test_cloud_data_h5/{doy}/*.h5')
    kwargs = {'pattern': pattern,
              'parallax_correct': False,
              'solar_shading': False,
              'remap_pc': False}
    factory_kwargs = {v: kwargs for v in cloud_vars}
    nsrdb_grid = os.path.join(TESTDATADIR, 'reference_grids',
                              'west_psm_extent.csv')

    msg = 'following NSRDB gids were further than'
    with pytest.warns(UserWarning, match=msg):
        data = DataModel.run_clouds(cloud_vars, date, nsrdb_grid,
                                    nsrdb_freq='1d', var_meta=var_meta,
                                    factory_kwargs=factory_kwargs,
                                    max_workers_regrid=1,
                                    max_workers_cloud_io=1)

    # test that the data model assigned missing values
    assert (data['cloud_type'] == -15).all()
    assert (data['cld_opd_dcomp'] == 0).all()
    assert (data['cld_reff_dcomp'] == 0).all()
    assert (data['cld_press_acha'] == 0).all()

    data = DataModel.run_clouds(cloud_vars, date, nsrdb_grid,
                                nsrdb_freq='1d', var_meta=var_meta,
                                factory_kwargs=factory_kwargs,
                                dist_lim=1e6,
                                max_workers_regrid=1,
                                max_workers_cloud_io=1)

    # test that the data model mapped NN over a large distance
    assert (data['cloud_type'] != -15).all()


def test_bad_kwargs():
    """Test that the cloud hander raises an error if proper path pattern is
    not provided"""

    cloud_vars = DataModel.CLOUD_VARS
    var_meta = DEFAULT_VAR_META
    date = datetime.date(year=2007, month=1, day=16)
    nsrdb_grid = os.path.join(TESTDATADIR, 'reference_grids',
                              'east_psm_extent.csv')

    with pytest.raises(RuntimeError):
        DataModel.run_clouds(cloud_vars, date, nsrdb_grid,
                             nsrdb_freq='1d', var_meta=var_meta,
                             max_workers_regrid=1,
                             max_workers_cloud_io=1)


def test_sensor_azi_calc():
    """Test the calculation of sensor azimuth angle from lat/lon/zenith. This
    is necessary because old files don't have sensor_azimuth_angle datasets"""

    fn = 'clavrx_OR_ABI-L1b-RadC-M6C01_G16_s20220041341174_PR.level2.nc'
    cdir = os.path.join(TESTDATADIR, 'uw_test_cloud_data_nc/2022/004/')
    fp = os.path.join(cdir, fn)

    dsets = ('cloud_type', 'cld_opd_dcomp', 'cld_reff_dcomp', 'cld_press_acha',
             'solar_zenith_angle', 'solar_azimuth_angle',
             'sensor_zenith_angle', 'sensor_azimuth_angle')
    cv_raw = CloudVarSingleNC(fp, dsets=dsets, parallax_correct=False,
                              solar_shading=False, remap_pc=False)

    grid = cv_raw.grid
    sensor_zenith = cv_raw.source_data['sensor_zenith_angle']
    azi_raw = cv_raw.source_data['sensor_azimuth_angle']

    azi_calc = CloudCoords.calc_sensor_azimuth(grid.latitude, grid.longitude,
                                               sensor_zenith)

    assert np.allclose(azi_raw, azi_calc, rtol=0.001)


def test_parallax_shading_correct(plot=False):
    """Test parallax/shading correction and remapping algorithms. Most of the
    power of this test in the ability to plot and manually inspect the results.
    """

    # solar zenith ~84deg
    fn = 'clavrx_OR_ABI-L1b-RadC-M6C01_G16_s20220041121174_PR.level2.nc'
    cdir = os.path.join(TESTDATADIR, 'uw_test_cloud_data_nc/2022/004/')

    # solar zenith ~57deg
    fn = 'clavrx_OR_ABI-L1b-RadC-M6C01_G16_s20220041341174_PR.level2.nc'
    cdir = os.path.join(TESTDATADIR, 'uw_test_cloud_data_nc/2022/004/')

    fp = os.path.join(cdir, fn)

    xlim = (-66, -65)
    ylim = (18, 19)

    dsets = ('cloud_type', 'cld_opd_dcomp', 'cld_reff_dcomp',
             'cld_height_acha', 'cld_press_acha', 'solar_zenith_angle',
             'solar_azimuth_angle', 'sensor_zenith_angle',
             'sensor_azimuth_angle')

    cv_raw = CloudVarSingleNC(fp, dsets=dsets, parallax_correct=False,
                              solar_shading=False, remap_pc=False)
    cv_pc = CloudVarSingleNC(fp, dsets=dsets, parallax_correct=True,
                             solar_shading=True, remap_pc=False)
    cv_remap = CloudVarSingleNC(fp, dsets=dsets, parallax_correct=True,
                                solar_shading=True, remap_pc=True)

    assert np.allclose(cv_remap.grid, cv_raw.grid)
    assert not np.allclose(cv_remap.grid, cv_pc.grid)

    if plot:
        import matplotlib.pyplot as plt

        jobs = {'raw': cv_raw, 'pc': cv_pc, 'remap': cv_remap}

        for name, cv_obj in jobs.items():
            grid = cv_obj.grid

            kws_dict = {'cloud_type': {'vmin': 0, 'vmax': 5, 'cmap': 'tab20'},
                        'cld_opd_dcomp': {'vmin': 0, 'vmax': 10}}

            for dset in ('cloud_type', 'cld_opd_dcomp', 'cld_height_acha'):
                data = cv_obj.source_data[dset]
                kws = kws_dict.get(dset, {})

                fig = plt.figure(figsize=(10, 7))
                a = plt.scatter(grid.longitude, grid.latitude, c=data,
                                marker='s', s=50, linewidth=0, **kws)
                plt.xlim(xlim)
                plt.ylim(ylim)
                plt.colorbar(a, label=dset)
                plt.savefig('./_focus_pr_clouds_{}_{}.png'.format(dset, name),
                            dpi=300)
                plt.close()

                fig = plt.figure(figsize=(10, 7))
                a = plt.scatter(grid.longitude, grid.latitude, c=data,
                                marker='s', s=10, linewidth=0, **kws)
                plt.colorbar(a, label=dset)
                plt.savefig('./_full_pr_clouds_{}_{}.png'.format(dset, name),
                            dpi=300)
                plt.close()


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
