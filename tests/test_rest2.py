# pylint: skip-file
"""
PyTest file for rest2.

Created on Feb 13th 2019

@author: gbuster
"""

import os
import h5py
import pytest
import numpy as np
import pandas as pd
import time
from warnings import warn
from nsrdb.all_sky.utilities import ti_to_radius, calc_beta, dark_night
from nsrdb.all_sky.rest2 import rest2, rest2_tddclr, rest2_tuuclr


RTOL = 0.001
ATOL = 0.001
TEST_FILE = './data/validation_nsrdb/nsrdb_surfrad_2017.h5'


@pytest.mark.parametrize('angle', (84.2608, 72.5424, 60.0000, 45.5730, 25.8419,
                                   0.00000))
def test_rest2_tddclr(angle):
    with h5py.File(TEST_FILE, 'r') as f:

        p = (f['surface_pressure'][...] /
             f['surface_pressure'].attrs['psm_scale_factor'])
        albedo = (f['surface_albedo'][...] /
                  f['surface_albedo'].attrs['psm_scale_factor'])
        ssa = (f['ssa'][...] / f['ssa'].attrs['psm_scale_factor'])
        alpha = (f['alpha'][...] / f['alpha'].attrs['psm_scale_factor'])
        ozone = (f['ozone'][...] / f['ozone'].attrs['psm_scale_factor'])
        w = (f['total_precipitable_water'][...] /
             f['total_precipitable_water'].attrs['psm_scale_factor'])

        ti = pd.to_datetime(f['time_index'][...].astype(str))

    radius = ti_to_radius(ti, n_cols=p.shape[1])
    z = angle

    t0 = time.time()
    Tddclr = rest2_tddclr(p, albedo, ssa, z, radius, alpha, 0, ozone, w)
    Tddclr = Tddclr[:, 0:10]
    print('Testing rest2_tddclr on data shape {0} took {1:.1f} seconds.'
          .format(p.shape, time.time() - t0))

    h5 = "./data/rest2/rest2_Tddclr.h5"

    dset = 'tddclr_{}'.format(angle)
    with h5py.File(h5, 'a') as f:
        if dset in f:
            baseline = f[dset][...]
            result = np.allclose(Tddclr, baseline, rtol=RTOL, atol=ATOL)
            assert result, 'rest2_tddclr test failed at angle {}'.format(angle)

        else:
            f.create_dataset(dset, data=Tddclr)
            raise ValueError('Baseline rest2_tddclr() outputs did not exist. '
                             'Test failed. Printed new outputs to: {}'
                             .format(h5))


def test_rest2_tuuclr():
    with h5py.File(TEST_FILE, 'r') as f:

        p = (f['surface_pressure'][:, 0:2] /
             f['surface_pressure'].attrs['psm_scale_factor'])
        albedo = (f['surface_albedo'][:, 0:2] /
                  f['surface_albedo'].attrs['psm_scale_factor'])
        ssa = (f['ssa'][:, 0:2] / f['ssa'].attrs['psm_scale_factor'])
        alpha = (f['alpha'][:, 0:2] / f['alpha'].attrs['psm_scale_factor'])
        ozone = (f['ozone'][:, 0:2] / f['ozone'].attrs['psm_scale_factor'])
        w = (f['total_precipitable_water'][:, 0:2] /
             f['total_precipitable_water'].attrs['psm_scale_factor'])

        ti = pd.to_datetime(f['time_index'][...].astype(str))

    radius = ti_to_radius(ti, n_cols=p.shape[1])

    t0 = time.time()

    Tuuclr = rest2_tuuclr(p, albedo, ssa, radius, alpha, ozone, w,
                          parallel=False)

    print('Testing rest2_tuuclr on data shape {0} took {1:.1f} seconds.'
          .format(p.shape, time.time() - t0))

    h5 = "./data/rest2/rest2_Tuuclr.h5"

    if os.path.exists(h5):
        with h5py.File(h5, 'r') as f:
            baseline = f['tuuclr'][...]
        result = np.allclose(Tuuclr, baseline, rtol=RTOL, atol=ATOL)
        assert result, 'rest2_tddclr() test failed.'

    else:
        with h5py.File(h5, 'w') as f:
            f.create_dataset('tuuclr', data=Tuuclr, dtype=Tuuclr.dtype)
        raise ValueError('Baseline rest2_tuuclr() outputs did not exist. '
                         'Test failed. Printed new outputs to: {}'
                         .format(h5))


def test_rest2():
    with h5py.File(TEST_FILE, 'r') as f:

        p = (f['surface_pressure'][...] /
             f['surface_pressure'].attrs['psm_scale_factor'])
        albedo = (f['surface_albedo'][...] /
                  f['surface_albedo'].attrs['psm_scale_factor'])
        aod = (f['aod'][...] / f['aod'].attrs['psm_scale_factor'])
        ssa = (f['ssa'][...] / f['ssa'].attrs['psm_scale_factor'])
        g = (f['asymmetry'][...] / f['asymmetry'].attrs['psm_scale_factor'])
        alpha = (f['alpha'][...] / f['alpha'].attrs['psm_scale_factor'])
        ozone = (f['ozone'][...] / f['ozone'].attrs['psm_scale_factor'])
        w = (f['total_precipitable_water'][...] /
             f['total_precipitable_water'].attrs['psm_scale_factor'])

        ti = pd.to_datetime(f['time_index'][...].astype(str))

        # moving forward, SolarPosition() should be used, #
        # but for benchmarking, must use old SZA
        z = (f['solar_zenith_angle'][...] /
             f['solar_zenith_angle'].attrs['psm_scale_factor'])

        baseline_dhi = f['clearsky_dhi'][...]
        baseline_dni = f['clearsky_dni'][...]
        baseline_ghi = f['clearsky_ghi'][...]

    radius = ti_to_radius(ti, n_cols=p.shape[1])
    beta = calc_beta(aod, alpha)

    t0 = time.time()
    rest_data = rest2(p, albedo, ssa, g, z, radius, alpha, beta, ozone, w)

    rest_data.dhi = dark_night(rest_data.dhi, z)
    rest_data.dni = dark_night(rest_data.dni, z)
    rest_data.ghi = dark_night(rest_data.ghi, z)

    print('Testing rest2 on data shape {0} took {1:.1f} seconds.'
          .format(p.shape, time.time() - t0))

    check_vars = ('dni', 'dhi', 'ghi', 'Ruuclr', 'Tddclr', 'Tduclr')

    # benchmark against previous results from this script
    for var in check_vars:
        data = getattr(rest_data, var)[:, 0:10]

        h5 = "./data/rest2/rest2.h5".format(var)

        with h5py.File(h5, 'a') as f:

            if var in f:
                baseline = f[var][...]
                result = np.allclose(data, baseline, rtol=RTOL, atol=ATOL)
                assert result, 'REST2 test failed for "{}"'.format(var)

            else:
                result = False
                f.create_dataset(var, data=data, dtype=data.dtype)
                warn('Baseline REST2 outputs for "{}" did not exist. '
                     'Test failed. Printed new outputs to: {}'
                     .format(var, h5))
    assert result

    baseline = (('dhi', baseline_dhi),
                ('dni', baseline_dni),
                ('ghi', baseline_ghi))

    # benchmark against previous results from NSRDB v3.0.1
    for name, baseline_data in baseline:
        data = getattr(rest_data, name)[:, 0:10]

        # reduced absolute tolerance for irradiance against NSRDB v3.0.1
        result = np.allclose(data, baseline_data[:, 0:10], rtol=RTOL, atol=5)

        diff = np.max(np.abs(data - baseline_data[:, 0:10]))

        msg = ('"{}" clearsky irradiance benchmark against original NSRDB '
               'data failed with max diff of {}.'.format(name, diff))
        assert result, msg


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
