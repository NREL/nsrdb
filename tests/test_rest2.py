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
from all_sky.utilities import ti_to_radius, calc_beta
from all_sky.rest2 import rest2, rest2_tddclr


RTOL = 1e-03
ATOL = 0.001


@pytest.mark.parametrize('angle', (84.2608, 72.5424, 60.0000, 45.5730, 25.8419,
                                   0.00000))
def test_rest2_tddclr(angle):
    test_file = './data/test_nsrdb_data.h5'
    with h5py.File(test_file, 'r') as f:

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
    z = angle * np.ones(p.shape)

    t0 = time.time()
    Tddclr = rest2_tddclr(p, albedo, ssa, z, radius, alpha, 0, ozone, w)
    Tddclr = Tddclr[:, 0:10]
    print('Testing rest2_tddclr on data shape {0} took {1:.1f} seconds.'
          .format(p.shape, time.time() - t0))

    csv = "./data/rest2/rest2_Tddclr_{}.csv".format(int(angle))

    if os.path.exists(csv):
        baseline = np.genfromtxt(csv, delimiter=',')
        result = np.allclose(Tddclr, baseline, rtol=RTOL, atol=ATOL)
        assert result, 'rest2_tddclr() test failed for angle {}'.format(angle)

    else:
        np.savetxt(csv, Tddclr, delimiter=",")
        raise ValueError('Baseline rest2_tddclr() outputs did not exist. '
                         'Test failed. Printed new outputs to: {}'
                         .format(csv))


def test_rest2():
    test_file = './data/test_nsrdb_data.h5'
    with h5py.File(test_file, 'r') as f:

        p = (f['surface_pressure'][...] /
             f['surface_pressure'].attrs['psm_scale_factor'])
        albedo = (f['surface_albedo'][...] /
                  f['surface_albedo'].attrs['psm_scale_factor'])
        aod = (f['aod'][...] / f['aod'].attrs['psm_scale_factor'])
        ssa = (f['ssa'][...] / f['ssa'].attrs['psm_scale_factor'])
        g = (f['asymmetry'][...] / f['asymmetry'].attrs['psm_scale_factor'])
        z = (f['solar_zenith_angle'][...] /
             f['solar_zenith_angle'].attrs['psm_scale_factor'])
        alpha = (f['alpha'][...] / f['alpha'].attrs['psm_scale_factor'])
        ozone = (f['ozone'][...] / f['ozone'].attrs['psm_scale_factor'])
        w = (f['total_precipitable_water'][...] /
             f['total_precipitable_water'].attrs['psm_scale_factor'])

        ti = pd.to_datetime(f['time_index'][...].astype(str))

    radius = ti_to_radius(ti, n_cols=p.shape[1])
    beta = calc_beta(aod, alpha)

    t0 = time.time()
    rest_data = rest2(p, albedo, ssa, g, z, radius, alpha, beta, ozone, w)
    print('Testing rest2 on data shape {0} took {1:.1f} seconds.'
          .format(p.shape, time.time() - t0))

    check_vars = ('dni', 'dhi', 'ghi', 'Ruuclr', 'Tddclr', 'Tduclr')

    for var in check_vars:
        data = getattr(rest_data, var)[:, 0:10]

        csv = "./data/rest2/rest2_{}.csv".format(var)

        if os.path.exists(csv):
            baseline = np.genfromtxt(csv, delimiter=',')
            result = np.allclose(data, baseline, rtol=RTOL, atol=ATOL)
            assert result, 'REST2 test failed for "{}"'.format(var)

        else:
            result = False
            np.savetxt(csv, data, delimiter=",")
            warn('Baseline REST2 outputs for "{}" did not exist. '
                 'Test failed. Printed new outputs to: {}'
                 .format(var, csv))
    assert result


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
