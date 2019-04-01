# pylint: skip-file
"""
PyTest file for PV generation in Rhode Island.

Created on Thu Nov 29 09:54:51 2018

@author: gbuster
"""

import os
import pytest
import numpy as np
import h5py
import datetime

from nsrdb import TESTDATADIR, CONFIGDIR
from nsrdb.ancillary import Ancillary
from nsrdb.utilities.loggers import init_logger


RTOL = 0.001
ATOL = 0.001


@pytest.mark.parametrize('var',
                         ('surface_pressure',
                          'asymmetry',
                          'air_temperature',
                          'ozone',
                          'total_precipitable_water',
                          'wind_speed',
                          'specific_humidity',
                          'alpha',
                          'aod',
                          'ssa',
                          'solar_zenith_angle',
                          'dew_point',
                          ))
def test_ancillary_single(var):
    """Test MERRA processed variables against baseline data."""
    init_logger('nsrdb.ancillary', log_file=None, log_level='DEBUG')

    out_dir = os.path.join(TESTDATADIR, 'processed_ancillary/')
    merra_dir = os.path.join(TESTDATADIR, 'merra2_source_files/')
    var_meta = os.path.join(CONFIGDIR, 'nsrdb_vars.csv')
    date = datetime.date(year=2017, month=1, day=1)
    grid = os.path.join(TESTDATADIR, 'reference_grids/', 'west_psm_extent.csv')

    data = Ancillary.process_single(var, var_meta, date, merra_dir, grid)

    baseline_path = os.path.join(out_dir, var + '.h5')
    if not os.path.exists(baseline_path):
        with h5py.File(baseline_path, 'w') as f:
            f.create_dataset(var, data=data)
        msg = 'Output file for "{}" did not exist, created'.format(var)
        assert False, msg
    else:
        with h5py.File(baseline_path, 'r') as f:
            data_baseline = f[var][...]
        assert np.allclose(data_baseline, data,
                           atol=ATOL, rtol=RTOL)


def test_parallel(var_list=('surface_pressure', 'air_temperature',
                            'specific_humidity', 'relative_humidity')):
    """Test the ancillary variable parallel processing with derived variable
    dependency."""
    init_logger('nsrdb.ancillary', log_file=None, log_level='DEBUG')

    out_dir = os.path.join(TESTDATADIR, 'processed_ancillary/')
    merra_dir = os.path.join(TESTDATADIR, 'merra2_source_files/')
    var_meta = os.path.join(CONFIGDIR, 'nsrdb_vars.csv')
    date = datetime.date(year=2017, month=1, day=1)
    grid = os.path.join(TESTDATADIR, 'reference_grids/', 'west_psm_extent.csv')

    data = Ancillary.process_multiple(var_list, var_meta, date, merra_dir,
                                      grid, parallel=True)

    for key, value in data.items():

        baseline_path = os.path.join(out_dir, key + '.h5')
        if not os.path.exists(baseline_path):
            with h5py.File(baseline_path, 'w') as f:
                f.create_dataset(key, data=value)
            msg = 'Output file for "{}" did not exist, created'.format(key)
            assert False, msg
        else:
            with h5py.File(baseline_path, 'r') as f:
                data_baseline = f[key][...]
            assert np.allclose(data_baseline, value,
                               atol=ATOL, rtol=RTOL)


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
