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
from nsrdb.daily_merra import MerraDay
from nsrdb.utilities.loggers import init_logger


RTOL = 0.001
ATOL = 0.001


@pytest.mark.parametrize('var',
                         ('surface_pressure',
                          'air_temperature',
                          'ozone',
                          'total_precipitable_water',
                          'wind_speed',
                          'specific_humidity',
                          'alpha',
                          'aod',
                          'ssa',
                          ))
def test_daily_merra2(var):
    """Test MERRA processed variables against baseline data."""
    init_logger(__name__, log_file=None, log_level='DEBUG')

    out_dir = os.path.join(TESTDATADIR, 'merra2/')
    merra_dir = os.path.join(TESTDATADIR, 'merra2_source_files/')
    var_meta = os.path.join(CONFIGDIR, 'nsrdb_vars.csv')
    date = datetime.date(year=2017, month=1, day=1)
    grid = os.path.join(TESTDATADIR, 'reference_grids/', 'west_psm_extent.csv')

    merra = MerraDay.run(var_meta, date, merra_dir, grid,
                         var_list=[var])

    baseline_path = os.path.join(out_dir, var + '.h5')
    if not os.path.exists(baseline_path):
        with h5py.File(baseline_path, 'w') as f:
            f.create_dataset(var, data=merra.nsrdb_data[var])
            f.create_dataset('meta', data=merra.nsrdb_grid.values,
                             dtype=merra.nsrdb_grid.values.dtype)
        assert False, 'Output file for "{}" did not exist, created'.format(var)
    else:
        with h5py.File(baseline_path, 'r') as f:
            data_baseline = f[var][...]
        assert np.allclose(data_baseline, merra.nsrdb_data[var],
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
