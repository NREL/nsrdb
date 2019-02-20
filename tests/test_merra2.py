# pylint: skip-file
"""
PyTest file for PV generation in Rhode Island.

Created on Thu Nov 29 09:54:51 2018

@author: gbuster
"""

from configobj import ConfigObj
import os
import pytest
import numpy as np

from nsrdb.merra2 import run_single


VARS = {'PS': 'surface_pressure',
        'T2M': 'air_temperature',
        'QV2M': 'specific_humidity',
        'TO3': 'ozone',
        'TQV': 'total_precipitable_water',
        'wind_speed': 'wind_speed',
        'wind_direction': 'wind_direction',
        'relative_humidity': 'relative_humidity',
        'dew_point': 'dew_point',
        'TOTANGSTR': 'alpha',
        'TOTEXTTAU': 'aod',
        'TOTSCATAU': 'ssa',
        'asymmetry': 'asymmetry',
        'surface_albedo': 'surface_albedo',
        }

TESTDIR = os.path.dirname(os.path.realpath(__file__))
CONFIG_FILE = os.path.join(TESTDIR, 'data', 'config', 'test_config.ini')
RTOL = 0.0
ATOL = 0.04


def purge_dir(d, tag=''):
    """Remove all files in directory d."""
    flist = os.listdir(d)
    for f in flist:
        if tag in f:
            try:
                os.remove(os.path.join(d, f))
            except Exception as e:
                pass


def benchmark_summaries(var,
                        d_baseline=os.path.join(TESTDIR, 'data', 'baseline'),
                        d_test=os.path.join(TESTDIR, 'data', 'output',
                                            'striped')):
    """Benchmark csv summary files in test dir against baseline dir."""
    flist = os.listdir(d_test)
    for f in flist:
        if f.endswith('.csv') and var in f:
            test_data = np.fromfile(os.path.join(d_test, f), sep=',')
            baseline = np.fromfile(os.path.join(d_baseline, f), sep=',')
            result = np.allclose(test_data, baseline, rtol=RTOL, atol=ATOL)
            return result
    return False


@pytest.mark.parametrize('var', list(VARS.keys()))
def test_merra2(var):
    """Test reV 2.0 generation for PV and benchmark against reV 1.0 results."""

    # fixed inputs
    date_range = (20170101, 20170102)
    region = 'west'

    # run and benchmark
    run_single(CONFIG_FILE, date_range, region, var, cores=1)
    result = benchmark_summaries(VARS[var])
    assert result


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

    if purge:
        config = ConfigObj(CONFIG_FILE, unrepr=True)
        for d in (config['ancillary']['out_dir'],
                  config['ancillary']['temp_dir']):
            purge_dir(d)


if __name__ == '__main__':
    execute_pytest()
