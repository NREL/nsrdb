# -*- coding: utf-8 -*-
"""
PyTest file for rest2.

Created on Feb 13th 2019

@author: gbuster
"""

import os
import pytest
import pandas as pd
import numpy as np
from nsrdb import TESTDATADIR
from nsrdb.tmy.tmy import Cdf, Tmy


RTOL = 0.001
ATOL = 0.001
NSRDB_DIR = os.path.join(TESTDATADIR, 'validation_nsrdb/')
BASELINE_DIR = os.path.join(TESTDATADIR, 'tgy_2017/')

BASELINES_FILES = {0: 'DRA_36.62_-116.02_tgy.csv',
                   3: 'FPK_48.31_-105.1_tgy.csv',
                   5: 'SXF_43.73_-96.62_tgy.csv',
                   6: 'GCM_34.25_-89.87_tgy.csv',
                   7: 'BON_40.05_-88.37_tgy.csv',
                   8: 'PSU_40.72_-77.93_tgy.csv'}


def test_tmy():
    """Test TMY and validate against baseline data."""
    years = list(range(1998, 2018))
    weights = {'ghi': 1}
    tgy = Tmy(NSRDB_DIR, years, weights, site_slice=slice(0, 2))
    tgy_years = tgy.calculate_tmy()
    ghi = tgy._make_tmy_timeseries('ghi', tgy_years)

    for i, fn in BASELINES_FILES.items():
        df = pd.read_csv(os.path.join(BASELINE_DIR, fn))
        cols = [c.strip(' ').lower() for c in df.columns]
        df.columns = cols
        check = np.allclose(df['ghi'].values, ghi[:, i])
        print(fn)
        print(check)
        if not check:
            break


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
    # execute_pytest()
    test_tmy()

    years = list(range(1998, 2018))
    weights = {'ghi': 1}
    tgy = Tmy(NSRDB_DIR, years, weights)
    arr = tgy._get_my_arr('ghi')
    meta = tgy.meta

    cdf = Cdf(arr, tgy.my_time_index)
    fs_all = cdf._fs_all
    long_term_frac = cdf._lt_frac
    annaul_frac = cdf._annual_frac

    cdf.plot_tmy_selection(month=2, site=0,
                           plot_years=[1999, 2000, 2007, 2017])
