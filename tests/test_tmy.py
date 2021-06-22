# -*- coding: utf-8 -*-
"""
PyTest file for rest2.

Created on Feb 13th 2019

@author: gbuster
"""
from copy import deepcopy
import os
import pytest
import pandas as pd
import numpy as np
from nsrdb import TESTDATADIR
from nsrdb.tmy.tmy import Cdf, Tmy


RTOL = 0.001
ATOL = 0.001
NSRDB_BASE_FP = os.path.join(TESTDATADIR, 'validation_nsrdb/nsrdb_*_{}.h5')
BASELINE_DIR = os.path.join(TESTDATADIR, 'tgy_2017/')

BASELINES_FILES = {0: 'DRA_36.62_-116.02_tgy.csv',
                   3: 'FPK_48.31_-105.1_tgy.csv',
                   5: 'SXF_43.73_-96.62_tgy.csv',
                   6: 'GCM_34.25_-89.87_tgy.csv',
                   7: 'BON_40.05_-88.37_tgy.csv',
                   8: 'PSU_40.72_-77.93_tgy.csv'}


@pytest.mark.parametrize(('mults', 'best_year'), (((1.0, 1.1, 1.2), 2014),
                                                  ((0.5, 1.1, 0.6), 2015),
                                                  ((0.8, 1.1, 0.6), 2013)))
def test_cdf_best_year(mults, best_year):
    """Test the CDF year selection using an arbitrary input array"""

    my_arr = [(mults[i] * np.arange(365, dtype=np.float32)).tolist()
              for i in range(3)]
    my_arr = np.expand_dims(np.array(my_arr).flatten(), axis=1)
    time_index = pd.date_range(start='1-1-2013', end='1-1-2016',
                               freq='1D', closed='left')
    cdf = Cdf(my_arr, time_index)
    years, _ = cdf._best_fs_year()

    assert all(years == best_year)


def test_cdf_fs():
    """Test the FS metric against baseline values."""
    baseline = {1: np.array([0.22484936, 0.07167668, 0.18745799]),
                2: np.array([0.33341603, 0.1550684, 0.32166262]),
                3: np.array([0.33340802, 0.16681612, 0.33340802]),
                4: np.array([0.33341051, 0.1668211, 0.33341051]),
                5: np.array([0.33340802, 0.16681612, 0.33340802]),
                }

    mults = [0.4, 1.0, 1.6]
    my_arr = [(mults[i] * np.arange(17520, dtype=np.float32)).tolist()
              for i in range(3)]
    my_arr = np.expand_dims(np.array(my_arr).flatten(), axis=1)
    time_index = pd.date_range(start='1-1-2013', end='1-1-2016',
                               freq='30min', closed='left')
    cdf = Cdf(my_arr, time_index)
    fs_all = cdf._fs_all
    for k, v in baseline.items():
        assert np.allclose(fs_all[k].flatten(), v)


def test_fw_weighting():
    """Test the combination of different FS weightings."""
    dw = 0.3
    gw = 0.7
    years = list(range(1998, 2005))
    g_weights = {'sum_ghi': 1}
    d_weights = {'sum_dni': 1}
    m_weights = {'sum_dni': dw, 'sum_ghi': gw}
    tgy = Tmy(NSRDB_BASE_FP, years, g_weights, site_slice=slice(0, 1))
    tdy = Tmy(NSRDB_BASE_FP, years, d_weights, site_slice=slice(0, 1))
    tmy = Tmy(NSRDB_BASE_FP, years, m_weights, site_slice=slice(0, 1))
    g_ws = tgy.get_weighted_fs()
    d_ws = tdy.get_weighted_fs()
    m_ws = tmy.get_weighted_fs()

    for k, v in m_ws.items():
        assert np.allclose((gw * g_ws[k] + dw * d_ws[k]), v)


def test_arr_sampling():
    """Test the array retrieval and daily sampling methods."""
    years = list(range(1998, 2005))
    weights = {'sum_air_temperature': 1.0}
    tmy = Tmy(NSRDB_BASE_FP, years, weights, site_slice=slice(0, 1))
    subhourly_temp = tmy._get_my_arr('air_temperature')
    sum_temp = tmy._get_my_arr('sum_air_temperature')
    mean_temp = tmy._get_my_arr('mean_air_temperature')
    min_temp = tmy._get_my_arr('min_air_temperature')
    max_temp = tmy._get_my_arr('max_air_temperature')
    _ = tmy.my_daily_time_index
    ws = tmy.get_weighted_fs()

    assert len(tmy.my_time_index) == len(subhourly_temp)
    assert len(tmy.my_daily_time_index) == len(sum_temp)
    assert len(tmy.my_daily_time_index) == len(mean_temp)
    assert len(tmy.my_daily_time_index) == len(max_temp)
    assert len(tmy.my_daily_time_index) == len(min_temp)

    assert (sum_temp[(mean_temp > 0)] > mean_temp[(mean_temp > 0)]).all()
    assert (max_temp > mean_temp).all()
    assert (mean_temp > min_temp).all()

    cdf = Cdf(sum_temp, tmy.my_daily_time_index)
    fs = cdf._fs_all

    for k, v in fs.items():
        msg = 'Array sampling failed in the FS metric for site {}'.format(k)
        assert np.allclose(ws[k], v, rtol=0.02), msg


def test_run_counting():
    """Test the persistence run counter."""
    arr = np.array([0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
    m, n = Tmy._count_runs(arr)
    assert m == 3
    assert n == 5

    arr = np.array([0, 0, 0, 0])
    m, n = Tmy._count_runs(arr)
    assert m == 0
    assert n == 0


def test_tmy_steps():
    """Test each step of the TMY."""

    years = list(range(1998, 2018))
    weights = {'sum_ghi': 1}
    tgy = Tmy(NSRDB_BASE_FP, years, weights, site_slice=slice(0, 2))

    emsg = ('STEP1: TMY year selection based on FS metric failed! '
            'Low FS metrics do not correspond to selected years.')
    ws = tgy.get_weighted_fs()
    tmy_years_5_init, _ = tgy.select_fs_years(ws)
    for j in range(len(tgy.meta)):
        for m in range(1, 13):
            good_years = int(tgy.years[0]) + np.argsort(ws[m], axis=0)
            for yi in range(5):
                assert tmy_years_5_init[m][yi, j] == good_years[yi, j], emsg
                assert tmy_years_5_init[m][yi, j] == good_years[yi, j], emsg

    emsg = ('STEP2: Sorting based on multi-year ghi mean and median failed! '
            'TMY years do not appear to be sorted correctly.')
    tmy_years_5_sorted, diffs = tgy.sort_years_mm(deepcopy(tmy_years_5_init))
    for j in range(len(tgy.meta)):
        for m in range(1, 13):
            diffs_sub = []
            for yi in range(5):
                ii = tmy_years_5_sorted[m][yi, j] - tgy.years[0]
                diffs_sub.append(diffs[m][ii, j])
            assert diffs_sub == sorted(deepcopy(diffs_sub)), emsg

    emsg = ('STEP3: Persistence filter failed! TMY year was chosen with {}')
    emsg2 = 'Selected TMY year not found in 5 candidate years!'
    tmy_years, max_run_len, n_runs = tgy.persistence_filter(tmy_years_5_sorted)
    for im, m in enumerate(range(1, 13)):
        for j in range(len(tgy.meta)):
            year = tmy_years[im, j]
            tmy_year_index = np.where(tmy_years_5_sorted[m][:, j] == year)[0]

            assert tmy_year_index.size, emsg2
            tmy_year_index = tmy_year_index[0]

            max_run = max_run_len[m][j][tmy_year_index]
            n_run = n_runs[m][j][tmy_year_index]

            emsg1 = emsg.format('a max length run')
            assert max_run != max(max_run_len[m][j]), emsg1
            emsg1 = emsg.format('a max number of runs')
            assert n_run != max(n_runs[m][j])
            emsg1 = emsg.format('zero runs')
            assert n_run != 0


def test_baseline_timeseries():
    """Calculate the TMY timeseries and validate against a baseline file."""

    f_baseline = os.path.join(TESTDATADIR, 'tmy_baseline/tgy_1998_2017.csv')
    years = list(range(1998, 2018))
    weights = {'sum_ghi': 1}
    tgy = Tmy(NSRDB_BASE_FP, years, weights, site_slice=slice(0, 2))
    data = tgy.get_tmy_timeseries('ghi')

    df_data = pd.DataFrame(data, index=tgy.time_index)
    df_tmy_years = pd.DataFrame(tgy.tmy_years_long, index=tgy.time_index)

    assert df_tmy_years.iloc[0, 0] == 1999
    assert df_tmy_years.iloc[720, 0] == 1999
    assert df_tmy_years.iloc[1080, 0] == 2007
    assert df_tmy_years.iloc[8759, 0] == 2001

    assert df_tmy_years.iloc[0, 1] == 2011
    assert df_tmy_years.iloc[720, 1] == 2011
    assert df_tmy_years.iloc[1080, 1] == 2007
    assert df_tmy_years.iloc[8759, 1] == 2000

    if not os.path.exists(f_baseline):
        df_data.to_csv(f_baseline)
    else:
        df_baseline = pd.read_csv(f_baseline, index_col=0)
        assert np.allclose(df_data.values, df_baseline.values)


def plot_cdf():
    """Plot the CDF graph emulating the plot from the TMY users guide."""
    years = list(range(1998, 2018))
    weights = {'sum_ghi': 1}
    tgy = Tmy(NSRDB_BASE_FP, years, weights, site_slice=slice(0, 1))
    arr = tgy._get_my_arr('sum_ghi')
    cdf = Cdf(arr, tgy.my_daily_time_index)
    cdf.plot_tmy_selection()


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
