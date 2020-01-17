# -*- coding: utf-8 -*-
"""
PyTest file for multi year mean.

Created on Jan 17th 2020

@author: mbannist
"""
import os
import pytest
# from nsrdb import TESTDATADIR
import nsrdb.albedo.modis as modis


def test_day_mapping():
    mfa = modis.ModisFileAcquisition
    assert mfa._nearest_modis_day(1) == 1
    assert mfa._nearest_modis_day(2) == 1
    assert mfa._nearest_modis_day(3) == 1
    assert mfa._nearest_modis_day(4) == 1
    assert mfa._nearest_modis_day(5) == 9
    assert mfa._nearest_modis_day(6) == 9
    assert mfa._nearest_modis_day(7) == 9
    assert mfa._nearest_modis_day(8) == 9
    assert mfa._nearest_modis_day(9) == 9
    assert mfa._nearest_modis_day(10) == 9
    assert mfa._nearest_modis_day(12) == 9
    assert mfa._nearest_modis_day(13) == 17
    assert mfa._nearest_modis_day(244) == 241
    assert mfa._nearest_modis_day(361) == 361
    assert mfa._nearest_modis_day(365) == 361
    assert mfa._nearest_modis_day(366) == 361


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
