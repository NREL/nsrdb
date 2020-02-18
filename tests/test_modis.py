# -*- coding: utf-8 -*-
"""
PyTest file for MODIS dry land albedo data processing

Created on Jan 17th 2020

@author: mbannist
"""
import os
import pytest
import nsrdb.albedo.modis as modis
from nsrdb.albedo.ims import get_dt
import tempfile
from datetime import datetime as dt

from nsrdb import TESTDATADIR
MODISTESTDATADIR = os.path.join(TESTDATADIR, 'albedo')


def test_last_year():
    """ Verify dates after most recently published data are handled """
    d1 = get_dt(2016, 233)
    d2 = get_dt(modis.LAST_YEAR, 123)
    d3 = get_dt(modis.LAST_YEAR + 2, 110)  # Nearest available data is 113

    with tempfile.TemporaryDirectory() as td:
        mfa = modis.ModisFileAcquisition(d1, td)
        assert mfa.filename == 'MCD43GF_wsa_shortwave_233_2016_V006.hdf'

        mfa = modis.ModisFileAcquisition(d2, td)
        assert mfa.filename == 'MCD43GF_wsa_shortwave_123_' + \
                               f'{modis.LAST_YEAR}_V006.hdf'

        mfa = modis.ModisFileAcquisition(d3, td)
        assert mfa.filename == f'MCD43GF_wsa_shortwave_110_' + \
                               f'{modis.LAST_YEAR}_V006.hdf'


def test_not_implemented_dl():
    """ Test downloading exception """
    d = dt(2002, 5, 15)
    mfa = modis.ModisFileAcquisition(d, 'fake')
    with pytest.raises(NotImplementedError):
        mfa._download()


def test_data_loading():
    """ Test data loading """
    d = dt(2015, 1, 1)
    md = modis.ModisDay(d, MODISTESTDATADIR, shape=(60, 61))
    assert md.data.shape == (60, 61)
    assert len(md.lon) == 61
    assert len(md.lat) == 60

    d = dt(2013, 1, 1)
    md = modis.ModisDay(d, MODISTESTDATADIR, shape=(122, 120))
    assert md.data.shape == (122, 120)
    assert len(md.lon) == 120
    assert len(md.lat) == 122


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
