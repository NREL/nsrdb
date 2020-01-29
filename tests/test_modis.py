# -*- coding: utf-8 -*-
"""
PyTest file for MODIS dry land albedo data processing

Created on Jan 17th 2020

@author: mbannist
"""
import os
import pytest
# from nsrdb import TESTDATADIR
import nsrdb.albedo.modis as modis
from nsrdb.albedo.ims import get_dt
import tempfile
from datetime import datetime as dt


def test_last_year():
    """ Verify dates after most recently published data are handled """
    d1 = get_dt(2016, 233)
    d2 = get_dt(2015, 233)
    d3 = get_dt(2017, 110)  # Nearest available data is 113

    with tempfile.TemporaryDirectory() as td:
        mfa = modis.ModisFileAcquisition(d1, td)
        print(mfa.filename)
        assert mfa.filename == 'MCD43GF_wsa_shortwave_233_2015.hdf'

        mfa = modis.ModisFileAcquisition(d2, td)
        print(mfa.filename)
        assert mfa.filename == 'MCD43GF_wsa_shortwave_233_2015.hdf'

        mfa = modis.ModisFileAcquisition(d3, td)
        print(mfa.filename)
        assert mfa.filename == 'MCD43GF_wsa_shortwave_113_2015.hdf'


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


def test_download_and_data_loading():
    with tempfile.TemporaryDirectory() as td:
        # Test old file that is not supported
        d = dt(2002, 5, 15)
        with pytest.raises(modis.ModisError):
            md = modis.ModisDay(d, td)

        # Test downlaod
        d = dt(2015, 6, 6)
        filename = modis.ModisFileAcquisition.get_file(d, td)
        assert os.path.isfile(filename)

        # test downloading fake file
        mfa = modis.ModisFileAcquisition(d, td)
        mfa.filename = 'fake'
        with pytest.raises(modis.ModisError):
            mfa._download()

        # Test data loading
        md = modis.ModisDay(d, td)
        assert md.data.shape == (21600, 43200)
        assert len(md.lon) == 43200
        assert len(md.lat) == 21600


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
