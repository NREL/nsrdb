# -*- coding: utf-8 -*-
"""
PyTest file for IMS snow data processing

Created on Jan 17th 2020

@author: mbannist
"""
import os
import pytest
# from nsrdb import TESTDATADIR
import nsrdb.albedo.ims as ims
from nsrdb.albedo.ims import get_dt
import tempfile
from datetime import datetime as dt


def test_early_date():
    d = dt(2004, 2, 21)
    with pytest.raises(ims.ImsError):
        ims.ImsFileAcquisition(d, '.')


def test_version_1_3_date_shift():
    """
    For data starting on 2014, 336, the file is dated one day after the data!!
    """
    d = get_dt(2014, 335)
    ifa = ims.ImsFileAcquisition(d, '.')
    assert ifa._pfilename == 'ims2014335_4km_v1.2.asc'

    d = get_dt(2014, 336)
    ifa = ims.ImsFileAcquisition(d, '.')
    assert ifa._pfilename == 'ims2014337_1km_v1.3.asc'

    d = get_dt(2014, 365)
    ifa = ims.ImsFileAcquisition(d, '.')
    assert ifa._pfilename == 'ims2015001_1km_v1.3.asc'


def test_ims_res():
    """ Verify correct resolution is selected by date """
    d = get_dt(2014, 336)
    ifa = ims.ImsFileAcquisition(d, '.')
    assert ifa.res == '1km'

    d = get_dt(2014, 335)
    ifa = ims.ImsFileAcquisition(d, '.')
    assert ifa.res == '4km'

    d = dt(2025, 1, 1)
    ifa = ims.ImsFileAcquisition(d, '.')
    assert ifa.res == '1km'

    d = dt(2006, 1, 1)
    ifa = ims.ImsFileAcquisition(d, '.')
    assert ifa.res == '4km'


def test_download():
    """
    For data on and after 2014, 336, the file is dated one day after the data!!

    Downloading meta data for 1km is slow (~4GB)
    """
    with tempfile.TemporaryDirectory() as td:
        # TODO - Remove next line
        td = 'scratch'

        d = get_dt(2005, 157)
        ifa = ims.ImsFileAcquisition(d, td)
        ifa.get_files()
        assert os.path.isfile(ifa.filename)
        assert ifa.filename.split('/')[-1] == 'ims2005157_4km_v1.2.asc'

        d = get_dt(2015, 157)
        ifa = ims.ImsFileAcquisition(d, td)
        ifa.get_files()
        assert os.path.isfile(ifa.filename)
        assert ifa.filename.split('/')[-1] == 'ims2015158_1km_v1.3.asc'

        # test downloading fake file
        ifa = ims.ImsFileAcquisition(d, td)
        ifa._pfilename = 'fake'
        with pytest.raises(ims.ImsError):
            ifa._download_data()


def test_data_loading():
    """
    For data on and after 2014, 336, the file is dated one day after the data!!

    Downloading meta data for 1km is slow (~4GB)
    """
    with tempfile.TemporaryDirectory() as td:
        # TODO - Remove next line
        td = 'scratch'
        d = get_dt(2005, 157)
        ims_day = ims.ImsDay(d, td)
        assert ims_day.data.shape == (ims_day.pixels['4km'],
                                      ims_day.pixels['4km'])
        assert ims_day.lon.shape == (ims_day.pixels['4km']**2,)
        assert ims_day.lat.shape == (ims_day.pixels['4km']**2,)

        d = get_dt(2015, 157)
        ims_day = ims.ImsDay(d, td)
        assert ims_day.data.shape == (ims_day.pixels['1km'],
                                      ims_day.pixels['1km'])
        assert ims_day.lon.shape == (ims_day.pixels['1km']**2,)
        assert ims_day.lat.shape == (ims_day.pixels['1km']**2,)


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
