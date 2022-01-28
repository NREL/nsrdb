# -*- coding: utf-8 -*-
"""
PyTest file for IMS snow data processing

Created on Jan 17th 2020

@author: mbannist
"""
import os
import pytest
import nsrdb.albedo.ims as ims
from nsrdb.albedo.ims import get_dt
import tempfile
from datetime import datetime as dt

from nsrdb import TESTDATADIR
IMSTESTDATADIR = os.path.join(TESTDATADIR, 'albedo')

METAFILES = ['IMS1kmLats.24576x24576x1.double',
             'IMS1kmLons.24576x24576x1.double',
             'imslat_4km.bin', 'imslon_4km.bin']


def test_gap_fill_date():
    """
    Verify gap fill code finds the correct nearest day for days with
    missing data.
    """
    # Missing days: 2014 - 293, 294, 295
    missing = get_dt(2014, 293)
    igf = ims.ImsGapFill(missing, '_', '_')
    assert igf._closest_day() == get_dt(2014, 292)

    missing = get_dt(2014, 295)
    igf = ims.ImsGapFill(missing, '_', '_')
    assert igf._closest_day() == get_dt(2014, 296)

    # For a tie, defaults to earlier day
    missing = get_dt(2014, 294)
    igf = ims.ImsGapFill(missing, '_', '_')
    assert igf._closest_day() == get_dt(2014, 292)

    # Test searching for data with inadequate search range
    missing = get_dt(2014, 294)
    igf = ims.ImsGapFill(missing, '_', '_', search_range=1)
    with pytest.raises(ims.ImsError):
        _ = igf._closest_day()


def test_too_early_date():
    """ Try a day before data is available """
    d = dt(1997, 2, 3)
    with pytest.raises(ims.ImsError):
        ims.ImsDay(d, '.')


def test_version_1_3_date_shift():
    """
    For data starting on 2014, 336, the file is dated one day after the data!!
    """
    d = get_dt(2014, 335)
    ifa = ims.ImsRealFileAcquisition(d, '.')
    assert ifa._pfilename == 'ims2014335_4km_v1.2.asc'

    d = get_dt(2014, 336)
    ifa = ims.ImsRealFileAcquisition(d, '.')
    assert ifa._pfilename == 'ims2014337_1km_v1.3.asc'

    d = get_dt(2014, 365)
    ifa = ims.ImsRealFileAcquisition(d, '.')
    assert ifa._pfilename == 'ims2015001_1km_v1.3.asc'


def test_ims_res():
    """ Verify correct resolution is selected by date """
    d = get_dt(2014, 336)
    ifa = ims.ImsRealFileAcquisition(d, '.')
    assert ifa.res == '1km'

    d = get_dt(2014, 335)
    ifa = ims.ImsRealFileAcquisition(d, '.')
    assert ifa.res == '4km'

    d = dt(2025, 1, 1)
    ifa = ims.ImsRealFileAcquisition(d, '.')
    assert ifa.res == '1km'

    d = dt(2006, 1, 1)
    ifa = ims.ImsRealFileAcquisition(d, '.')
    assert ifa.res == '4km'

    d = get_dt(2004, 54)
    ifa = ims.ImsRealFileAcquisition(d, '.')
    assert ifa.res == '24km'


def test_missing_data():
    """
    Verify appropriate exception is raised when a missing date is requested.
    """
    with tempfile.TemporaryDirectory() as td:
        with pytest.raises(ims.ImsDataNotFound):
            for mf in METAFILES:
                with open(os.path.join(td, mf),
                          'wt') as f:
                    f.write('fake metafile data')

            d = get_dt(2015, 108)
            ifa = ims.ImsRealFileAcquisition(d, td)
            ifa.get_files()


def test_data_loading():
    """ Test data loading """
    d = get_dt(2015, 1)
    ims_day = ims.ImsDay(d, IMSTESTDATADIR, shape=(64, 50))
    assert ims_day.data.shape == (64, 50)
    assert ims_day.lon.shape == (64 * 50,)
    assert ims_day.lat.shape == (64 * 50,)

    d = get_dt(2013, 1)
    ims_day = ims.ImsDay(d, IMSTESTDATADIR, shape=(32, 25))
    assert ims_day.data.shape == (32, 25)
    assert ims_day.lon.shape == (32 * 25,)
    assert ims_day.lat.shape == (32 * 25,)


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
