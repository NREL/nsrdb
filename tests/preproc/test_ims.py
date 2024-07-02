# -*- coding: utf-8 -*-
"""
PyTest file for IMS snow data processing

Created on Jan 17th 2020

@author: mbannist
"""

import os
import tempfile
from datetime import datetime as dt

import pytest

from nsrdb import TESTDATADIR
from nsrdb.albedo import ims
from nsrdb.albedo.ims import get_dt
from nsrdb.utilities.pytest import execute_pytest

IMSTESTDATADIR = os.path.join(TESTDATADIR, 'albedo')

METAFILES = [
    'IMS1kmLats.24576x24576x1.double',
    'IMS1kmLons.24576x24576x1.double',
    'imslat_4km.bin',
    'imslon_4km.bin',
]


def test_too_early_date():
    """Try a day before data is available"""
    d = dt(1997, 2, 3)
    with pytest.raises(ims.ImsError):
        ims.ImsDay(d, '.')


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
    """Verify correct resolution is selected by date"""
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

    d = get_dt(2004, 54)
    ifa = ims.ImsFileAcquisition(d, '.')
    assert ifa.res == '24km'


def test_missing_data():
    """
    Verify appropriate exception is raised when a missing date is requested.
    """
    with (
        tempfile.TemporaryDirectory() as td,
        pytest.raises(ims.ImsDataNotFoundError),
    ):
        for mf in METAFILES:
            with open(os.path.join(td, mf), 'w', encoding='utf-8') as f:
                f.write('fake metafile data')

        d = get_dt(2015, 108)
        ifa = ims.ImsFileAcquisition(d, td)
        ifa.get_file()


def test_data_loading():
    """Test data loading"""
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


if __name__ == '__main__':
    execute_pytest(__file__)
