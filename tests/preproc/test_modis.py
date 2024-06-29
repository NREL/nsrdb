"""
PyTest file for MODIS dry land albedo data processing

Created on Jan 17th 2020

@author: mbannist
"""
import os
import tempfile
from datetime import datetime as dt

import pytest

from nsrdb.utilities.pytest import execute_pytest

pytest.importorskip("pyhdf")
from nsrdb import TESTDATADIR
from nsrdb.albedo import modis
from nsrdb.albedo.ims import get_dt

MODISTESTDATADIR = os.path.join(TESTDATADIR, 'albedo')


def test_first_year():
    """Verify dates before oldest available data are handled properly"""
    early_day = modis.FIRST_DAY - 1
    early_day_str = str(early_day).zfill(3)
    first_day_str = str(modis.FIRST_DAY).zfill(3)

    d1 = get_dt(modis.FIRST_YEAR, early_day)  # 62/2000
    d2 = get_dt(modis.FIRST_YEAR - 1, early_day)  # 62/1999
    d3 = get_dt(modis.FIRST_YEAR - 1, modis.FIRST_DAY)  # 63/1999
    d4 = get_dt(modis.FIRST_YEAR, modis.FIRST_DAY)  # 63/2000
    d5 = get_dt(modis.FIRST_YEAR, 100)  # 100/2000

    with tempfile.TemporaryDirectory() as td:
        # Grab 62/2001 for 62/2000
        mfa = modis.ModisFileAcquisition(d1, td)
        assert mfa.filename == ('MCD43GF_wsa_shortwave_{}_{}_V006.hdf'
                                ''.format(early_day_str, modis.FIRST_YEAR + 1))

        # Grab 62/2001 for 62/1999
        mfa = modis.ModisFileAcquisition(d2, td)
        assert mfa.filename == ('MCD43GF_wsa_shortwave_{}_{}_V006.hdf'
                                ''.format(early_day_str, modis.FIRST_YEAR + 1))

        # Grab 63/2000 for 63/1999
        mfa = modis.ModisFileAcquisition(d3, td)
        assert mfa.filename == ('MCD43GF_wsa_shortwave_{}_{}_V006.hdf'
                                ''.format(first_day_str, modis.FIRST_YEAR))

        # Grab 63/2000 for 63/2000
        mfa = modis.ModisFileAcquisition(d4, td)
        assert mfa.filename == ('MCD43GF_wsa_shortwave_{}_{}_V006.hdf'
                                ''.format(first_day_str, modis.FIRST_YEAR))
        # Grab 100/2000 for 100/2000
        mfa = modis.ModisFileAcquisition(d5, td)
        assert mfa.filename == ('MCD43GF_wsa_shortwave_{}_{}_V006.hdf'
                                ''.format(100, modis.FIRST_YEAR))


def test_last_year():
    """Verify dates after most recently published data are handled"""
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
        assert mfa.filename == 'MCD43GF_wsa_shortwave_110_' + \
                               f'{modis.LAST_YEAR}_V006.hdf'

        # Verify leap years after 2017 are handled properly
        mfa = modis.ModisFileAcquisition(get_dt(2020, 1), td)
        assert mfa.filename == 'MCD43GF_wsa_shortwave_001_' + \
                               f'{modis.LAST_YEAR}_V006.hdf'

        mfa = modis.ModisFileAcquisition(get_dt(2020, 365), td)
        assert mfa.filename == 'MCD43GF_wsa_shortwave_365_' + \
                               f'{modis.LAST_YEAR}_V006.hdf'

        mfa = modis.ModisFileAcquisition(get_dt(2020, 366), td)
        assert mfa.filename == 'MCD43GF_wsa_shortwave_365_' + \
                               f'{modis.LAST_YEAR}_V006.hdf'


def test_not_implemented_dl():
    """Test downloading exception"""
    d = dt(2002, 5, 15)
    mfa = modis.ModisFileAcquisition(d, 'fake')
    with pytest.raises(NotImplementedError):
        mfa._download()


def test_data_loading():
    """Test data loading"""
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


if __name__ == '__main__':
    execute_pytest(__file__)
