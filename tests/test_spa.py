# pylint: skip-file
"""
PyTest file for solar position algorithms.

Created on Jun 4th 2019

@author: gbuster
"""
import pytest
import numpy as np
import os
from nsrdb.solar_position.spa import SPA
from nsrdb.solar_position.solpos import SolPos
from nsrdb import TESTDATADIR
from nsrdb.file_handlers.resource import Resource


def test_spa_solpo():
    """Test that SPA and SolPo match."""

    h5 = os.path.join(TESTDATADIR, 'validation_nsrdb', 'nsrdb_surfrad_2017.h5')

    with Resource(h5) as res:
        meta = res.meta
        ti = res.time_index
        pres = res['surface_pressure'] * 10
        temp = res['air_temperature']

    lat_lon = meta[['latitude', 'longitude']].values
    elev = meta['elevation'].values

    sza = SPA.zenith(ti, lat_lon, elev=elev)
    apparent_sza = SPA.apparent_zenith(ti, lat_lon,
                                       elev=elev,
                                       pressure=pres,
                                       temperature=temp)

    sza_solpos = SolPos(ti, lat_lon).zenith

    result = np.allclose(sza, sza_solpos, atol=1.5)
    assert result, 'SPA does not match SolPo.'
    result = np.allclose(sza, apparent_sza, rtol=1)
    assert result, 'Apparent SZA does not match.'


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