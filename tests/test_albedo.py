# -*- coding: utf-8 -*-
"""
PyTest file for composite albedo creation. Test date created using Jupyter
Notebook at tests/data/albedo/Test data creation.ipynb

Created on Jan 23th 2020

@author: mbannist
"""
import os
import pytest
# from nsrdb import TESTDATADIR
from datetime import datetime as dt
import numpy as np
import h5py

import nsrdb.albedo.albedo as albedo
# from nsrdb.albedo.ims import get_dt

BASE_DIR = os.path.dirname(__file__)
TEST_DATA_DIR = os.path.join(BASE_DIR, './data/albedo')


def test_4km_data():
    # Create composite albedo data for 4km IMS
    d = dt(2013, 1, 1)
    cad = albedo.CompositeAlbedoDay.run(d, TEST_DATA_DIR, TEST_DATA_DIR,
                                        '_', ims_shape=(32, 25),
                                        modis_shape=(122, 120))

    test_data = os.path.join(TEST_DATA_DIR, 'nsrdb_albedo_2013_001.h5')
    with h5py.File(test_data, 'r') as f:
        data = np.array(f['surface_albedo'])

    assert np.array_equal(data, cad.albedo)


def test_1km_data():
    # Create composite albedo data for 1km IMS
    d = dt(2015, 1, 1)
    cad = albedo.CompositeAlbedoDay.run(d, TEST_DATA_DIR, TEST_DATA_DIR,
                                        '_', ims_shape=(64, 50),
                                        modis_shape=(60, 61))

    test_data = os.path.join(TEST_DATA_DIR, 'nsrdb_albedo_2015_001.h5')
    with h5py.File(test_data, 'r') as f:
        data = np.array(f['surface_albedo'])

    assert np.array_equal(data, cad.albedo)


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
