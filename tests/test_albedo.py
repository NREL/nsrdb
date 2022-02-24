# -*- coding: utf-8 -*-
"""
PyTest file for composite albedo creation. Test date created using Jupyter
Notebook at tests/data/albedo/Test data creation.ipynb

Created on Jan 23th 2020

@author: mbannist
"""
import os
import pytest
from datetime import datetime as dt
import numpy as np
import h5py
import tempfile
import logging

pytest.importorskip("pyhdf")

import nsrdb.albedo.albedo as albedo

from nsrdb import TESTDATADIR
ALBEDOTESTDATADIR = os.path.join(TESTDATADIR, 'albedo')
MERRATESTDATADIR = os.path.join(TESTDATADIR, 'merra2_source_files')


logger = logging.getLogger()


def test_4km_data():
    """ Create composite albedo data using 4km IMS """
    test_file = os.path.join(ALBEDOTESTDATADIR, 'nsrdb_albedo_2013_001.h5')
    with h5py.File(test_file, 'r') as f:
        data = np.array(f['surface_albedo'])

    d = dt(2013, 1, 1)
    with tempfile.TemporaryDirectory() as td:
        cad = albedo.CompositeAlbedoDay.run(d, ALBEDOTESTDATADIR,
                                            ALBEDOTESTDATADIR,
                                            td,
                                            ims_shape=(32, 25),
                                            modis_shape=(122, 120))

        assert np.array_equal(data, cad.albedo)

        cad.write_albedo()
        new_albedo_file = os.path.join(td, 'nsrdb_albedo_2013_001.h5')
        with h5py.File(new_albedo_file, 'r') as f:
            new_data = np.array(f['surface_albedo'])
        assert np.array_equal(data, new_data)


def test_1km_data():
    """ Create composite albedo data using 1km IMS """
    test_file = os.path.join(ALBEDOTESTDATADIR, 'nsrdb_albedo_2015_001.h5')
    with h5py.File(test_file, 'r') as f:
        data = np.array(f['surface_albedo'])

    d = dt(2015, 1, 1)
    with tempfile.TemporaryDirectory() as td:
        cad = albedo.CompositeAlbedoDay.run(d, ALBEDOTESTDATADIR,
                                            ALBEDOTESTDATADIR,
                                            td,
                                            ims_shape=(64, 50),
                                            modis_shape=(60, 61))

        assert np.array_equal(data, cad.albedo)

        cad.write_albedo()
        new_albedo_file = os.path.join(td, 'nsrdb_albedo_2015_001.h5')
        with h5py.File(new_albedo_file, 'r') as f:
            new_data = np.array(f['surface_albedo'])
        assert np.array_equal(data, new_data)


def test_single_worker():
    """ Create composite albedo data using 4km IMS and one worker"""
    test_file = os.path.join(ALBEDOTESTDATADIR, 'nsrdb_albedo_2013_001.h5')
    with h5py.File(test_file, 'r') as f:
        data = np.array(f['surface_albedo'])

    d = dt(2013, 1, 1)
    with tempfile.TemporaryDirectory() as td:
        cad = albedo.CompositeAlbedoDay.run(d, ALBEDOTESTDATADIR,
                                            ALBEDOTESTDATADIR,
                                            td,
                                            ims_shape=(32, 25),
                                            modis_shape=(122, 120),
                                            max_workers=1)

        assert np.array_equal(data, cad.albedo)

        cad.write_albedo()
        new_albedo_file = os.path.join(td, 'nsrdb_albedo_2013_001.h5')
        with h5py.File(new_albedo_file, 'r') as f:
            new_data = np.array(f['surface_albedo'])
        assert np.array_equal(data, new_data)


def test_five_workers():
    """ Create composite albedo data using 4km IMS and an arbitrary number
    of workers (5).
    """
    test_file = os.path.join(ALBEDOTESTDATADIR, 'nsrdb_albedo_2013_001.h5')
    with h5py.File(test_file, 'r') as f:
        data = np.array(f['surface_albedo'])

    d = dt(2013, 1, 1)
    with tempfile.TemporaryDirectory() as td:
        cad = albedo.CompositeAlbedoDay.run(d, ALBEDOTESTDATADIR,
                                            ALBEDOTESTDATADIR,
                                            td,
                                            ims_shape=(32, 25),
                                            modis_shape=(122, 120),
                                            max_workers=5)

        assert np.array_equal(data, cad.albedo)

        cad.write_albedo()
        new_albedo_file = os.path.join(td, 'nsrdb_albedo_2013_001.h5')
        with h5py.File(new_albedo_file, 'r') as f:
            new_data = np.array(f['surface_albedo'])
        assert np.array_equal(data, new_data)


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
