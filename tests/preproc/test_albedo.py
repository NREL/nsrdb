# -*- coding: utf-8 -*-
"""
PyTest file for composite albedo creation. Test date created using Jupyter
Notebook at tests/data/albedo/Test data creation.ipynb

Created on Jan 23th 2020

@author: mbannist
"""

import logging
import os
import tempfile
from datetime import datetime as dt

import h5py
import numpy as np
import pytest

from nsrdb import TESTDATADIR
from nsrdb.albedo import albedo
from nsrdb.albedo import temperature_model as tm
from nsrdb.utilities.pytest import execute_pytest

pytest.importorskip('pyhdf')

ALBEDOTESTDATADIR = os.path.join(TESTDATADIR, 'albedo')
MERRATESTDATADIR = os.path.join(TESTDATADIR, 'merra2_source_files')


logger = logging.getLogger()


def test_merra_grid_mapping():
    """Make sure lat/lon correspond to the correct
    mask grid cell in the merra data used for albedo
    calculations
    """

    d = dt(2013, 1, 1)
    with tempfile.TemporaryDirectory() as td:
        cad = albedo.CompositeAlbedoDay.run(
            d,
            ALBEDOTESTDATADIR,
            ALBEDOTESTDATADIR,
            td,
            MERRATESTDATADIR,
            ims_shape=(32, 25),
            modis_shape=(122, 120),
        )
    grid = tm.DataHandler.get_grid(cad._modis.lat, cad._modis.lon, cad._mask)

    cad_grid = np.zeros((len(cad._modis.lat), len(cad._modis.lon), 2))

    for i, lat in enumerate(cad._modis.lat):
        for j, lon in enumerate(cad._modis.lon):
            cad_grid[i, j, 0] = lat
            cad_grid[i, j, 1] = lon

    lats = cad_grid[:, :, 0][cad._mask == 1].reshape(-1)
    lons = cad_grid[:, :, 1][cad._mask == 1].reshape(-1)

    assert np.array_equal(lats, np.array(grid['latitude']))
    assert np.array_equal(lons, np.array(grid['longitude']))


def test_increasing_temp_decreasing_albedo():
    """Check that albedo increases with
    decreasing temp"""

    d = dt(2013, 1, 1)
    with tempfile.TemporaryDirectory() as td:
        cad = albedo.CompositeAlbedoDay.run(
            d,
            ALBEDOTESTDATADIR,
            ALBEDOTESTDATADIR,
            td,
            MERRATESTDATADIR,
            ims_shape=(32, 25),
            modis_shape=(122, 120),
        )

        T = cad._merra_data
        T.sort()
        albedo_array = tm.TemperatureModel.get_snow_albedo(T)
        albedo_array_sorted = albedo_array.copy()
        albedo_array_sorted.sort()
        albedo_array_sorted = albedo_array_sorted[::-1]
        assert np.array_equal(albedo_array, albedo_array_sorted)


def test_4km_data_with_temp_model():
    """Create composite albedo data with temperature dependent
    albedo model using 4km IMS"""
    test_file = os.path.join(ALBEDOTESTDATADIR, 'nsrdb_albedo_2013_001.h5')
    with h5py.File(test_file, 'r') as f:
        data = np.array(f['surface_albedo'])

    d = dt(2013, 1, 1)
    with tempfile.TemporaryDirectory() as td:
        cad = albedo.CompositeAlbedoDay.run(
            d,
            ALBEDOTESTDATADIR,
            ALBEDOTESTDATADIR,
            td,
            MERRATESTDATADIR,
            ims_shape=(32, 25),
            modis_shape=(122, 120),
        )

        assert np.array_equal(data[cad._mask == 0], cad.albedo[cad._mask == 0])

        assert not np.array_equal(
            data[cad._mask == 1], cad.albedo[cad._mask == 1]
        )

        cad.write_albedo()
        new_albedo_file = os.path.join(td, 'nsrdb_albedo_2013_001.h5')
        with h5py.File(new_albedo_file, 'r') as f:
            new_data = np.array(f['surface_albedo'])
        assert np.array_equal(data[cad._mask == 0], new_data[cad._mask == 0])


def test_4km_data():
    """Create composite albedo data using 4km IMS"""
    test_file = os.path.join(ALBEDOTESTDATADIR, 'nsrdb_albedo_2013_001.h5')
    with h5py.File(test_file, 'r') as f:
        data = np.array(f['surface_albedo'])

    d = dt(2013, 1, 1)
    with tempfile.TemporaryDirectory() as td:
        cad = albedo.CompositeAlbedoDay.run(
            d,
            ALBEDOTESTDATADIR,
            ALBEDOTESTDATADIR,
            td,
            ims_shape=(32, 25),
            modis_shape=(122, 120),
        )

        assert np.array_equal(data, cad.albedo)

        cad.write_albedo()
        new_albedo_file = os.path.join(td, 'nsrdb_albedo_2013_001.h5')
        with h5py.File(new_albedo_file, 'r') as f:
            new_data = np.array(f['surface_albedo'])
        assert np.array_equal(data, new_data)


def test_1km_data():
    """Create composite albedo data using 1km IMS"""
    test_file = os.path.join(ALBEDOTESTDATADIR, 'nsrdb_albedo_2015_001.h5')
    with h5py.File(test_file, 'r') as f:
        data = np.array(f['surface_albedo'])

    d = dt(2015, 1, 1)
    with tempfile.TemporaryDirectory() as td:
        cad = albedo.CompositeAlbedoDay.run(
            d,
            ALBEDOTESTDATADIR,
            ALBEDOTESTDATADIR,
            td,
            ims_shape=(64, 50),
            modis_shape=(60, 61),
        )

        assert np.array_equal(data, cad.albedo)

        cad.write_albedo()
        new_albedo_file = os.path.join(td, 'nsrdb_albedo_2015_001.h5')
        with h5py.File(new_albedo_file, 'r') as f:
            new_data = np.array(f['surface_albedo'])
        assert np.array_equal(data, new_data)


def test_single_worker():
    """Create composite albedo data using 4km IMS and one worker"""
    test_file = os.path.join(ALBEDOTESTDATADIR, 'nsrdb_albedo_2013_001.h5')
    with h5py.File(test_file, 'r') as f:
        data = np.array(f['surface_albedo'])

    d = dt(2013, 1, 1)
    with tempfile.TemporaryDirectory() as td:
        cad = albedo.CompositeAlbedoDay.run(
            d,
            ALBEDOTESTDATADIR,
            ALBEDOTESTDATADIR,
            td,
            ims_shape=(32, 25),
            modis_shape=(122, 120),
            max_workers=1,
        )

        assert np.array_equal(data, cad.albedo)

        cad.write_albedo()
        new_albedo_file = os.path.join(td, 'nsrdb_albedo_2013_001.h5')
        with h5py.File(new_albedo_file, 'r') as f:
            new_data = np.array(f['surface_albedo'])
        assert np.array_equal(data, new_data)


def test_five_workers():
    """Create composite albedo data using 4km IMS and an arbitrary number
    of workers (5).
    """
    test_file = os.path.join(ALBEDOTESTDATADIR, 'nsrdb_albedo_2013_001.h5')
    with h5py.File(test_file, 'r') as f:
        data = np.array(f['surface_albedo'])

    d = dt(2013, 1, 1)
    with tempfile.TemporaryDirectory() as td:
        cad = albedo.CompositeAlbedoDay.run(
            d,
            ALBEDOTESTDATADIR,
            ALBEDOTESTDATADIR,
            td,
            ims_shape=(32, 25),
            modis_shape=(122, 120),
            max_workers=5,
        )

        assert np.array_equal(data, cad.albedo)

        cad.write_albedo()
        new_albedo_file = os.path.join(td, 'nsrdb_albedo_2013_001.h5')
        with h5py.File(new_albedo_file, 'r') as f:
            new_data = np.array(f['surface_albedo'])
        assert np.array_equal(data, new_data)


if __name__ == '__main__':
    execute_pytest(__file__)
