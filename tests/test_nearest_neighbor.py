"""
PyTest file for PV generation in Rhode Island.

Created on Thu Nov 29 09:54:51 2018

@author: gbuster
"""

import numpy as np
import pandas as pd

from nsrdb.utilities.nearest_neighbor import geo_nn, knn, reg_grid_nn
from nsrdb.utilities.pytest import execute_pytest

latitude = np.array(range(0, 90))
longitude = np.array(range(-180, 0))

lon_mesh, lat_mesh = np.meshgrid(longitude, latitude)
lon_mesh = lon_mesh.flatten()
lat_mesh = lat_mesh.flatten()
source_meta = pd.DataFrame({'latitude': lat_mesh, 'longitude': lon_mesh})

nsrdb_meta = pd.DataFrame({
    'latitude': [1.1, 12.8, 23.3, 43.4, 44.2],
    'longitude': [-120.4, -112.3, -101.1, -91.3, -88.4],
})

baseline4 = np.array([
    [[1, -120], [1, -121], [2, -120], [2, -121]],
    [[13, -112], [13, -113], [12, -112], [12, -113]],
    [[23, -101], [24, -101], [23, -102], [23, -100]],
    [[43, -91], [44, -91], [43, -92], [44, -92]],
    [[44, -88], [44, -89], [45, -88], [45, -89]],
])

# calculated using Vincenty: https://www.cqsrg.org/tools/GCDistance/
baseline_dist = np.array([
    45.873,
    39.365,
    34.766,
    50.688,
    38.983,
])


def test_geo_haversine_nn():
    """Test the geographic haversine nearest neighbor."""

    _, ind = geo_nn(source_meta, nsrdb_meta, k=4)
    coords_closest = source_meta.values[ind]
    result = np.allclose(baseline4, coords_closest)
    msg = 'Haversine NN failed!'
    assert result, msg


def test_geo_haversine_dist():
    """Test the geographic haversine nearest neighbor."""

    dist, _ = geo_nn(source_meta, nsrdb_meta, k=1)
    dist = dist.flatten()
    result = np.allclose(baseline_dist, dist, rtol=0.01, atol=0.0)
    diff = np.abs(baseline_dist - dist)
    msg = 'Haversine distance failed! ' '\nDist: \n{}\nDiff:\n{}'.format(
        dist, diff
    )
    assert result, msg


def test_knn():
    """Test the k - nearest neighbor."""

    ind = knn(source_meta, nsrdb_meta, k=4)[1]
    coords_closest = source_meta.values[ind]
    result = np.allclose(baseline4, coords_closest)
    msg = 'KNN failed!'
    assert result, msg


def test_regular_grid_nn():
    """Test a nearest neighbor lookup on a regular grid and compare to knn."""
    ind = reg_grid_nn(latitude, longitude, nsrdb_meta)
    coords_closest_reg = source_meta.values[ind]
    ind = knn(source_meta, nsrdb_meta, k=1)[1].flatten()
    coords_closest_knn = source_meta.values[ind]
    msg = 'Regular grid NN failed!'
    assert np.allclose(coords_closest_knn, coords_closest_reg), msg


if __name__ == '__main__':
    execute_pytest(__file__)
