# -*- coding: utf-8 -*-
"""2018 aggregation run script - WRF for Jaemo and Rahul

Created on Tue Feb 11 14:04:45 2020

@author: gbuster
"""
import os
import shutil
import tempfile

import numpy as np
from rex import NSRDB
from scipy.stats import mode

from nsrdb import TESTDATADIR
from nsrdb.aggregation.aggregation import Aggregation, Manager
from nsrdb.utilities.pytest import execute_pytest

meta_dir = os.path.join(TESTDATADIR, 'meta/')

# i0 and i1 are the NN indices of the surfrad meta data, closest to
# the two test sites in test_meta_agg.csv
i0 = [2, 1, 8, 3]
i1 = [5, 0, 4, 7]

TESTJOB1 = {'source': {'data_sub_dir': 'nsrdb_2018',
                       'tree_file': 'kdtree_surfrad_meta.pkl',
                       'meta_file': 'surfrad_meta.csv',
                       'spatial': '2km',
                       'temporal': '5min'},
            'final': {'data_sub_dir': 'agg_out',
                      'fout': 'agg_out_test_2018.h5',
                      'tree_file': 'kdtree_test_meta_agg.pkl',
                      'meta_file': 'test_meta_agg.csv',
                      'spatial': '4km',
                      'temporal': '5min'},
            }

TESTJOB2 = {'source': {'data_sub_dir': 'nsrdb_2018',
                       'tree_file': 'kdtree_surfrad_meta.pkl',
                       'meta_file': 'surfrad_meta.csv',
                       'spatial': '2km',
                       'temporal': '5min'},
            'final': {'data_sub_dir': 'agg_out',
                      'fout': 'agg_out_test_2018.h5',
                      'tree_file': 'kdtree_test_meta_agg.pkl',
                      'meta_file': 'test_meta_agg.csv',
                      'spatial': '4km',
                      'temporal': '30min'},
            }

IGNORE_DSETS = ["alpha",
                "asymmetry",
                "ssa",
                "ozone",
                "surface_albedo",
                "surface_pressure",
                "total_precipitable_water",
                "cloud_press_acha",
                "solar_zenith_angle",
                "clearsky_dni",
                "clearsky_dhi",
                "clearsky_ghi",
                "ghi",
                "dhi",
                "air_temperature",
                "dew_point",
                "relative_humidity",
                "wind_direction",
                "wind_speed",
                "cld_reff_dcomp",
                ]

FP_IN_IRRAD = os.path.join(TESTDATADIR, TESTJOB1['source']['data_sub_dir'],
                           'nsrdb_irradiance_2018.h5')
FP_IN_ANCIL = os.path.join(TESTDATADIR, TESTJOB1['source']['data_sub_dir'],
                           'nsrdb_ancillary_2018.h5')
FP_IN_CLOUD = os.path.join(TESTDATADIR, TESTJOB1['source']['data_sub_dir'],
                           'nsrdb_clouds_2018.h5')


def copy_dir(src, dst):
    """
    Copy all files in src to dst
    """
    if not os.path.exists(dst):
        os.makedirs(dst)

    for file in os.listdir(src):
        shutil.copy(os.path.join(src, file), dst)


def test_rolling_window_index():
    """Test the rolling window index method for cloud property stuff."""

    out = Aggregation._get_rolling_window_index(L=10, w=3)
    assert all(out[0, :] == [0, 0, 1])
    assert all(out[1, :] == [0, 1, 2])
    assert all(out[9, :] == [8, 9, 9])

    out = Aggregation._get_rolling_window_index(L=10, w=1)
    assert out[0] == 0
    assert out[1] == 1
    assert out[9] == 9

    out = Aggregation._get_rolling_window_index(L=10, w=2)
    assert all(out[0] == [0, 1])
    assert all(out[1] == [1, 2])
    assert all(out[9] == [9, 9])


def test_spatial_agg():
    """Test spatial aggregation only (1-to-1 temporal agg)"""

    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(TESTDATADIR, TESTJOB1['source']['data_sub_dir'])
        dst = os.path.join(td, TESTJOB1['source']['data_sub_dir'])
        copy_dir(src, dst)

        Manager.run_chunk(TESTJOB1, td, meta_dir, 0, 1,
                          year=2018, parallel=False, log_level=False,
                          ignore_dsets=IGNORE_DSETS)

        fp_out = TESTJOB1['final']['fout'].replace('.h5', '_0.h5')
        fp_out = os.path.join(td, TESTJOB1['final']['data_sub_dir'], fp_out)
        with NSRDB(fp_out, mode='r') as f:
            dni = f['dni']
            aod = f['aod']
            ctype = f['cloud_type']

        with NSRDB(FP_IN_IRRAD, mode='r') as f:
            dni_in = f['dni']

        with NSRDB(FP_IN_ANCIL, mode='r') as f:
            aod_in = f['aod']

        with NSRDB(FP_IN_CLOUD, mode='r') as f:
            ctype_in = f['cloud_type']
            ctype_in[(ctype_in == 1)] = 0

        assert np.allclose(aod[:, 0], aod_in[:, i0[0]])
        assert np.allclose(aod[:, 1], aod_in[:, i1[0]])

        assert np.allclose(dni[:, 0], np.round(dni_in[:, i0].mean(axis=1)))
        assert np.allclose(dni[:, 1], np.round(dni_in[:, i1].mean(axis=1)))

        a = ctype[:, 0]
        b = mode(ctype_in[:, i0], axis=1)[0].flatten()
        assert np.array_equal(a, b)

        a = ctype[:, 1]
        b = mode(ctype_in[:, i1], axis=1)[0].flatten()
        assert np.array_equal(a, b)


def test_spatiotemporal_agg():
    """Test spatiotemporal aggregation"""
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(TESTDATADIR, TESTJOB2['source']['data_sub_dir'])
        dst = os.path.join(td, TESTJOB2['source']['data_sub_dir'])
        copy_dir(src, dst)

        Manager.run_chunk(TESTJOB2, td, meta_dir, 0, 1,
                          year=2018, parallel=False, log_level=False,
                          ignore_dsets=IGNORE_DSETS)

        fp_out = TESTJOB2['final']['fout'].replace('.h5', '_0.h5')
        fp_out = os.path.join(td, TESTJOB2['final']['data_sub_dir'], fp_out)
        with NSRDB(fp_out, mode='r') as f:
            ti = f.time_index
            dni = f['dni']
            aod = f['aod']
            ctype = f['cloud_type']
            opd = f['cld_opd_dcomp']

        with NSRDB(FP_IN_IRRAD, mode='r') as f:
            ti_in = f.time_index
            dni_in = f['dni']

        with NSRDB(FP_IN_ANCIL, mode='r') as f:
            aod_in = f['aod']

        with NSRDB(FP_IN_CLOUD, mode='r') as f:
            opd_in = f['cld_opd_dcomp']
            ctype_in = f['cloud_type']
            ctype_in[(ctype_in == 1)] = 0

        w = 7

        indices = Aggregation._get_rolling_window_index(len(dni_in), w=w)

        for i in [415, 420, 510, 520, 8760, 8770]:
            step = ti[i]
            i_in = np.where(ti_in == step)[0][0]
            iw = indices[i_in, :]

            truth = np.round(dni_in[iw, :][:, i0].mean())
            assert dni[i, 0] == truth
            truth = np.round(dni_in[iw, :][:, i1].mean())
            assert dni[i, 1] == truth

            assert aod[i, 0] == aod_in[iw[3], i0[0]]
            assert aod[i, 1] == aod_in[iw[3], i1[0]]

            assert ctype[i, 0] == mode(ctype_in[iw, :][:, i0].flatten())[0]
            assert ctype[i, 1] == mode(ctype_in[iw, :][:, i1].flatten())[0]

            mask = (ctype[i, 0] == ctype_in[iw, :][:, i0].flatten())
            opd_in_masked = opd_in[iw, :][:, i0].flatten()[mask]
            assert opd[i, 0] == np.round(opd_in_masked.mean(), decimals=2)

            mask = (ctype[i, 1] == ctype_in[iw, :][:, i1].flatten())
            opd_in_masked = opd_in[iw, :][:, i1].flatten()[mask]
            assert opd[i, 1] == np.round(opd_in_masked.mean(), decimals=2)


def test_multi_file():
    """Simple test for multi*file.h5 fpath specifications"""
    with tempfile.TemporaryDirectory() as td:

        fpath_multi = os.path.join(td, 'nsrdb_*_2018.h5')
        fpath_out = os.path.join(td, 'agg_out/agg_out_test_2018.h5')
        TESTJOB3 = {'source': {'fpath': fpath_multi,
                               'tree_file': 'kdtree_surfrad_meta.pkl',
                               'meta_file': 'surfrad_meta.csv',
                               'spatial': '2km',
                               'temporal': '5min'},
                    'final': {'fpath': fpath_out,
                              'tree_file': 'kdtree_test_meta_agg.pkl',
                              'meta_file': 'test_meta_agg.csv',
                              'spatial': '4km',
                              'temporal': '30min'},
                    }

        src = os.path.join(TESTDATADIR, 'nsrdb_2018')
        copy_dir(src, td)

        Manager.run_chunk(TESTJOB3, td, meta_dir, 0, 1,
                          year=2018, parallel=False, log_level=False,
                          ignore_dsets=IGNORE_DSETS)

        fpath_out = fpath_out.replace('.h5', '_0.h5')
        with NSRDB(fpath_out, mode='r') as f:
            dsets = ('dni', 'aod', 'cloud_type', 'cld_opd_dcomp')
            assert all(d in f for d in dsets)


if __name__ == '__main__':
    execute_pytest(__file__)
