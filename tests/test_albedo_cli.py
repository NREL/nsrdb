# -*- coding: utf-8 -*-
"""
PyTest file for albedo CLI. Test date created using Jupyter
Notebook at tests/data/albedo/Test data creation.ipynb

Created on Jan 23th 2020

@author: mbannist
"""
import os
import pytest
import numpy as np
import h5py
import tempfile
from click.testing import CliRunner

import nsrdb.albedo.cli as cli

BASE_DIR = os.path.dirname(__file__)
TEST_DATA_DIR = os.path.join(BASE_DIR, './data/albedo')


@pytest.fixture(scope="module")
def runner():
    return CliRunner()


def test_cli_4km_data(runner):
    """ Test CLI with 4km IMS data """
    with tempfile.TemporaryDirectory() as td:
        result = runner.invoke(cli.main, ['-m', TEST_DATA_DIR,
                                          '-i', TEST_DATA_DIR,
                                          '-a', td,
                                          'singleday', '2013001',
                                          '--modis-shape', '122', '120',
                                          '--ims-shape', '32', '25'])
        assert result.exit_code == 0

        # Compare against known output
        fname = os.path.join(TEST_DATA_DIR, 'nsrdb_albedo_2013_001.h5')
        with h5py.File(fname, 'r') as f:
            known_data = np.array(f['surface_albedo'])

        fname = os.path.join(td, 'nsrdb_albedo_2013_001.h5')
        with h5py.File(fname, 'r') as f:
            new_data = np.array(f['surface_albedo'])

        assert np.array_equal(known_data, new_data)

        # Compare against wrong output
        fname = os.path.join(TEST_DATA_DIR, 'nsrdb_albedo_2015_001.h5')
        with h5py.File(fname, 'r') as f:
            wrong_data = np.array(f['surface_albedo'])

        assert not np.array_equal(wrong_data, new_data)


def test_cli_1km_data(runner):
    """ Test CLI with 1km IMS data """
    with tempfile.TemporaryDirectory() as td:
        result = runner.invoke(cli.main, ['-m', TEST_DATA_DIR,
                                          '-i', TEST_DATA_DIR,
                                          '-a', td,
                                          'singleday', '20150101',
                                          '--modis-shape', '60', '61',
                                          '--ims-shape', '64', '50'])
        assert result.exit_code == 0

        # Compare against known output
        fname = os.path.join(TEST_DATA_DIR, 'nsrdb_albedo_2015_001.h5')
        with h5py.File(fname, 'r') as f:
            known_data = np.array(f['surface_albedo'])

        fname = os.path.join(td, 'nsrdb_albedo_2015_001.h5')
        with h5py.File(fname, 'r') as f:
            new_data = np.array(f['surface_albedo'])
        assert np.array_equal(known_data, new_data)

        # Compare against wrong output
        fname = os.path.join(TEST_DATA_DIR, 'nsrdb_albedo_2013_001.h5')
        with h5py.File(fname, 'r') as f:
            wrong_data = np.array(f['surface_albedo'])
        assert not np.array_equal(wrong_data, new_data)


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