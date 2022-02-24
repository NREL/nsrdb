# -*- coding: utf-8 -*-
"""
PyTest file for albedo CLI. Test date created using Jupyter
Notebook at tests/data/albedo/Test data creation.ipynb

Created on Jan 23th 2020

@author: mbannist
"""
import os
import h5py
import pytest
import tempfile
import traceback
import numpy as np
from click.testing import CliRunner

from nsrdb.albedo import cli
from nsrdb import TESTDATADIR
from rex.utilities.loggers import LOGGERS

pytest.importorskip("pyhdf")

BASE_DIR = os.path.dirname(__file__)
TEST_DATA_DIR = os.path.join(BASE_DIR, './data/albedo')
TEST_MERRA_DIR = os.path.join(TESTDATADIR, 'merra2_source_files')


@pytest.fixture(scope="module")
def runner():
    """ Runner for testing click CLIs """
    return CliRunner()


def test_cli_4km_data_with_temp_model(runner):
    """ Test CLI with 4km IMS data """
    with tempfile.TemporaryDirectory() as td:
        log_file = os.path.join(td, 'test.log')
        result = runner.invoke(cli.main, ['-m', TEST_DATA_DIR,
                                          '-i', TEST_DATA_DIR,
                                          '-a', td,
                                          '-me', TEST_MERRA_DIR,
                                          '--log-file', log_file,
                                          'singleday', '2013001',
                                          '--modis-shape', '122', '120',
                                          '--ims-shape', '32', '25', ])
        # assert result.exit_code == 0
        if result.exit_code != 0:
            msg = ('Failed with error {}'
                   .format(traceback.print_exception(*result.exc_info)))
            raise RuntimeError(msg)

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
        LOGGERS.clear()


def test_cli_4km_data(runner):
    """ Test CLI with 4km IMS data """
    with tempfile.TemporaryDirectory() as td:
        log_file = os.path.join(td, 'test.log')
        result = runner.invoke(cli.main, ['-m', TEST_DATA_DIR,
                                          '-i', TEST_DATA_DIR,
                                          '-a', td,
                                          '--log-file', log_file,
                                          'singleday', '2013001',
                                          '--modis-shape', '122', '120',
                                          '--ims-shape', '32', '25', ])
        # assert result.exit_code == 0
        if result.exit_code != 0:
            msg = ('Failed with error {}'
                   .format(traceback.print_exception(*result.exc_info)))
            raise RuntimeError(msg)

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
        LOGGERS.clear()


def test_cli_1km_data(runner):
    """ Test CLI with 1km IMS data """
    with tempfile.TemporaryDirectory() as td:
        log_file = os.path.join(td, 'test2.log')
        result = runner.invoke(cli.main, ['-m', TEST_DATA_DIR,
                                          '-i', TEST_DATA_DIR,
                                          '-a', td,
                                          '--log-file', log_file,
                                          'singleday', '20150101',
                                          '--modis-shape', '60', '61',
                                          '--ims-shape', '64', '50',
                                          ])
        # assert result.exit_code == 0
        if result.exit_code != 0:
            msg = ('Failed with error {}'
                   .format(traceback.print_exception(*result.exc_info)))
            raise RuntimeError(msg)

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
        LOGGERS.clear()


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
