"""PyTest file for main nsrdb CLI."""

import json
import os
import traceback

import numpy as np
import pytest
from rex import ResourceX

from nsrdb import TESTDATADIR, cli
from nsrdb.tmy import TmyRunner
from nsrdb.utilities.pytest import execute_pytest

NSRDB_BASE_FP = os.path.join(TESTDATADIR, 'validation_nsrdb/nsrdb_*_{}.h5')
site_slice = [0, 2]
years = list(range(1998, 2008))
tmy_types = ['tmy', 'tdy', 'tgy']


@pytest.fixture(scope='function')
def tmy_file(tmpdir_factory):
    """Run tmy directly."""
    direct_dir_pattern = str(tmpdir_factory.mktemp('tmp')) + '/{tmy_type}/'

    for tmy_type in tmy_types:
        TmyRunner.tmy(
            NSRDB_BASE_FP,
            years,
            tmy_type=tmy_type,
            site_slice=slice(*site_slice),
            out_dir=direct_dir_pattern.format(tmy_type=tmy_type),
            fn_out=f'tmy_{tmy_type}.h5',
        )
        TmyRunner.collect(
            NSRDB_BASE_FP,
            years,
            site_slice=slice(*site_slice),
            out_dir=direct_dir_pattern.format(tmy_type=tmy_type),
            fn_out=f'tmy_{tmy_type}.h5',
        )

    return direct_dir_pattern + '/tmy_{tmy_type}.h5'


def test_tmy_cli(runner, tmpdir_factory):
    """Make sure tmy runner functions correctly."""

    td = str(tmpdir_factory.mktemp('tmp'))
    dir_pattern = os.path.join(td, '{tmy_type}/')
    file_pattern = os.path.join(dir_pattern, 'tmy_{tmy_type}.h5')
    config = {
        'collect-tmy': {'purge_chunks': True},
        'tmy': {},
        'direct': {
            'site_slice': site_slice,
            'tmy_types': tmy_types,
            'nsrdb_base_fp': NSRDB_BASE_FP,
            'years': years,
            'out_dir': td,
            'fn_out': 'tmy.h5',
            'log_level': 'DEBUG',
        },
    }

    config_file = os.path.join(td, 'config.json')
    with open(config_file, 'w') as f:
        f.write(json.dumps(config))

    result = runner.invoke(cli.tmy, ['-c', config_file, '-v'])
    assert result.exit_code == 0, traceback.print_exception(*result.exc_info)
    result = runner.invoke(cli.tmy, ['-c', config_file, '--collect', '-v'])
    assert result.exit_code == 0, traceback.print_exception(*result.exc_info)
    for tmy_type in tmy_types:
        assert os.path.exists(file_pattern.format(tmy_type=tmy_type))


def test_tmy_regression(runner, tmpdir_factory, tmy_file):
    """Make sure tmy cli agrees with direct calc."""

    td = str(tmpdir_factory.mktemp('tmp'))
    dir_pattern = os.path.join(td, '{tmy_type}/')
    file_pattern = os.path.join(dir_pattern, 'tmy_{tmy_type}.h5')
    tmy_types = ['tmy']
    config = {
        'collect-tmy': {'purge_chunks': True},
        'tmy': {},
        'direct': {
            'site_slice': site_slice,
            'tmy_types': tmy_types,
            'nsrdb_base_fp': NSRDB_BASE_FP,
            'years': years,
            'out_dir': td,
            'fn_out': 'tmy.h5',
            'log_level': 'DEBUG',
        },
    }

    config_file = os.path.join(td, 'config.json')
    with open(config_file, 'w') as f:
        f.write(json.dumps(config))

    result = runner.invoke(cli.tmy, ['-c', config_file, '-v'])
    assert result.exit_code == 0, traceback.print_exception(*result.exc_info)
    result = runner.invoke(cli.tmy, ['-c', config_file, '--collect', '-v'])
    assert result.exit_code == 0, traceback.print_exception(*result.exc_info)

    for tmy_type in tmy_types:
        with (
            ResourceX(file_pattern.format(tmy_type=tmy_type)) as c_tmy,
            ResourceX(tmy_file.format(tmy_type=tmy_type)) as d_tmy,
        ):
            for dset in c_tmy.dsets:
                assert np.array_equal(c_tmy[dset, :], d_tmy[dset, :])


if __name__ == '__main__':
    execute_pytest(__file__)
