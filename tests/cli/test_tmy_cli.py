"""PyTest file for main nsrdb CLI."""

import json
import os
import traceback
from tempfile import TemporaryDirectory

import numpy as np
import pytest
from click.testing import CliRunner
from rex import ResourceX

from nsrdb import TESTDATADIR, cli
from nsrdb.tmy.tmy import TmyRunner
from nsrdb.utilities.pytest import execute_pytest

NSRDB_BASE_FP = os.path.join(TESTDATADIR, 'validation_nsrdb/nsrdb_*_{}.h5')


@pytest.fixture(scope='module')
def runner():
    """Runner for testing click CLIs"""
    return CliRunner()


def test_tmy_cli(runner):
    """Make sure tmy runner functions correctly."""

    with TemporaryDirectory() as td:
        cli_dir = os.path.join(td, 'cli')
        cli_dir_pattern = os.path.join(cli_dir, '{tmy_type}/')
        cli_file_pattern = os.path.join(cli_dir_pattern, 'tmy_{tmy_type}.h5')
        direct_dir_pattern = os.path.join(td, 'direct', '{tmy_type}/')
        direct_file_pattern = os.path.join(
            direct_dir_pattern, 'tmy_{tmy_type}.h5'
        )
        tmy_types = ['tmy', 'tdy', 'tgy']
        years = list(range(1998, 2018))
        config = {
            'collect-tmy': {},
            'tmy': {},
            'direct': {
                'tmy_types': tmy_types,
                'nsrdb_base_fp': NSRDB_BASE_FP,
                'years': years,
                'out_dir': cli_dir,
                'fn_out': 'tmy.h5',
                'log_level': 'DEBUG',
            },
        }

        os.makedirs(cli_dir, exist_ok=True)
        config_file = os.path.join(cli_dir, 'config.json')
        with open(config_file, 'w') as f:
            f.write(json.dumps(config))

        result = runner.invoke(cli.tmy, ['-c', config_file, '-v'])
        assert result.exit_code == 0, traceback.print_exception(
            *result.exc_info
        )
        result = runner.invoke(cli.tmy, ['-c', config_file, '--collect', '-v'])
        assert result.exit_code == 0, traceback.print_exception(
            *result.exc_info
        )
        for tmy_type in tmy_types:
            assert os.path.exists(cli_file_pattern.format(tmy_type=tmy_type))

        for tmy_type in tmy_types:
            func = getattr(TmyRunner, tmy_type)
            func(
                NSRDB_BASE_FP,
                years,
                out_dir=direct_dir_pattern.format(tmy_type=tmy_type),
                fn_out=f'tmy_{tmy_type}.h5',
            )
            TmyRunner.collect(
                NSRDB_BASE_FP,
                years,
                out_dir=direct_dir_pattern.format(tmy_type=tmy_type),
                fn_out=f'tmy_{tmy_type}.h5',
            )

        for tmy_type in tmy_types:
            with (
                ResourceX(cli_file_pattern.format(tmy_type=tmy_type)) as c_tmy,
                ResourceX(
                    direct_file_pattern.format(tmy_type=tmy_type)
                ) as d_tmy,
            ):
                for dset in c_tmy.dsets:
                    assert np.array_equal(c_tmy[dset, :], d_tmy[dset, :])


if __name__ == '__main__':
    execute_pytest(__file__)
