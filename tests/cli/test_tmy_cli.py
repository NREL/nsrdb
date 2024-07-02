"""PyTest file for main nsrdb CLI."""

import json
import os
import traceback

import numpy as np
import pytest
from click.testing import CliRunner
from rex import ResourceX

from nsrdb import TESTDATADIR, cli
from nsrdb.tmy.tmy import TmyRunner
from nsrdb.utilities.pytest import execute_pytest

NSRDB_BASE_FP = os.path.join(TESTDATADIR, 'validation_nsrdb/nsrdb_*_{}.h5')
CDIR = os.path.join(TESTDATADIR, 'uw_test_cloud_data_nc/2013/{doy}/')
CPATTERN = os.path.join(CDIR, 'clavrx_OR_ABI-L1b-RadC-M6C01_G16_s*.level2.nc')
ALBEDO_FILE = os.path.join(TESTDATADIR, 'albedo', 'nsrdb_albedo_2013_001.h5')
kwargs = {
    'pattern': CPATTERN,
    'parallax_correct': False,
    'solar_shading': False,
    'remap_pc': False,
}
NSRDB_GRID = os.path.join(TESTDATADIR, 'meta', 'surfrad_meta.csv')
DATA_MODEL_CONFIG = {
    'doy_range': [1, 2],
    'max_workers': 1,
    'max_workers_regrid': 1,
    'dist_lim': 1.0,
}


@pytest.fixture(scope='module')
def runner():
    """Runner for testing click CLIs"""
    return CliRunner()


def test_tmy_cli(runner, tmpdir_factory):
    """Make sure tmy runner functions correctly."""

    td = str(tmpdir_factory.mktemp('tmp'))
    td_direct = str(tmpdir_factory.mktemp('direct'))
    tmy_types = ['tmy', 'tdy', 'tgy']
    years = list(range(1998, 2018))
    config = {
        'collect-tmy': {},
        'tmy': {},
        'direct': {
            'tmy_types': tmy_types,
            'nsrdb_base_fp': NSRDB_BASE_FP,
            'years': years,
            'out_dir': str(td),
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
        assert os.path.exists(
            os.path.join(f'{td}/{tmy_type}', f'tmy_{tmy_type}.h5')
        )

    for tmy_type in tmy_types:
        func = getattr(TmyRunner, tmy_type)
        func(
            NSRDB_BASE_FP,
            years,
            out_dir=f'{td_direct}/{tmy_type}',
            fn_out=f'tmy_{tmy_type}.h5',
        )
        TmyRunner.collect(
            NSRDB_BASE_FP,
            years,
            out_dir=f'{td_direct}/{tmy_type}',
            fn_out=f'tmy_{tmy_type}.h5',
        )

    for tmy_type in tmy_types:
        with (
            ResourceX(f'{td}/{tmy_type}/tmy_{tmy_type}.h5') as c_tmy,
            ResourceX(f'{td_direct}/{tmy_type}/tmy_{tmy_type}.h5') as d_tmy,
        ):
            for dset in c_tmy.dsets:
                assert np.array_equal(c_tmy[dset, :], d_tmy[dset, :])


if __name__ == '__main__':
    execute_pytest(__file__)
