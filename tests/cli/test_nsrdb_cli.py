"""PyTest file for main nsrdb CLI."""

import json
import os
import shutil
import tempfile
import traceback
from glob import glob

import pytest
from click.testing import CliRunner

from nsrdb import DEFAULT_VAR_META, NSRDB, TESTDATADIR, cli
from nsrdb.data_model import DataModel
from nsrdb.utilities.pytest import execute_pytest


@pytest.fixture(scope='module')
def runner():
    """Runner for testing click CLIs"""
    return CliRunner()


@pytest.fixture(scope='module')
def run_config(tmpdir_factory):
    """Write configs for cli calls."""
    td = str(tmpdir_factory.mktemp('tmp'))
    fn = 'clavrx_OR_ABI-L1b-RadC-M6C01_G16_s*.level2.nc'
    cdir = os.path.join(TESTDATADIR, 'uw_test_cloud_data_nc/2022/{doy}/')
    pattern = os.path.join(cdir, fn)
    albedo_file = tmpdir_factory.mktemp('albedo').join('albedo_2020_004.h5')
    shutil.copy(
        os.path.join(TESTDATADIR, 'albedo', 'nsrdb_albedo_2013_001.h5'),
        albedo_file,
    )
    var_meta = DEFAULT_VAR_META

    kwargs = {
        'pattern': pattern,
        'parallax_correct': False,
        'solar_shading': False,
        'remap_pc': False,
    }
    cloud_vars = list(DataModel.CLOUD_VARS) + list(DataModel.MLCLOUDS_VARS)
    factory_kwargs = dict.fromkeys(cloud_vars, kwargs)
    factory_kwargs['surface_albedo'] = {
        'source_dir': os.path.dirname(albedo_file),
        'cache_file': False,
    }
    nsrdb_grid = os.path.join(TESTDATADIR, 'meta', 'surfrad_meta.csv')
    config_file = os.path.join(td, 'nsrdb_config.json')
    config_dict = {
        'direct': {
            'out_dir': td,
            'year': 2020,
            'nsrdb_grid': nsrdb_grid,
            'nsrdb_freq': '30min',
            'var_meta': var_meta,
        },
        'data-model': {
            'doy_range': [4, 5],
            'max_workers': 1,
            'max_workers_regrid': 1,
            'dist_lim': 1.0,
            'mlclouds': True,
            'factory_kwargs': factory_kwargs,
        },
        'ml-cloud-fill': {
            'fill_all': False,
            'max_workers': 1,
            'col_chunk': 100,
        },
        'daily-all-sky': {},
        'collect-data-model': {'max_workers': 1, 'final': True},
        'execution': {'option': 'local'},
    }
    with open(config_file, 'w') as f:
        f.write(json.dumps(config_dict))
    return config_file


def test_cli_create_configs(runner):
    """Test nsrdb.cli create-configs"""
    with tempfile.TemporaryDirectory() as td:
        kwargs = {
            'year': 2020,
            'outdir': td,
            'satellite': 'east',
            'extent': 'conus',
            'spatial': '4km',
            'freq': '5min',
        }
        result = runner.invoke(cli.create_configs, ['-kw', kwargs])

        if result.exit_code != 0:
            msg = 'Failed with error {}'.format(
                traceback.print_exception(*result.exc_info)
            )
            raise RuntimeError(msg)

        out_dir = f'{td}/nsrdb_east_conus_2020_4km_5min'
        assert os.path.exists(os.path.join(out_dir, 'config_nsrdb.json'))
        assert os.path.exists(os.path.join(out_dir, 'config_pipeline.json'))
        assert os.path.exists(os.path.join(out_dir, 'run.sh'))


def test_cli_pipeline(runner, run_config):
    """Test cli for full pipeline: data-model, gap-fill, all-sky, and
    collection"""

    result = runner.invoke(
        cli.config,
        ['--config_file', run_config, '--command', 'data-model'],
    )
    result = runner.invoke(
        cli.config,
        ['--config_file', run_config, '--command', 'ml-cloud-fill'],
    )

    # specific_humidity and cloud_fill_flag not included in ALL_VARS_ML
    assert len(glob(f'{os.path.dirname(run_config)}/daily/*.h5')) == 2 + len(
        DataModel.ALL_VARS_ML
    )

    result = runner.invoke(
        cli.config,
        ['--config_file', run_config, '--command', 'daily-all-sky'],
    )

    # specific_humidity not included in OUTS or MLCLOUDS_VARS
    assert len(glob(f'{os.path.dirname(run_config)}/daily/*.h5')) == 1 + len(
        DataModel.MLCLOUDS_VARS
    ) + sum(len(v) for v in NSRDB.OUTS.values())

    result = runner.invoke(
        cli.config,
        ['--config_file', run_config, '--command', 'collect-data-model'],
    )
    assert len(glob(f'{os.path.dirname(run_config)}/final/*.h5')) == 7

    if result.exit_code != 0:
        msg = 'Failed with error {}'.format(
            traceback.print_exception(*result.exc_info)
        )
        raise RuntimeError(msg)


if __name__ == '__main__':
    execute_pytest(__file__)
