"""PyTest file for main nsrdb CLI."""

import json
import os
import tempfile
import traceback
from glob import glob

import pytest
from rex import safe_json_load

from nsrdb import NSRDB, TESTDATADIR, cli
from nsrdb.data_model import DataModel
from nsrdb.utilities.pytest import execute_pytest

VAR_META = os.path.join(TESTDATADIR, 'nsrdb_vars.csv')
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


@pytest.fixture(scope='function')
def legacy_config(tmpdir_factory):
    """Write configs for cli calls. Uses legacy gap fill method"""
    td = str(tmpdir_factory.mktemp('tmp'))
    config_file = os.path.join(td, 'config_nsrdb.json')
    pipeline_file = os.path.join(td, 'config_pipeline.json')
    factory_kwargs = dict.fromkeys(list(DataModel.CLOUD_VARS), kwargs)
    factory_kwargs['surface_albedo'] = {
        'source_dir': os.path.dirname(ALBEDO_FILE),
        'cache_file': False,
    }
    config_dict = {
        'direct': {
            'out_dir': td,
            'year': 2013,
            'grid': NSRDB_GRID,
            'freq': '30min',
            'var_meta': VAR_META,
        },
        'data-model': {
            **DATA_MODEL_CONFIG,
            'mlclouds': False,
            'factory_kwargs': factory_kwargs,
        },
        'collect-data-model': {
            'max_workers': 1,
            'n_chunks': 1,
            'final': False,
        },
        'cloud-fill': {},
        'all-sky': {'n_chunks': 1},
        'collect-final': {'collect_dir': f'{td}/collect'},
        'execution_control': {'option': 'local'},
    }
    with open(config_file, 'w') as f:
        f.write(json.dumps(config_dict))

    with open(pipeline_file, 'w') as f:
        f.write(
            json.dumps(
                {
                    'logging': {'log_file': None, 'log_level': 'INFO'},
                    'pipeline': [
                        {'data-model': config_file},
                        {'collect-data-model': config_file},
                        {'cloud-fill': config_file},
                        {'all-sky': config_file},
                        {'collect-final': config_file},
                    ],
                }
            )
        )
    return config_file, pipeline_file


@pytest.fixture(scope='function')
def modern_config(tmpdir_factory):
    """Write configs for cli calls."""
    td = str(tmpdir_factory.mktemp('tmp'))
    config_file = os.path.join(td, 'config_nsrdb.json')
    pipeline_file = os.path.join(td, 'config_pipeline.json')
    cloud_vars = list(DataModel.CLOUD_VARS) + list(DataModel.MLCLOUDS_VARS)
    factory_kwargs = dict.fromkeys(cloud_vars, kwargs)
    factory_kwargs['surface_albedo'] = {
        'source_dir': os.path.dirname(ALBEDO_FILE),
        'cache_file': False,
    }
    config_dict = {
        'direct': {
            'out_dir': td,
            'year': 2013,
            'grid': NSRDB_GRID,
            'freq': '30min',
            'var_meta': VAR_META,
        },
        'data-model': {
            **DATA_MODEL_CONFIG,
            'mlclouds': True,
            'factory_kwargs': factory_kwargs,
        },
        'ml-cloud-fill': {
            'fill_all': False,
            'max_workers': 1,
            'col_chunk': 100,
        },
        'daily-all-sky': {},
        'collect-data-model': {
            'max_workers': 1,
            'final': True,
            'final_file_name': 'test',
        },
        'execution_control': {'option': 'local'},
    }
    with open(config_file, 'w') as f:
        f.write(json.dumps(config_dict))

    with open(pipeline_file, 'w') as f:
        f.write(
            json.dumps(
                {
                    'logging': {'log_file': None, 'log_level': 'INFO'},
                    'pipeline': [
                        {'data-model': config_file},
                        {'ml-cloud-fill': config_file},
                        {'daily-all-sky': config_file},
                        {'collect-data-model': config_file},
                    ],
                }
            )
        )
    return config_file, pipeline_file


def test_cli_create_main_configs(runner):
    """Test nsrdb.cli create-configs"""
    with tempfile.TemporaryDirectory() as td:
        kwargs = {
            'year': 2020,
            'out_dir': td,
            'satellite': 'east',
            'extent': 'conus',
            'spatial': '4km',
            'freq': '5min',
        }
        result = runner.invoke(cli.create_configs, ['-c', kwargs])

        assert result.exit_code == 0, traceback.print_exception(
            *result.exc_info
        )

        out_dir = f'{td}/nsrdb_east_conus_2020_4km_5min'
        assert os.path.exists(os.path.join(out_dir, 'config_nsrdb.json'))
        assert os.path.exists(os.path.join(out_dir, 'config_pipeline.json'))
        assert os.path.exists(os.path.join(out_dir, 'run.sh'))

        kwargs = {'year': 2020, 'out_dir': td}
        result = runner.invoke(
            cli.create_configs, ['-c', kwargs, '--all_domains']
        )

        assert result.exit_code == 0, traceback.print_exception(
            *result.exc_info
        )

        out_dirs = [
            f'{td}/nsrdb_east_conus_2020_2km_5min',
            f'{td}/nsrdb_west_conus_2020_2km_5min',
            f'{td}/nsrdb_east_full_2020_2km_10min',
            f'{td}/nsrdb_west_full_2020_2km_10min',
        ]
        for out_dir in out_dirs:
            assert os.path.exists(os.path.join(out_dir, 'config_nsrdb.json'))
            assert os.path.exists(
                os.path.join(out_dir, 'config_pipeline.json')
            )
            assert os.path.exists(os.path.join(out_dir, 'run.sh'))


def test_cli_create_blend_configs(runner):
    """Test nsrdb.cli create-configs --run_type blend"""
    with tempfile.TemporaryDirectory() as td:
        kwargs = {'year': 2020, 'out_dir': td, 'extent': 'conus'}
        result = runner.invoke(
            cli.create_configs, ['-c', kwargs, '--run_type', 'blend']
        )

        assert result.exit_code == 0, traceback.print_exception(
            *result.exc_info
        )

        config_file = os.path.join(td, 'config_blend.json')
        assert os.path.exists(config_file)
        config = safe_json_load(config_file)
        assert config['year'] == 2020
        assert 'west_dir' in config and 'east_dir' in config


def test_cli_create_agg_configs(runner):
    """Test nsrdb.cli create-configs --run_type aggregate"""
    with tempfile.TemporaryDirectory() as td:
        kwargs = {'year': 2020, 'out_dir': td}
        result = runner.invoke(
            cli.create_configs, ['-c', kwargs, '--run_type', 'aggregate']
        )

        assert result.exit_code == 0, traceback.print_exception(
            *result.exc_info
        )

        config_file = os.path.join(td, 'config_aggregate.json')
        assert os.path.exists(config_file)

        config = safe_json_load(config_file)
        assert config['year'] == 2020
        assert config['full_freq'] == '10min'
        assert config['conus_freq'] == '5min'
        assert config['final_freq'] == '30min'


def test_cli_steps(runner, modern_config):
    """Test cli for full pipeline, using separate config calls to data-model,
    gap-fill, all-sky, and collection"""

    config_file, _ = modern_config
    out_dir = os.path.dirname(config_file)
    result = runner.invoke(cli.data_model, ['-c', config_file, '-v'])
    assert result.exit_code == 0, traceback.print_exception(*result.exc_info)
    result = runner.invoke(cli.ml_cloud_fill, ['-c', config_file])
    assert result.exit_code == 0, traceback.print_exception(*result.exc_info)

    # specific_humidity and cloud_fill_flag not included in ALL_VARS_ML
    assert len(glob(f'{out_dir}/daily/*.h5')) == 2 + len(DataModel.ALL_VARS_ML)

    result = runner.invoke(cli.daily_all_sky, ['-c', config_file, '-v'])
    assert result.exit_code == 0, traceback.print_exception(*result.exc_info)

    # specific_humidity not included in OUTS or MLCLOUDS_VARS
    assert len(glob(f'{out_dir}/daily/*.h5')) == 1 + len(
        DataModel.MLCLOUDS_VARS
    ) + sum(len(v) for v in NSRDB.OUTS.values())

    result = runner.invoke(cli.collect_data_model, ['-c', config_file, '-v'])
    assert result.exit_code == 0, traceback.print_exception(*result.exc_info)
    assert len(glob(f'{out_dir}/final/*.h5')) == 7

    status_files = glob(out_dir + '/.gaps/jobstatus*.json')
    status_dicts = [safe_json_load(sf) for sf in status_files]
    for sd in status_dicts:
        assert all('successful' in str(vals) for vals in sd.values())


def test_cli_pipeline(runner, modern_config):
    """Test cli for pipeline, run using cli.pipeline"""

    config_file, pipeline_file = modern_config
    config = safe_json_load(config_file)
    out_dir = os.path.dirname(pipeline_file)
    n_days = len(config['data-model']['doy_range']) - 1

    # data-model
    result = runner.invoke(cli.pipeline, ['-c', pipeline_file, '-v'])
    assert result.exit_code == 0, traceback.print_exception(*result.exc_info)
    assert len(glob(out_dir + '/logs/data_model/*.log')) == n_days

    # ml-cloud-fill
    result = runner.invoke(cli.pipeline, ['-c', pipeline_file, '-v'])
    assert result.exit_code == 0, traceback.print_exception(*result.exc_info)
    assert len(glob(out_dir + '/logs/ml_cloud_fill/*.log')) == n_days

    # specific_humidity and cloud_fill_flag not included in ALL_VARS_ML
    assert len(glob(f'{out_dir}/daily/*.h5')) == 2 + len(DataModel.ALL_VARS_ML)

    # all-sky
    result = runner.invoke(cli.pipeline, ['-c', pipeline_file, '-v'])
    assert result.exit_code == 0, traceback.print_exception(*result.exc_info)
    assert len(glob(out_dir + '/logs/daily_all_sky/*.log')) == n_days

    # specific_humidity not included in OUTS or MLCLOUDS_VARS
    assert len(glob(f'{out_dir}/daily/*.h5')) == 1 + len(
        DataModel.MLCLOUDS_VARS
    ) + sum(len(v) for v in NSRDB.OUTS.values())

    # data collection
    result = runner.invoke(cli.pipeline, ['-c', pipeline_file, '-v'])
    assert result.exit_code == 0, traceback.print_exception(*result.exc_info)
    final_files = glob(f'{out_dir}/final/*.h5')
    final_files = sorted([os.path.basename(f) for f in final_files])
    target_files = sorted(
        f.format(y=config['direct']['year']) for f in sorted(NSRDB.OUTS.keys())
    )
    target_files = [
        f.replace(
            'nsrdb_',
            f"{config['collect-data-model'].get('final_file_name', '')}_",
        )
        for f in target_files
    ]
    assert target_files == final_files
    assert len(glob(out_dir + '/logs/collect_data_model/*.log')) == len(
        NSRDB.OUTS
    )

    # final status file update
    result = runner.invoke(cli.pipeline, ['-c', pipeline_file])

    status_file = glob(out_dir + '/.gaps/*.json')[0]
    status_dict = safe_json_load(status_file)
    assert all('successful' in str(vals) for vals in status_dict.values())


def test_cli_pipeline_legacy(runner, legacy_config):
    """Test cli for pipeline, run using cli.pipeline. Uses legacy gap-fill
    method"""

    config_file, pipeline_file = legacy_config
    config = safe_json_load(config_file)
    out_dir = os.path.dirname(pipeline_file)
    n_chunks = config['collect-data-model']['n_chunks']

    # data-model
    result = runner.invoke(cli.pipeline, ['-c', pipeline_file, '-v'])
    assert result.exit_code == 0, traceback.print_exception(*result.exc_info)

    # pre gap-fill collection
    result = runner.invoke(cli.pipeline, ['-c', pipeline_file, '-v'])
    assert result.exit_code == 0, traceback.print_exception(*result.exc_info)

    # specific humidity not included in ALL_VARS
    assert len(
        glob(f'{os.path.dirname(pipeline_file)}/daily/*.h5')
    ) == 1 + len(DataModel.ALL_VARS)

    # collected data doesn't include all-sky files yet (irradiance / clearsky)
    assert len(glob(f'{out_dir}/collect/*.h5')) == n_chunks * 5
    assert (
        len(glob(out_dir + '/logs/collect_data_model/*.log')) == n_chunks * 5
    )

    # gap-fill
    result = runner.invoke(cli.pipeline, ['-c', pipeline_file, '-v'])
    assert result.exit_code == 0, traceback.print_exception(*result.exc_info)

    # all-sky
    result = runner.invoke(cli.pipeline, ['-c', pipeline_file, '-v'])
    assert result.exit_code == 0, traceback.print_exception(*result.exc_info)

    # irrad and clearsky now in collect directory
    assert len(glob(f'{out_dir}/collect/*.h5')) == n_chunks * len(NSRDB.OUTS)

    # final collection
    result = runner.invoke(cli.pipeline, ['-c', pipeline_file, '-v'])
    assert result.exit_code == 0, traceback.print_exception(*result.exc_info)
    final_files = glob(f'{out_dir}/final/*.h5')
    final_files = sorted([os.path.basename(f) for f in final_files])
    assert (
        sorted(
            f.format(y=config['direct']['year'])
            for f in sorted(NSRDB.OUTS.keys())
        )
        == final_files
    )
    assert len(glob(out_dir + '/logs/collect_final/*.log')) == len(NSRDB.OUTS)

    # final status file update
    result = runner.invoke(cli.pipeline, ['-c', pipeline_file, '-v'])

    status_file = glob(out_dir + '/.gaps/*.json')[0]
    status_dict = safe_json_load(status_file)
    assert all('successful' in str(vals) for vals in status_dict.values())


if __name__ == '__main__':
    execute_pytest(__file__)
