# -*- coding: utf-8 -*-
"""NSRDB Command Line Interface (CLI).

Created on Mon Oct 21 15:39:01 2019

@author: gbuster
"""

import json
import logging
import os

import click
from gaps import Pipeline
from gaps.cli.pipeline import pipeline as gaps_pipeline
from rex import safe_json_load
from rex.utilities.fun_utils import get_fun_call_str
from rex.utilities.loggers import init_logger

from nsrdb import __version__
from nsrdb.file_handlers.collection import Collector
from nsrdb.nsrdb import NSRDB
from nsrdb.utilities import ModuleName
from nsrdb.utilities.cli import BaseCLI
from nsrdb.utilities.file_utils import ts_freq_check

logger = logging.getLogger(__name__)


class DictOrFile(click.ParamType):
    """Dict or file click input argument type."""

    name = 'dict_or_file'

    @staticmethod
    def convert(value, param, ctx):
        """Convert to dict or return as None."""
        if isinstance(value, dict):
            return value
        if isinstance(value, str) and os.path.exists(value):
            return value
        if isinstance(value, str):
            try:
                return json.loads(value)
            except Exception as e:
                msg = (
                    f'Could not load {value} as a dictionary. Make sure '
                    'you provide a valid string representation'
                )
                raise ValueError(msg) from e
        if value is None:
            return None
        raise TypeError(
            'Cannot recognize input type: {} {} {} {}'.format(
                value, type(value), param, ctx
            )
        )


CONFIG_TYPE = DictOrFile()


@click.group()
@click.version_option(version=__version__)
@click.option(
    '--config',
    '-c',
    required=True,
    type=click.Path(exists=True),
    help='NSRDB config file json for a single module.',
)
@click.option(
    '-v',
    '--verbose',
    is_flag=True,
    help='Flag to turn on debug logging. Default is not verbose.',
)
@click.pass_context
def main(ctx, config_file, verbose):
    """NSRDB command line interface.

    Try using the following commands to pull up the help pages for the
    respective NSRDB CLIs::

        $ nsrdb --help

        $ nsrdb -c config.json pipeline --help

        $ nsrdb -c config.json data-model --help

        $ nsrdb -c config.json ml-cloud-fill --help

        $ nsrdb -c config.json daily-all-sky --help

        $ nsrdb -c config.json collect-data-model --help

    Typically, a good place to start is to set up a sup3r job with a pipeline
    config that points to several NSRDB modules that you want to run in serial.
    You would call the NSRDB pipeline CLI using::

        $ nsrdb -c config_pipeline.json pipeline

    See the help pages of the module CLIs for more details on the config files
    for each CLI.
    """
    ctx.ensure_object(dict)
    ctx.obj['CONFIG_FILE'] = config_file
    ctx.obj['VERBOSE'] = verbose


@main.group(invoke_without_command=True)
@click.option(
    '--config',
    '-c',
    required=True,
    type=click.Path(exists=True),
    help='NSRDB pipeline configuration json file.',
)
@click.option(
    '--cancel',
    is_flag=True,
    help='Flag to cancel all jobs associated with a given pipeline.',
)
@click.option(
    '--monitor',
    is_flag=True,
    help='Flag to monitor pipeline jobs continuously. '
    'Default is not to monitor (kick off jobs and exit).',
)
@click.option(
    '--background',
    is_flag=True,
    help='Flag to monitor pipeline jobs continuously in the '
    'background using the nohup command. This only works with the '
    '--monitor flag. Note that the stdout/stderr will not be '
    'captured, but you can set a pipeline log_file to capture logs.',
)
@click.option(
    '-v',
    '--verbose',
    is_flag=True,
    help='Flag to turn on debug logging. Default is not verbose.',
)
@click.pass_context
def pipeline(ctx, config, cancel, monitor, background, verbose):
    """Run NSRDB pipeline from a pipeline config file."""

    ctx.ensure_object(dict)
    ctx.obj['VERBOSE'] = verbose or ctx.obj.get('VERBOSE', False)
    gaps_pipeline(config, cancel, monitor, background)


@main.command()
@click.option(
    '--config',
    '-c',
    required=True,
    type=CONFIG_TYPE,
    help='Either a path to a config file or a dictionary. '
    'Needs to include year. If a dictionary it needs to be provided '
    'in a string format. e.g. \'{"year":2019, "freq":"5min"}\'. '
    '\n\nAvailable keys: '
    'year, freq, outdir (parent directory for run directory), '
    'satellite (east/west), '
    'spatial (meta file resolution), '
    'extent (full/conus), '
    'basename (file prefix), '
    'meta_file. (auto populated if None), '
    'doy_range (all days of year if None).'
    '\n\ndefault_kwargs = {"basename": "nsrdb", '
    '"freq": "5min", "satellite": "east", '
    '"extent": "conus", "outdir": "./", '
    '"spatial": "4km", "meta_file" : None, '
    '"doy_range": None}',
)
@click.option(
    '-all_domains',
    '-ad',
    is_flag=True,
    help='Flag to generate config files for all '
    'domains. If True config files for east/west and '
    'conus/full will be generated. (just full if year '
    'is < 2018). satellite, extent, spatial, freq, and '
    'meta_file will be auto populated. ',
)
@click.pass_context
def create_configs(ctx, kwargs, all_domains):
    """Create config files for standard NSRDB runs using config templates."""

    ctx.ensure_object(dict)
    if all_domains:
        NSRDB.create_configs_all_domains(kwargs)
    else:
        NSRDB.create_config_files(kwargs)


@main.group()
@click.option(
    '--config',
    '-c',
    type=str,
    required=True,
    help='Path to config file with kwargs for NSRDB.blend_files()',
)
@click.option(
    '-v',
    '--verbose',
    is_flag=True,
    help='Flag to turn on debug logging. Default is not verbose.',
)
@click.pass_context
def blend(ctx, config, verbose=False, pipeline_step=None):
    """Blend files from separate domains (e.g. east / west) into a single
    domain."""

    config = BaseCLI.from_config_preflight(
        ModuleName.BLEND, ctx, config, verbose, pipeline_step=pipeline_step
    )
    log_level = config.get('log_level', 'INFO')
    log_arg_str = f'"nsrdb", log_level="{log_level}"'
    log_file = config.get('log_file', None)

    if log_file is not None:
        log_arg_str += f', log_file="{log_file}"'

    ctx.obj['LOG_ARG_STR'] = log_arg_str
    ctx.obj['FUN_STR'] = get_fun_call_str(NSRDB.blend_files, config)
    BaseCLI.kickoff_job(ModuleName.BLEND, config, ctx)


@main.group()
@click.option(
    '--config',
    '-c',
    type=str,
    required=True,
    help='Path to config file with kwargs for NSRDB.collect_blended()',
)
@click.option(
    '-v',
    '--verbose',
    is_flag=True,
    help='Flag to turn on debug logging. Default is not verbose.',
)
def collect_blended(ctx, config, verbose=False, pipeline_step=None):
    """Collect blended data chunks into a single file."""

    config = BaseCLI.from_config_preflight(
        ModuleName.COLLECT_BLENDED,
        ctx,
        config,
        verbose,
        pipeline_step=pipeline_step,
    )

    ctx.obj['FUN_STR'] = get_fun_call_str(NSRDB.collect_blended, config)
    BaseCLI.kickoff_job(ModuleName.COLLECT_BLENDED, config, ctx)


@main.group()
@click.option(
    '--config',
    '-c',
    type=str,
    required=True,
    help='Path to config file with kwargs for NSRDB.aggregate_files()',
)
@click.option(
    '-v',
    '--verbose',
    is_flag=True,
    help='Flag to turn on debug logging. Default is not verbose.',
)
@click.pass_context
def aggregate(ctx, config, verbose=False, pipeline_step=None):
    """Aggregate data files to a lower resolution.

    Note
    ----
    Used to create data files from high-resolution years (2018+) which match
    resolution of low-resolution years (pre 2018)
    """

    config = BaseCLI.from_config_preflight(
        ModuleName.AGGREGATE, ctx, config, verbose, pipeline_step=pipeline_step
    )
    ctx.obj['FUN_STR'] = get_fun_call_str(NSRDB.aggregate_files, config)
    BaseCLI.kickoff_job(ModuleName.AGGREGATE, config, ctx)


@main.group()
@click.option(
    '--config',
    '-c',
    type=str,
    required=True,
    help='Path to config file with kwargs for NSRDB.collect_aggregation()',
)
@click.option(
    '-v',
    '--verbose',
    is_flag=True,
    help='Flag to turn on debug logging. Default is not verbose.',
)
def collect_aggregation(ctx, config, verbose=False, pipeline_step=None):
    """Collect aggregated data chunks."""

    config = BaseCLI.from_config_preflight(
        ModuleName.COLLECT_AGG,
        ctx,
        config,
        verbose,
        pipeline_step=pipeline_step,
    )

    ctx.obj['FUN_STR'] = get_fun_call_str(NSRDB.collect_aggregation, config)
    BaseCLI.kickoff_job(ModuleName.COLLECT_AGG, config, ctx)


@main.command()
@click.option(
    '--config',
    '-c',
    required=True,
    type=CONFIG_TYPE,
    help='Path to .json config file or str rep of dictionary.',
)
@click.option(
    '--command',
    '-cmd',
    type=str,
    required=True,
    help='NSRDB CLI command string.',
)
@click.option(
    '-v',
    '--verbose',
    is_flag=True,
    help='Flag to turn on debug logging. Default is not verbose.',
)
@click.pass_context
def config(ctx, config, command, verbose=False):
    """NSRDB processing CLI from config json file."""

    ctx.ensure_object(dict)
    config_dict = safe_json_load(config)
    direct = config_dict.get('direct', {})
    msg = 'Config must include "freq" key.'
    assert 'freq' in config_dict or 'freq' in direct, msg
    nsrdb_freq = config_dict.get('freq', direct['freq'])
    ts_freq_check(nsrdb_freq)

    ctx.obj['LOG_LEVEL'] = 'DEBUG' if verbose else 'INFO'

    init_logger('nsrdb.cli', log_level=ctx.obj['LOG_LEVEL'], log_file=None)

    if command == 'data-model':
        ctx.invoke(data_model, config=config, verbose=verbose)
    elif command == 'ml-cloud-fill':
        ctx.invoke(ml_cloud_fill, config=config, verbose=verbose)
    elif command == 'daily-all-sky':
        ctx.invoke(daily_all_sky, config=config, verbose=verbose)
    elif command == 'collect-data-model':
        ctx.invoke(collect_data_model, config=config, verbose=verbose)
    elif command == 'cloud-fill':
        ctx.invoke(cloud_fill, config=config, verbose=verbose)
    elif command == 'all-sky':
        ctx.invoke(all_sky, config=config, verbose=verbose)
    elif command == 'collect-daily':
        ctx.invoke(collect_daily, config=config, verbose=verbose)
    elif command == 'collect-flist':
        ctx.invoke(collect_flist, config=config, verbose=verbose)
    elif command == 'collect-final':
        ctx.invoke(collect_final, config=config, verbose=verbose)
    else:
        raise KeyError('Command not recognized: "{}"'.format(command))


@main.group()
@click.option(
    '--config',
    '-c',
    type=CONFIG_TYPE,
    required=True,
    help='Path to config file or dict of kwargs for NSRDB.run_data_model()',
)
@click.option(
    '-v',
    '--verbose',
    is_flag=True,
    help='Flag to turn on debug logging. Default is not verbose.',
)
@click.pass_context
def data_model(ctx, config, verbose=False, pipeline_step=None):
    """Run daily data model and save output files."""
    BaseCLI.kickoff_multiday(
        module_name=ModuleName.DATA_MODEL,
        func=NSRDB.run_data_model,
        config=config,
        ctx=ctx,
        verbose=verbose,
        pipeline_step=pipeline_step,
    )


@main.group()
@click.option(
    '--config',
    '-c',
    type=CONFIG_TYPE,
    required=True,
    help='Path to config file or dict with kwargs for NSRDB.ml_cloud_fill()',
)
@click.option(
    '-v',
    '--verbose',
    is_flag=True,
    help='Flag to turn on debug logging. Default is not verbose.',
)
@click.pass_context
def ml_cloud_fill(ctx, config, verbose=False, pipeline_step=None):
    """Gap fill cloud properties using mlclouds."""
    BaseCLI.kickoff_multiday(
        module_name=ModuleName.ML_CLOUD_FILL,
        func=NSRDB.ml_cloud_fill,
        config=config,
        ctx=ctx,
        verbose=verbose,
        pipeline_step=pipeline_step,
    )


@main.group()
@click.option(
    '--config',
    '-c',
    type=CONFIG_TYPE,
    required=True,
    help=(
        'Path to config file or dict with kwargs for '
        'NSRDB.run_daily_all_sky()'
    ),
)
@click.option(
    '-v',
    '--verbose',
    is_flag=True,
    help='Flag to turn on debug logging. Default is not verbose.',
)
@click.pass_context
def daily_all_sky(ctx, config, verbose=False, pipeline_step=None):
    """Run all-sky physics model on daily data model output."""
    BaseCLI.kickoff_multiday(
        ModuleName.DAILY_ALL_SKY,
        func=NSRDB.run_daily_all_sky,
        config=config,
        ctx=ctx,
        verbose=verbose,
        pipeline_step=pipeline_step,
    )


@main.group()
@click.option(
    '--config',
    '-c',
    type=CONFIG_TYPE,
    required=True,
    help='Path to config file or dict with kwargs for NSRDB.cloud_fill()',
)
@click.option(
    '-v',
    '--verbose',
    is_flag=True,
    help='Flag to turn on debug logging. Default is not verbose.',
)
@click.pass_context
def cloud_fill(ctx, config, verbose=False, pipeline_step=None):
    """Gap fill cloud properties in a collect data model output file, using
    legacy gap-fill method."""

    config = BaseCLI.from_config_preflight(
        ModuleName.CLOUD_FILL,
        config=config,
        ctx=ctx,
        verbose=verbose,
        pipeline_step=pipeline_step,
    )

    log_file = config.get('log_file', None)
    log_level = config.get('log_level', 'INFO')
    log_arg_str = f'"nsrdb", log_level="{log_level}"'
    config['n_chunks'] = config.get('n_chunks', 1)
    name = ctx.obj['NAME']

    for i_chunk in range(config['n_chunks']):
        if log_file is not None:
            log_file_i = log_file.replace('.log', f'_{i_chunk}.log')
            log_arg_str_i = f'{log_arg_str}, log_file="{log_file_i}"'
            ctx.obj['LOG_ARG_STR'] = log_arg_str_i
        config['i_chunk'] = i_chunk
        config['job_name'] = f'{name}_{i_chunk}'
        ctx.obj['NAME'] = config['job_name']
        ctx.obj['FUN_STR'] = get_fun_call_str(NSRDB.gap_fill_clouds, config)

        BaseCLI.kickoff_job(ModuleName.CLOUD_FILL, config, ctx)


@main.group()
@click.option(
    '--config',
    '-c',
    type=CONFIG_TYPE,
    required=True,
    help='Path to config file or dict with kwargs for NSRDB.run_all_sky()',
)
@click.option(
    '-v',
    '--verbose',
    is_flag=True,
    help='Flag to turn on debug logging. Default is not verbose.',
)
@click.pass_context
def all_sky(ctx, config, verbose=False, pipeline_step=None):
    """Run all-sky physics model on collected data model output files."""

    config = BaseCLI.from_config_preflight(
        ModuleName.ALL_SKY,
        ctx=ctx,
        config=config,
        verbose=verbose,
        pipeline_step=pipeline_step,
    )

    log_level = config.get('log_level', 'INFO')
    log_arg_str = f'"nsrdb", log_level="{log_level}"'
    config['n_chunks'] = config.get('n_chunks', 1)
    name = ctx.obj['NAME']

    for i_chunk in range(config['n_chunks']):
        log_file = f'{ctx.obj["out_dir"]}/all_sky/all_sky_{i_chunk}.log'
        log_arg_str_i = f'{log_arg_str}, log_file="{log_file}"'
        config['i_chunk'] = i_chunk
        config['job_name'] = f'{name}_{i_chunk}'
        ctx.obj['LOG_ARG_STR'] = log_arg_str_i
        ctx.obj['NAME'] = config['job_name']
        ctx.obj['FUN_STR'] = get_fun_call_str(NSRDB.run_all_sky, config)

        BaseCLI.kickoff_job(ModuleName.ALL_SKY, config, ctx)


@main.group()
@click.option(
    '--config',
    '-c',
    type=CONFIG_TYPE,
    required=True,
    help=(
        'Path to config file or dict with kwargs for '
        'NSRDB.collect_data_model()'
    ),
)
@click.option(
    '-v',
    '--verbose',
    is_flag=True,
    help='Flag to turn on debug logging. Default is not verbose.',
)
@click.pass_context
def collect_data_model(ctx, config, verbose=False, pipeline_step=None):
    """Collect data model output files to a single site-chunked output file."""
    config = BaseCLI.from_config_preflight(
        ModuleName.COLLECT_DATA_MODEL,
        ctx=ctx,
        config=config,
        verbose=verbose,
        pipeline_step=pipeline_step,
    )

    log_file = config.get('log_file', None)
    log_level = config.get('log_level', 'INFO')
    log_arg_str = f'"nsrdb", log_level="{log_level}"'

    config['n_chunks'] = config.get('n_chunks', 1)
    config['n_writes'] = config.get('n_writes', 1)
    config['final'] = config.get('final', False)
    n_files_tot = len(NSRDB.OUTS)
    n_files_default = (0, 1, 3, 4, 6)  # all files minus irrad and clearsky
    i_files = config.get('collect_files', n_files_default)
    fnames = sorted(NSRDB.OUTS.keys())
    name = ctx.obj['NAME']
    if config['final']:
        i_files = range(n_files_tot)

    if config['final'] and config['n_chunks'] != 1:
        msg = 'collect-data-model was marked as final but n_chunks != 1'
        logger.error(msg)
        raise ValueError(msg)

    for i_chunk in range(config['n_chunks']):
        for i_fname in i_files:
            if log_file is not None:
                log_file_i = log_file.replace('.log', f'_{i_fname}.log')
                log_arg_str_i = f'{log_arg_str}, log_file="{log_file_i}"'
                ctx.obj['LOG_ARG_STR'] = log_arg_str_i

            config['final_file_name'] = name
            config['i_chunk'] = i_chunk
            config['i_fname'] = i_fname
            fn_tag = fnames[i_fname].split('_')[1]
            config['job_name'] = f'{name}_{i_fname}_{fn_tag}_{i_chunk}'
            ctx.obj['NAME'] = config['job_name']
            ctx.obj['FUN_STR'] = get_fun_call_str(
                NSRDB.collect_data_model, config
            )

            BaseCLI.kickoff_job(ModuleName.COLLECT_DATA_MODEL, config, ctx)


@main.group(invoke_without_command=True)
@click.option(
    '--config',
    '-c',
    type=CONFIG_TYPE,
    required=True,
    help='Path to config file or dict with kwargs for NSRDB.all_sky()',
)
@click.option(
    '-v',
    '--verbose',
    is_flag=True,
    help='Flag to turn on debug logging. Default is not verbose.',
)
@click.pass_context
def collect_daily(ctx, config, verbose=False, pipeline_step=None):
    """Collect daily data model output files from a directory to a single
    file"""

    config = BaseCLI.from_config_preflight(
        ModuleName.COLLECT_DAILY,
        ctx=ctx,
        config=config,
        verbose=verbose,
        pipeline_step=pipeline_step,
    )

    ctx.obj['FUN_STR'] = get_fun_call_str(Collector.collect_daily, config)
    BaseCLI.kickoff_job(ModuleName.COLLECT_DAILY, config, ctx)


@main.group(invoke_without_command=True)
@click.option(
    '--config',
    '-c',
    type=CONFIG_TYPE,
    required=True,
    help='Path to config file or dict with kwargs for NSRDB.all_sky()',
)
@click.option(
    '-v',
    '--verbose',
    is_flag=True,
    help='Flag to turn on debug logging. Default is not verbose.',
)
@click.pass_context
def collect_flist(ctx, config, verbose=False, pipeline_step=None):
    """Run the file collection method with explicitly defined flist."""

    config = BaseCLI.from_config_preflight(
        ModuleName.COLLECT_FLIST,
        ctx=ctx,
        config=config,
        verbose=verbose,
        pipeline_step=pipeline_step,
    )

    ctx.obj['FUN_STR'] = get_fun_call_str(Collector.collect_flist, config)
    BaseCLI.kickoff_job(ModuleName.COLLECT_FLIST, config, ctx)


@main.group()
@click.option(
    '--config',
    '-c',
    type=CONFIG_TYPE,
    required=True,
    help='Path to config file or dict with kwargs for NSRDB.all_sky()',
)
@click.option(
    '-v',
    '--verbose',
    is_flag=True,
    help='Flag to turn on debug logging. Default is not verbose.',
)
@click.pass_context
def collect_final(ctx, config, verbose=False, pipeline_step=None):
    """Collect chunked files with final data into final full files."""

    config = BaseCLI.from_config_preflight(
        ModuleName.COLLECT_FINAL,
        ctx=ctx,
        config=config,
        verbose=verbose,
        pipeline_step=pipeline_step,
    )

    log_level = config.get('log_level', 'INFO')
    log_arg_str = f'"nsrdb", log_level="{log_level}"'
    name = ctx.obj['NAME']
    n_files = len(NSRDB.OUTS)
    for i_fname in range(n_files):
        log_file = f'{ctx.obj["out_dir"]}/final/final_collection_{i_fname}.log'
        log_arg_str_i = f'{log_arg_str}, log_file="{log_file}"'
        ctx.obj['LOG_ARG_STR'] = log_arg_str_i
        config['job_name'] = f'{name}_{i_fname}'
        ctx.obj['NAME'] = config['job_name']
        ctx.obj['FUN_STR'] = get_fun_call_str(NSRDB.collect_final, config)
        BaseCLI.kickoff_job(ModuleName.COLLECT_FINAL, config, ctx)


Pipeline.COMMANDS[ModuleName.DATA_MODEL] = data_model
Pipeline.COMMANDS[ModuleName.CLOUD_FILL] = cloud_fill
Pipeline.COMMANDS[ModuleName.ALL_SKY] = all_sky
Pipeline.COMMANDS[ModuleName.DAILY_ALL_SKY] = daily_all_sky
Pipeline.COMMANDS[ModuleName.ML_CLOUD_FILL] = ml_cloud_fill
Pipeline.COMMANDS[ModuleName.BLEND] = blend
Pipeline.COMMANDS[ModuleName.AGGREGATE] = aggregate
Pipeline.COMMANDS[ModuleName.COLLECT_DATA_MODEL] = collect_data_model
Pipeline.COMMANDS[ModuleName.COLLECT_FINAL] = collect_final
Pipeline.COMMANDS[ModuleName.COLLECT_FLIST] = collect_flist
Pipeline.COMMANDS[ModuleName.COLLECT_DAILY] = collect_daily
Pipeline.COMMANDS[ModuleName.COLLECT_BLENDED] = collect_blended
Pipeline.COMMANDS[ModuleName.COLLECT_AGG] = collect_aggregation


if __name__ == '__main__':
    try:
        main(obj={})
    except Exception:
        logger.error('Error running NSRDB CLI.')
        raise
