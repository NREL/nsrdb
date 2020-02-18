# -*- coding: utf-8 -*-
"""NSRDB Command Line Interface (CLI).

Created on Mon Oct 21 15:39:01 2019

@author: gbuster
"""
import os
import json
import logging
import click
from nsrdb.main import NSRDB
from nsrdb.utilities.cli_dtypes import STR, INT, DICT, STRLIST
from nsrdb.utilities.file_utils import safe_json_load
from nsrdb.utilities.execution import SLURM
from nsrdb.pipeline.status import Status
from nsrdb.pipeline.pipeline import Pipeline
from nsrdb.file_handlers.collection import Collector


logger = logging.getLogger(__name__)


@click.group()
@click.pass_context
def main(ctx):
    """NSRDB processing CLI."""
    ctx.ensure_object(dict)


@main.command()
@click.option('--config_file', '-c', required=True,
              type=click.Path(exists=True),
              help='NSRDB pipeline configuration json file.')
@click.option('--cancel', is_flag=True,
              help='Flag to cancel all jobs associated with a given pipeline.')
@click.option('--monitor', is_flag=True,
              help='Flag to monitor pipeline jobs continuously. '
              'Default is not to monitor (kick off jobs and exit).')
@click.pass_context
def pipeline(ctx, config_file, cancel, monitor):
    """NSRDB pipeline from a pipeline config file."""

    ctx.ensure_object(dict)
    if cancel:
        Pipeline.cancel_all(config_file)
    else:
        Pipeline.run(config_file, monitor=monitor)


@main.command()
@click.option('--config_file', '-c', required=True,
              type=click.Path(exists=True),
              help='Filepath to config file.')
@click.option('--command', '-cmd', type=str, required=True,
              help='NSRDB CLI command string.')
@click.pass_context
def config(ctx, config_file, command):
    """NSRDB processing CLI from config json file."""

    run_config = safe_json_load(config_file)

    direct_args = run_config.pop('direct')
    eagle_args = run_config.pop('eagle')
    cmd_args = run_config.pop(command)

    # replace any args with higher priority entries in command dict
    for k in eagle_args.keys():
        if k in cmd_args:
            eagle_args[k] = cmd_args[k]
    for k in direct_args.keys():
        if k in cmd_args:
            direct_args[k] = cmd_args[k]

    name = direct_args['name']
    ctx.obj['YEAR'] = direct_args['year']
    ctx.obj['NSRDB_GRID'] = direct_args['nsrdb_grid']
    ctx.obj['NSRDB_FREQ'] = direct_args['nsrdb_freq']
    ctx.obj['VAR_META'] = direct_args.get('var_meta', None)
    ctx.obj['OUT_DIR'] = direct_args['out_dir']
    ctx.obj['LOG_LEVEL'] = direct_args['log_level']

    if command == 'data-model':
        doy_range = cmd_args['doy_range']
        for doy in range(doy_range[0], doy_range[1]):
            ctx.obj['NAME'] = name + '_{}'.format(doy)
            ctx.invoke(data_model, doy=doy,
                       var_list=cmd_args.get('var_list', None),
                       factory_kwargs=cmd_args.get('factory_kwargs', None))
            ctx.invoke(eagle, **eagle_args)

    elif command == 'collect-data-model':
        n_chunks = cmd_args['n_chunks']
        for i_chunk in range(n_chunks):
            for i_fname in range(3):
                ctx.obj['NAME'] = name + '_{}_{}'.format(i_fname, i_chunk)
                ctx.invoke(collect_data_model,
                           daily_dir=cmd_args['daily_dir'],
                           n_chunks=n_chunks, i_chunk=i_chunk, i_fname=i_fname,
                           n_workers=cmd_args['n_workers'])
                ctx.invoke(eagle, **eagle_args)

    elif command == 'cloud-fill':
        n_chunks = cmd_args['n_chunks']
        for i_chunk in range(n_chunks):
            ctx.obj['NAME'] = name + '_{}'.format(i_chunk)
            ctx.invoke(cloud_fill, i_chunk=i_chunk,
                       col_chunk=cmd_args['col_chunk'])
            ctx.invoke(eagle, **eagle_args)

    elif command == 'all-sky':
        n_chunks = cmd_args['n_chunks']
        for i_chunk in range(n_chunks):
            ctx.obj['NAME'] = name + '_{}'.format(i_chunk)
            ctx.invoke(all_sky, i_chunk=i_chunk)
            ctx.invoke(eagle, **eagle_args)

    elif command == 'collect-final':
        for i_fname in range(4):
            ctx.obj['NAME'] = name + '_{}'.format(i_fname)
            ctx.invoke(collect_final, collect_dir=cmd_args['collect_dir'],
                       i_fname=i_fname)
            ctx.invoke(eagle, **eagle_args)

    else:
        raise KeyError('Command not recognized: "{}"'.format(command))


@main.group()
@click.option('--name', '-n', default='NSRDB', type=str,
              help='Job and node name.')
@click.option('--year', '-y', default=None, type=INT,
              help='Year of analysis.')
@click.option('--nsrdb_grid', '-g', default=None, type=STR,
              help='File path to NSRDB meta data grid.')
@click.option('--nsrdb_freq', '-f', default=None, type=STR,
              help='NSRDB frequency (e.g. "5min", "30min").')
@click.option('--var_meta', '-vm', default=None, type=STR,
              help='CSV file or dataframe containing meta data for all NSRDB '
              'variables. Defaults to the NSRDB var meta csv in git repo.')
@click.option('--out_dir', '-od', default=None, type=STR,
              help='Output directory.')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def direct(ctx, name, year, nsrdb_grid, nsrdb_freq, var_meta,
           out_dir, verbose):
    """NSRDB direct processing CLI (no config file)."""

    ctx.obj['NAME'] = name
    ctx.obj['YEAR'] = year
    ctx.obj['NSRDB_GRID'] = nsrdb_grid
    ctx.obj['NSRDB_FREQ'] = nsrdb_freq
    ctx.obj['VAR_META'] = var_meta
    ctx.obj['OUT_DIR'] = out_dir

    if verbose:
        ctx.obj['LOG_LEVEL'] = 'DEBUG'
    else:
        ctx.obj['LOG_LEVEL'] = 'INFO'


@main.group(invoke_without_command=True)
@click.option('--collect_dir', '-cd', type=str, required=True,
              help='Directory containing chunked files to collect from.')
@click.option('--f_out', '-fo', type=str, required=True,
              help='Full output filepath.')
@click.option('--dsets', '-ds', type=STRLIST, required=True,
              help='List of dataset names to collect.')
@click.option('--flist', '-fl', default=None, type=STRLIST,
              help='Optional list of filenames in collect_dir to collect. '
              'Using this option will superscede the default behavior of '
              'collecting daily data model outputs in collect_dir.')
@click.option('-p', '--parallel', is_flag=True,
              help='Flag for parallel daily data model file collection.')
@click.pass_context
def collect(ctx, collect_dir, f_out, dsets, flist, parallel):
    """Run the NSRDB file collection method."""
    ctx.ensure_object(dict)
    if flist is not None:
        for dset in dsets:
            Collector.collect_flist_lowmem(flist, collect_dir, f_out, dset)
    else:
        Collector.collect(collect_dir, f_out, dsets, parallel=parallel)


@direct.group()
@click.option('--doy', '-d', type=int, required=True,
              help='Integer day-of-year to run data model for.')
@click.option('--var_list', '-vl', type=STRLIST, required=False, default=None,
              help='Variables to process with the data model. None will '
              'default to all NSRDB variables.')
@click.option('--factory_kwargs', '-kw', type=DICT,
              required=False, default=None,
              help='Optional namespace of kwargs to use to initialize '
              'variable data handlers from the data models variable factory. '
              'Keyed by variable name. Values can be "source_dir", "handler", '
              'etc... source_dir for cloud variables can be a normal '
              'directory path or /directory/prefix*suffix where /directory/ '
              'can have more sub dirs.')
@click.pass_context
def data_model(ctx, doy, var_list, factory_kwargs):
    """Run the data model for a single day."""

    name = ctx.obj['NAME']
    year = ctx.obj['YEAR']
    out_dir = ctx.obj['OUT_DIR']
    nsrdb_grid = ctx.obj['NSRDB_GRID']
    nsrdb_freq = ctx.obj['NSRDB_FREQ']
    var_meta = ctx.obj['VAR_META']
    log_level = ctx.obj['LOG_LEVEL']

    if var_list is not None:
        var_list = json.dumps(var_list)
    if factory_kwargs is not None:
        factory_kwargs = json.dumps(factory_kwargs)

    date = NSRDB.doy_to_datestr(year, doy)
    fun_str = 'NSRDB.run_data_model'
    arg_str = ('"{}", "{}", "{}", freq="{}", var_list={}, '
               'log_level="{}", job_name="{}", factory_kwargs={}'
               .format(out_dir, date, nsrdb_grid, nsrdb_freq,
                       var_list, log_level, name, factory_kwargs))
    if var_meta is not None:
        arg_str += ', var_meta="{}"'.format(var_meta)
    ctx.obj['IMPORT_STR'] = 'from nsrdb.main import NSRDB'
    ctx.obj['FUN_STR'] = fun_str
    ctx.obj['ARG_STR'] = arg_str
    ctx.obj['COMMAND'] = 'data-model'


@direct.group()
@click.option('--daily_dir', '-d', type=str, required=True,
              help='Data model output directory to collect to out_dir.')
@click.option('--n_chunks', '-n', type=int, required=True,
              help='Number of chunks to collect into.')
@click.option('--i_chunk', '-ic', type=int, required=True,
              help='Chunk index.')
@click.option('--i_fname', '-if', type=int, required=True,
              help='Filename index (0: ancillary, 1: clouds, 2: sam vars).')
@click.option('--n_workers', '-w', type=int, required=True,
              help='Number of parallel workers to use.')
@click.pass_context
def collect_data_model(ctx, daily_dir, n_chunks, i_chunk, i_fname, n_workers):
    """Collect data model results into cohesive timseries file chunks."""

    name = ctx.obj['NAME']
    year = ctx.obj['YEAR']
    out_dir = ctx.obj['OUT_DIR']
    nsrdb_grid = ctx.obj['NSRDB_GRID']
    nsrdb_freq = ctx.obj['NSRDB_FREQ']
    var_meta = ctx.obj['VAR_META']
    log_level = ctx.obj['LOG_LEVEL']

    log_file = 'collect_{}_{}.log'.format(i_fname, i_chunk)

    fun_str = 'NSRDB.collect_data_model'
    arg_str = ('"{}", "{}", {}, "{}", n_chunks={}, i_chunk={}, '
               'i_fname={}, freq="{}", parallel={}, '
               'log_file="{}", log_level="{}", job_name="{}"'
               .format(daily_dir, out_dir, year, nsrdb_grid, n_chunks,
                       i_chunk, i_fname, nsrdb_freq, n_workers,
                       log_file, log_level, name))
    if var_meta is not None:
        arg_str += ', var_meta="{}"'.format(var_meta)
    ctx.obj['IMPORT_STR'] = 'from nsrdb.main import NSRDB'
    ctx.obj['FUN_STR'] = fun_str
    ctx.obj['ARG_STR'] = arg_str
    ctx.obj['COMMAND'] = 'collect-data-model'


@direct.group()
@click.option('--i_chunk', '-i', type=int, required=True,
              help='Chunked file index in out_dir to run cloud fill for.')
@click.option('--col_chunk', '-ch', type=int, required=True, default=10000,
              help='Column chunk to process at one time.')
@click.pass_context
def cloud_fill(ctx, i_chunk, col_chunk):
    """Gap fill a cloud data file."""

    name = ctx.obj['NAME']
    year = ctx.obj['YEAR']
    out_dir = ctx.obj['OUT_DIR']
    log_level = ctx.obj['LOG_LEVEL']
    var_meta = ctx.obj['VAR_META']
    log_file = 'cloud_fill_{}.log'.format(i_chunk)

    fun_str = 'NSRDB.gap_fill_clouds'
    arg_str = ('"{}", {}, {}, col_chunk={}, log_file="{}", '
               'log_level="{}", job_name="{}"'
               .format(out_dir, year, i_chunk, col_chunk,
                       log_file, log_level, name))
    if var_meta is not None:
        arg_str += ', var_meta="{}"'.format(var_meta)
    ctx.obj['IMPORT_STR'] = 'from nsrdb.main import NSRDB'
    ctx.obj['FUN_STR'] = fun_str
    ctx.obj['ARG_STR'] = arg_str
    ctx.obj['COMMAND'] = 'cloud-fill'


@direct.group()
@click.option('--i_chunk', '-i', type=int, required=True,
              help='Chunked file index in out_dir to run allsky for.')
@click.pass_context
def all_sky(ctx, i_chunk):
    """Run allsky for a single chunked file"""

    name = ctx.obj['NAME']
    year = ctx.obj['YEAR']
    out_dir = ctx.obj['OUT_DIR']
    nsrdb_grid = ctx.obj['NSRDB_GRID']
    nsrdb_freq = ctx.obj['NSRDB_FREQ']
    var_meta = ctx.obj['VAR_META']
    log_level = ctx.obj['LOG_LEVEL']

    log_file = 'all_sky_{}.log'.format(i_chunk)
    fun_str = 'NSRDB.run_all_sky'
    arg_str = ('"{}", {}, "{}", freq="{}", i_chunk={}, '
               'log_file="{}", log_level="{}", job_name="{}"'
               .format(out_dir, year, nsrdb_grid, nsrdb_freq, i_chunk,
                       log_file, log_level, name))
    if var_meta is not None:
        arg_str += ', var_meta="{}"'.format(var_meta)
    ctx.obj['IMPORT_STR'] = 'from nsrdb.main import NSRDB'
    ctx.obj['FUN_STR'] = fun_str
    ctx.obj['ARG_STR'] = arg_str
    ctx.obj['COMMAND'] = 'all-sky'


@direct.group()
@click.option('--collect_dir', '-d', type=str, required=True,
              help='Chunked directory to collect to out_dir.')
@click.option('--i_fname', '-if', type=int, required=True,
              help='Filename index (0: ancillary, 1: clouds, '
              '2: irrad, 3: sam vars).')
@click.pass_context
def collect_final(ctx, collect_dir, i_fname):
    """Collect chunked files with final data into final full files."""

    name = ctx.obj['NAME']
    year = ctx.obj['YEAR']
    out_dir = ctx.obj['OUT_DIR']
    nsrdb_grid = ctx.obj['NSRDB_GRID']
    nsrdb_freq = ctx.obj['NSRDB_FREQ']
    var_meta = ctx.obj['VAR_META']
    log_level = ctx.obj['LOG_LEVEL']

    log_file = 'final_collection_{}.log'.format(i_fname)

    fun_str = 'NSRDB.collect_final'
    arg_str = ('"{}", "{}", {}, "{}", freq="{}", '
               'i_fname={}, log_file="{}", log_level="{}", '
               'tmp=False, job_name="{}"'
               .format(collect_dir, out_dir, year, nsrdb_grid, nsrdb_freq,
                       i_fname, log_file, log_level, name))
    if var_meta is not None:
        arg_str += ', var_meta="{}"'.format(var_meta)
    ctx.obj['IMPORT_STR'] = 'from nsrdb.main import NSRDB'
    ctx.obj['FUN_STR'] = fun_str
    ctx.obj['ARG_STR'] = arg_str
    ctx.obj['COMMAND'] = 'collect-final'


@data_model.command()
@click.option('--alloc', '-a', required=True, type=STR,
              help='Eagle allocation account name.')
@click.option('--memory', '-mem', default=None, type=INT,
              help='Eagle node memory request in GB. Default is None')
@click.option('--walltime', '-wt', default=1.0, type=float,
              help='Eagle walltime request in hours. Default is 1.0')
@click.option('--feature', '-l', default=None, type=STR,
              help=('Additional flags for SLURM job. Format is "--qos=high" '
                    'or "--depend=[state:job_id]". Default is None.'))
@click.option('--stdout_path', '-sout', default=None, type=STR,
              help='Subprocess standard output path. Default is in out_dir.')
@click.pass_context
def eagle(ctx, alloc, memory, walltime, feature, stdout_path):
    """Eagle submission tool for reV supply curve aggregation."""

    name = ctx.obj['NAME']
    out_dir = ctx.obj['OUT_DIR']
    import_str = ctx.obj['IMPORT_STR']
    fun_str = ctx.obj['FUN_STR']
    arg_str = ctx.obj['ARG_STR']
    command = ctx.obj['COMMAND']

    if stdout_path is None:
        stdout_path = os.path.join(out_dir, 'stdout/')

    status = Status.retrieve_job_status(out_dir, command, name)
    if status == 'successful':
        msg = ('Job "{}" is successful in status json found in "{}", '
               'not re-running.'.format(name, out_dir))
        click.echo(msg)
        logger.info(msg)
    else:
        cmd = ("python -c '{import_str};{f}({a})'"
               .format(import_str=import_str, f=fun_str, a=arg_str))
        print('cmd: {}'.format(cmd))
        slurm = SLURM(cmd, alloc=alloc, memory=memory, walltime=walltime,
                      feature=feature, name=name, stdout_path=stdout_path)

        if slurm.id:
            msg = ('Kicked off job "{}" (SLURM jobid #{}) on Eagle.'
                   .format(name, slurm.id))
            Status.add_job(
                out_dir, command, name, replace=True,
                job_attrs={'job_id': slurm.id,
                           'hardware': 'eagle',
                           'out_dir': out_dir})
        else:
            msg = ('Was unable to kick off job "{}". '
                   'Please see the stdout error messages'
                   .format(name))
        print(msg)


collect_data_model.add_command(eagle)
cloud_fill.add_command(eagle)
all_sky.add_command(eagle)
collect_final.add_command(eagle)


if __name__ == '__main__':
    try:
        main(obj={})
    except Exception:
        logger.error('Error running NSRDB CLI.')
        raise
