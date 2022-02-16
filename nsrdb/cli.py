# -*- coding: utf-8 -*-
"""NSRDB Command Line Interface (CLI).

Created on Mon Oct 21 15:39:01 2019

@author: gbuster
"""
import click
import json
import logging
import os

from rex.utilities.cli_dtypes import STR, INT, FLOAT, STRLIST
from rex.utilities.hpc import SLURM
from rex.utilities.loggers import init_logger
from rex.utilities.utilities import safe_json_load

from nsrdb.file_handlers.collection import Collector
from nsrdb.nsrdb import NSRDB
from nsrdb.pipeline import Status, NsrdbPipeline
from nsrdb.utilities.file_utils import ts_freq_check

logger = logging.getLogger(__name__)


class DictType(click.ParamType):
    """Dict click input argument type."""

    name = 'dict'

    @staticmethod
    def convert(value, param, ctx):
        """Convert to dict or return as None."""
        if isinstance(value, dict):
            return value
        elif isinstance(value, str):
            return json.loads(value)
        elif value is None:
            return None
        else:
            raise TypeError('Cannot recognize int type: {} {} {} {}'
                            .format(value, type(value), param, ctx))


DICT = DictType()


@click.group()
@click.pass_context
def main(ctx):
    """NSRDB processing CLI."""
    ctx.ensure_object(dict)


@main.command()
@click.option('--kwargs', '-kw', required=True, type=DICT,
              help='Argument dictionary. Needs to include year. '
              'e.g. \'{"year":2019, "freq":"5min"}\'. '
              '\n\nAvailable keys: '
              'year, freq, outdir (config file directory), '
              'sat (satellite, e.g. east/west), '
              'reg (region, e.g. RadC), '
              'basename (file prefix). '
              '\n\ndefault_kwargs = {"basename": "nsrdb", '
              '"freq": "5min", "sat": "east", '
              '"reg": "RadC", "outdir": "./"}')
@click.pass_context
def create_configs(ctx, kwargs):
    """NSRDB config file creation from templates."""

    ctx.ensure_object(dict)
    NSRDB.create_config_files(kwargs)


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
        NsrdbPipeline.cancel_all(config_file)
    else:
        NsrdbPipeline.run(config_file, monitor=monitor)


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

    if cmd_args is None:
        cmd_args = {}

    # replace any args with higher priority entries in command dict
    for k in eagle_args.keys():
        if k in cmd_args:
            eagle_args[k] = cmd_args[k]
    for k in direct_args.keys():
        if k in cmd_args:
            direct_args[k] = cmd_args[k]

    name = direct_args['name']
    ctx.obj['NAME'] = name
    ctx.obj['YEAR'] = direct_args['year']
    ctx.obj['NSRDB_GRID'] = direct_args['nsrdb_grid']
    ctx.obj['NSRDB_FREQ'] = direct_args['nsrdb_freq']
    ctx.obj['VAR_META'] = direct_args.get('var_meta', None)
    ctx.obj['OUT_DIR'] = direct_args['out_dir']
    ctx.obj['LOG_LEVEL'] = direct_args['log_level']
    ctx.obj['SLURM_MANAGER'] = SLURM()

    init_logger('nsrdb.cli', log_level=direct_args['log_level'], log_file=None)

    if command == 'data-model':
        ConfigRunners.run_data_model_config(ctx, name, cmd_args, eagle_args,
                                            direct_args)
    elif command == 'cloud-fill':
        ConfigRunners.run_cloud_fill_config(ctx, name, cmd_args, eagle_args)
    elif command == 'ml-cloud-fill':
        ConfigRunners.run_ml_cloud_fill_config(ctx, name, cmd_args,
                                               eagle_args, direct_args,
                                               run_config)
    elif command == 'all-sky':
        ConfigRunners.run_all_sky_config(ctx, name, cmd_args, eagle_args)
    elif command == 'daily-all-sky':
        ConfigRunners.run_daily_all_sky_config(ctx, name, cmd_args, eagle_args,
                                               direct_args, run_config)
    elif command == 'collect-data-model':
        ConfigRunners.run_collect_data_model_config(ctx, name, cmd_args,
                                                    eagle_args)
    elif command == 'collect-daily':
        ConfigRunners.run_collect_daily_config(ctx, name, cmd_args, eagle_args)
    elif command == 'collect-flist':
        ConfigRunners.run_collect_flist_config(ctx, name, cmd_args, eagle_args)
    elif command == 'collect-final':
        ConfigRunners.run_collect_final_config(ctx, name, cmd_args, eagle_args,
                                               direct_args)
    else:
        raise KeyError('Command not recognized: "{}"'.format(command))


class ConfigRunners:
    """Class to hold static methods that kickoff nsrdb modules from extracted
    nsrdb config objects"""

    @staticmethod
    def run_data_model_config(ctx, name, cmd_args, eagle_args, direct_args):
        """Run the daily data model processing code for each day of year.

        Parameters
        ----------
        ctx : click.pass_context
            Click context object.
        name : str
            Base jobname.
        cmd_args : dict
            Dictionary of kwargs from the nsrdb config file specifically for
            this command block.
        eagle_args : dict
            Dictionary of kwargs from the nsrdb config to make eagle submission
        direct_args : dict
            Dictionary of kwargs from the nsrdb config file under the "direct"
            key that are common to all command blocks.
        """
        doy_range = cmd_args['doy_range']
        for doy in range(doy_range[0], doy_range[1]):
            date = NSRDB.doy_to_datestr(direct_args['year'], doy)
            ctx.obj['NAME'] = name + '_data_model_{}_{}'.format(doy, date)
            max_workers_regrid = cmd_args.get('max_workers_regrid', None)

            ctx.invoke(data_model, doy=doy,
                       var_list=cmd_args.get('var_list', None),
                       dist_lim=cmd_args.get('dist_lim', 1.0),
                       factory_kwargs=cmd_args.get('factory_kwargs', None),
                       max_workers=cmd_args.get('max_workers', None),
                       max_workers_regrid=max_workers_regrid,
                       mlclouds=cmd_args.get('mlclouds', False))
            ctx.invoke(eagle, **eagle_args)

    @staticmethod
    def run_cloud_fill_config(ctx, name, cmd_args, eagle_args):
        """Run the cloud gap fill using simple legacy nearest neighbor methods.

        Parameters
        ----------
        ctx : click.pass_context
            Click context object.
        name : str
            Base jobname.
        cmd_args : dict
            Dictionary of kwargs from the nsrdb config file specifically for
            this command block.
        eagle_args : dict
            Dictionary of kwargs from the nsrdb config to make eagle submission
        """
        n_chunks = cmd_args['n_chunks']
        for i_chunk in range(n_chunks):
            ctx.obj['NAME'] = name + '_cloud_fill_{}'.format(i_chunk)
            ctx.invoke(cloud_fill, i_chunk=i_chunk,
                       col_chunk=cmd_args.get('col_chunk', None))
            ctx.invoke(eagle, **eagle_args)

    @staticmethod
    def run_ml_cloud_fill_config(ctx, name, cmd_args, eagle_args, direct_args,
                                 run_config):
        """Run the cloud gap fill using machine learning methods (phygnn).

        Parameters
        ----------
        ctx : click.pass_context
            Click context object.
        name : str
            Base jobname.
        cmd_args : dict
            Dictionary of kwargs from the nsrdb config file specifically for
            this command block.
        eagle_args : dict
            Dictionary of kwargs from the nsrdb config to make eagle submission
        direct_args : dict
            Dictionary of kwargs from the nsrdb config file under the "direct"
            key that are common to all command blocks.
        run_config : dict
            Dictionary of the full nsrdb config file. Used here to extract
            inputs from the data-model input block.
        """

        if 'doy_range' in cmd_args:
            doy_range = cmd_args['doy_range']
        else:
            doy_range = run_config['data-model']['doy_range']
        for doy in range(doy_range[0], doy_range[1]):
            date = NSRDB.doy_to_datestr(direct_args['year'], doy)
            ctx.obj['NAME'] = name + '_mlclouds_{}_{}'.format(doy, date)
            ctx.invoke(ml_cloud_fill, date=date,
                       fill_all=cmd_args.get('fill_all', False),
                       model_path=cmd_args.get('model_path', None),
                       col_chunk=cmd_args.get('col_chunk', None),
                       max_workers=cmd_args.get('max_workers', None))
            ctx.invoke(eagle, **eagle_args)

    @staticmethod
    def run_all_sky_config(ctx, name, cmd_args, eagle_args):
        """Run the all sky module to produce irradiance outputs.

        Parameters
        ----------
        ctx : click.pass_context
            Click context object.
        name : str
            Base jobname.
        cmd_args : dict
            Dictionary of kwargs from the nsrdb config file specifically for
            this command block.
        eagle_args : dict
            Dictionary of kwargs from the nsrdb config to make eagle submission
        """
        n_chunks = cmd_args['n_chunks']
        for i_chunk in range(n_chunks):
            ctx.obj['NAME'] = name + '_all_sky_{}'.format(i_chunk)
            ctx.invoke(all_sky, i_chunk=i_chunk,
                       col_chunk=cmd_args.get('col_chunk', 10))
            ctx.invoke(eagle, **eagle_args)

    @staticmethod
    def run_daily_all_sky_config(ctx, name, cmd_args, eagle_args,
                                 direct_args, run_config):
        """Run the all sky module to produce irradiance outputs using daily
        data model outputs as source.

        Parameters
        ----------
        ctx : click.pass_context
            Click context object.
        name : str
            Base jobname.
        cmd_args : dict
            Dictionary of kwargs from the nsrdb config file specifically for
            this command block.
        eagle_args : dict
            Dictionary of kwargs from the nsrdb config to make eagle submission
        direct_args : dict
            Dictionary of kwargs from the nsrdb config file under the "direct"
            key that are common to all command blocks.
        run_config : dict
            Dictionary of the full nsrdb config file. Used here to extract
            inputs from the data-model input block.
        """
        if 'doy_range' in cmd_args:
            doy_range = cmd_args['doy_range']
        else:
            doy_range = run_config['data-model']['doy_range']
        for doy in range(doy_range[0], doy_range[1]):
            date = NSRDB.doy_to_datestr(direct_args['year'], doy)
            ctx.obj['NAME'] = name + '_all_sky_{}_{}'.format(doy, date)
            ctx.invoke(daily_all_sky, date=date,
                       col_chunk=cmd_args.get('col_chunk', 500))
            ctx.invoke(eagle, **eagle_args)

    @staticmethod
    def run_collect_data_model_config(ctx, name, cmd_args, eagle_args):
        """Run collection of daily data model outputs to multiple files
        chunked by sites (n_chunks argument in cmd_args)

        Parameters
        ----------
        ctx : click.pass_context
            Click context object.
        name : str
            Base jobname.
        cmd_args : dict
            Dictionary of kwargs from the nsrdb config file specifically for
            this command block.
        eagle_args : dict
            Dictionary of kwargs from the nsrdb config to make eagle submission
        """
        n_chunks = cmd_args.get('n_chunks', 1)
        n_files_tot = len(NSRDB.OUTS)
        n_files_default = (0, 1, 3, 4, 6)  # all files minus irrad and clearsky
        i_files = cmd_args.get('collect_files', n_files_default)
        final = cmd_args.get('final', False)
        fnames = sorted(list(NSRDB.OUTS.keys()))
        if final:
            i_files = range(n_files_tot)

        if final and n_chunks != 1:
            msg = 'Collect data model was marked as final but n_chunks != 1'
            logger.error(msg)
            raise ValueError(msg)

        for i_chunk in range(n_chunks):
            for i_fname in i_files:
                final_file_name = name
                fn_tag = fnames[i_fname].split('_')[1]
                tag = '_{}_{}_{}'.format(i_fname, fn_tag, i_chunk)
                ctx.obj['NAME'] = name + tag

                ctx.invoke(collect_data_model,
                           n_chunks=n_chunks, i_chunk=i_chunk, i_fname=i_fname,
                           n_writes=cmd_args.get('n_writes', 1),
                           max_workers=cmd_args['max_workers'],
                           final=final, final_file_name=final_file_name)
                ctx.invoke(eagle, **eagle_args)

    @staticmethod
    def run_collect_daily_config(ctx, name, cmd_args, eagle_args):
        """Run full collection of all daily data model outputs to a single file

        Parameters
        ----------
        ctx : click.pass_context
            Click context object.
        name : str
            Base jobname.
        cmd_args : dict
            Dictionary of kwargs from the nsrdb config file specifically for
            this command block.
        eagle_args : dict
            Dictionary of kwargs from the nsrdb config to make eagle submission
        """
        ctx.obj['NAME'] = name + '_collect_daily'
        ctx.invoke(collect_daily, collect_dir=cmd_args['collect_dir'],
                   fn_out=cmd_args['fn_out'], dsets=cmd_args['dsets'],
                   n_writes=cmd_args.get('n_writes', 1),
                   max_workers=cmd_args.get('max_workers', None),
                   eagle=True)
        ctx.invoke(eagle, **eagle_args)

    @staticmethod
    def run_collect_flist_config(ctx, name, cmd_args, eagle_args):
        """Run custom file collection.

        Parameters
        ----------
        ctx : click.pass_context
            Click context object.
        name : str
            Base jobname.
        cmd_args : dict
            Dictionary of kwargs from the nsrdb config file specifically for
            this command block.
        eagle_args : dict
            Dictionary of kwargs from the nsrdb config to make eagle submission
        """
        ctx.obj['NAME'] = name + '_collect_flist'
        ctx.invoke(collect_flist, flist=cmd_args['flist'],
                   collect_dir=cmd_args['collect_dir'],
                   fn_out=cmd_args['fn_out'], dsets=cmd_args['dsets'],
                   max_workers=cmd_args.get('max_workers', None), eagle=True)
        ctx.invoke(eagle, **eagle_args)

    @staticmethod
    def run_collect_final_config(ctx, name, cmd_args, eagle_args, direct_args):
        """Run final file collection.

        Parameters
        ----------
        ctx : click.pass_context
            Click context object.
        name : str
            Base jobname.
        cmd_args : dict
            Dictionary of kwargs from the nsrdb config file specifically for
            this command block.
        eagle_args : dict
            Dictionary of kwargs from the nsrdb config to make eagle submission
        direct_args : dict
            Dictionary of kwargs from the nsrdb config file under the "direct"
            key that are common to all command blocks.
        """
        def_dir = os.path.join(direct_args['out_dir'], 'collect/')
        n_files = len(NSRDB.OUTS)
        for i_fname in range(n_files):
            ctx.obj['NAME'] = name + '_{}'.format(i_fname)
            ctx.invoke(collect_final,
                       collect_dir=cmd_args.get('collect_dir', def_dir),
                       i_fname=i_fname)
            ctx.invoke(eagle, **eagle_args)


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
@click.option('--out_dir', '-od', type=STR, required=True,
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

    ts_freq_check(nsrdb_freq)

    if verbose:
        ctx.obj['LOG_LEVEL'] = 'DEBUG'
    else:
        ctx.obj['LOG_LEVEL'] = 'INFO'


@direct.group()
@click.option('--doy', '-d', type=int, required=True,
              help='Integer day-of-year to run data model for.')
@click.option('--var_list', '-vl', type=STRLIST, required=False, default=None,
              help='Variables to process with the data model. None will '
              'default to all NSRDB variables.')
@click.option('--dist_lim', '-dl', type=FLOAT, required=True, default=1.0,
              help='Return only neighbors within this distance during cloud '
              'regrid. The distance is in decimal degrees (more efficient '
              'than real distance). NSRDB sites further than this value from '
              'GOES data pixels will be warned and given missing cloud types '
              'and properties resulting in a full clearsky timeseries.')
@click.option('--factory_kwargs', '-kw', type=DICT,
              required=False, default=None,
              help='Optional namespace of kwargs to use to initialize '
              'variable data handlers from the data models variable factory. '
              'Keyed by variable name. Values can be "source_dir", "handler", '
              'etc... source_dir for cloud variables can be a normal '
              'directory path or /directory/prefix*suffix where /directory/ '
              'can have more sub dirs.')
@click.option('--max_workers', '-w', type=INT, default=None,
              help='Number of workers to use in parallel.')
@click.option('--max_workers_regrid', '-mwr', type=INT, default=None,
              help='Number of workers to use in parallel for the '
                   'cloud regrid algorithm.')
@click.option('-ml', '--mlclouds', is_flag=True,
              help='Flag to process additional variables if mlclouds gap fill'
              'is going to be run after the data_model step.')
@click.pass_context
def data_model(ctx, doy, var_list, dist_lim, factory_kwargs, max_workers,
               max_workers_regrid, mlclouds):
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
        factory_kwargs = factory_kwargs.replace('true', 'True')
        factory_kwargs = factory_kwargs.replace('false', 'False')
        factory_kwargs = factory_kwargs.replace('null', 'None')

    log_file = 'data_model/data_model.log'
    date = NSRDB.doy_to_datestr(year, doy)
    fun_str = 'NSRDB.run_data_model'
    arg_str = ('"{}", "{}", "{}", freq="{}", var_list={}, '
               'dist_lim={}, max_workers={}, '
               'max_workers_regrid={}, '
               'log_level="{}", log_file="{}", '
               'job_name="{}", factory_kwargs={}, mlclouds={}'
               .format(out_dir, date, nsrdb_grid, nsrdb_freq,
                       var_list, dist_lim, max_workers, max_workers_regrid,
                       log_level, log_file, name,
                       factory_kwargs, mlclouds))

    if var_meta is not None:
        arg_str += ', var_meta="{}"'.format(var_meta)
    ctx.obj['IMPORT_STR'] = 'from nsrdb.nsrdb import NSRDB'
    ctx.obj['FUN_STR'] = fun_str
    ctx.obj['ARG_STR'] = arg_str
    ctx.obj['COMMAND'] = 'data-model'


@direct.group()
@click.option('--i_chunk', '-i', type=int, required=True,
              help='Chunked file index in out_dir to run cloud fill for.')
@click.option('--col_chunk', '-ch', type=INT, required=True, default=None,
              help='Optional chunking method to gap fill one column chunk at '
              'a time to reduce memory requirements. If provided, this should '
              'be an int specifying how many columns to work on at one time.')
@click.pass_context
def cloud_fill(ctx, i_chunk, col_chunk):
    """Gap fill a cloud data file."""

    name = ctx.obj['NAME']
    year = ctx.obj['YEAR']
    out_dir = ctx.obj['OUT_DIR']
    log_level = ctx.obj['LOG_LEVEL']
    var_meta = ctx.obj['VAR_META']
    log_file = 'gap_fill/cloud_fill_{}.log'.format(i_chunk)

    fun_str = 'NSRDB.gap_fill_clouds'
    arg_str = ('"{}", {}, {}, col_chunk={}, log_file="{}", '
               'log_level="{}", job_name="{}"'
               .format(out_dir, year, i_chunk, col_chunk,
                       log_file, log_level, name))
    if var_meta is not None:
        arg_str += ', var_meta="{}"'.format(var_meta)
    ctx.obj['IMPORT_STR'] = 'from nsrdb.nsrdb import NSRDB'
    ctx.obj['FUN_STR'] = fun_str
    ctx.obj['ARG_STR'] = arg_str
    ctx.obj['COMMAND'] = 'cloud-fill'


@direct.group()
@click.option('--date', '-d', type=str, required=True,
              help='Single day data model output to run cloud fill on.'
              'Must be str in YYYYMMDD format.')
@click.option('--fill_all', '-all', type=bool, default=False,
              help='Flag to fill all cloud properties for all timesteps where '
              'cloud_type is cloudy.')
@click.option('--model_path', '-mp', type=STR, default=None,
              help='Directory to load phygnn model from. This is typically '
              'a fpath to a .pkl file with an accompanying .json file in the '
              'same directory.')
@click.option('--col_chunk', '-ch', type=INT, required=True, default=None,
              help='Optional chunking method to gap fill one column chunk at '
              'a time to reduce memory requirements. If provided, this should '
              'be an int specifying how many columns to work on at one time.')
@click.option('--max_workers', '-mw', type=INT, required=True, default=None,
              help='Maximum workers to clean data in parallel. 1 is serial '
              'and None uses all available workers.')
@click.pass_context
def ml_cloud_fill(ctx, date, fill_all, model_path, col_chunk, max_workers):
    """Gap fill cloud properties in daily data model outputs using a physics
    guided neural network (phgynn)."""

    name = ctx.obj['NAME']
    out_dir = ctx.obj['OUT_DIR']
    log_level = ctx.obj['LOG_LEVEL']
    var_meta = ctx.obj['VAR_META']
    log_file = 'gap_fill/cloud_fill_{}.log'.format(date)

    if isinstance(model_path, str):
        model_path = '"{}"'.format(model_path)

    fun_str = 'NSRDB.ml_cloud_fill'
    arg_str = ('"{}", "{}", fill_all={}, model_path={}, log_file="{}", '
               'log_level="{}", job_name="{}", col_chunk={}, max_workers={}'
               .format(out_dir, date, fill_all, model_path, log_file,
                       log_level, name, col_chunk, max_workers))
    if var_meta is not None:
        arg_str += ', var_meta="{}"'.format(var_meta)
    ctx.obj['IMPORT_STR'] = 'from nsrdb.nsrdb import NSRDB'
    ctx.obj['FUN_STR'] = fun_str
    ctx.obj['ARG_STR'] = arg_str
    ctx.obj['COMMAND'] = 'ml-cloud-fill'


@direct.group()
@click.option('--i_chunk', '-i', type=int, required=True,
              help='Chunked file index in out_dir to run allsky for.')
@click.option('--col_chunk', '-ch', type=int, required=True, default=10,
              help='Chunking method to run all sky one column chunk at a time '
              'to reduce memory requirements. This is an integer specifying '
              'how many columns to work on at one time.')
@click.pass_context
def all_sky(ctx, i_chunk, col_chunk):
    """Run allsky for a single chunked file"""

    name = ctx.obj['NAME']
    year = ctx.obj['YEAR']
    out_dir = ctx.obj['OUT_DIR']
    nsrdb_grid = ctx.obj['NSRDB_GRID']
    nsrdb_freq = ctx.obj['NSRDB_FREQ']
    var_meta = ctx.obj['VAR_META']
    log_level = ctx.obj['LOG_LEVEL']

    log_file = 'all_sky/all_sky_{}.log'.format(i_chunk)
    fun_str = 'NSRDB.run_all_sky'
    arg_str = ('"{}", {}, "{}", freq="{}", i_chunk={}, col_chunk={}, '
               'log_file="{}", log_level="{}", job_name="{}"'
               .format(out_dir, year, nsrdb_grid, nsrdb_freq, i_chunk,
                       col_chunk, log_file, log_level, name))
    if var_meta is not None:
        arg_str += ', var_meta="{}"'.format(var_meta)
    ctx.obj['IMPORT_STR'] = 'from nsrdb.nsrdb import NSRDB'
    ctx.obj['FUN_STR'] = fun_str
    ctx.obj['ARG_STR'] = arg_str
    ctx.obj['COMMAND'] = 'all-sky'


@direct.group()
@click.option('--date', '-d', type=str, required=True,
              help='Single day data model output to run cloud fill on.'
              'Must be str in YYYYMMDD format.')
@click.option('--col_chunk', '-ch', type=int, required=True, default=500,
              help='Chunking method to run all sky one column chunk at a time '
              'to reduce memory requirements. This is an integer specifying '
              'how many columns to work on at one time.')
@click.pass_context
def daily_all_sky(ctx, date, col_chunk):
    """Run allsky for a single day using daily data model output files as
    source data"""

    name = ctx.obj['NAME']
    year = ctx.obj['YEAR']
    out_dir = ctx.obj['OUT_DIR']
    nsrdb_grid = ctx.obj['NSRDB_GRID']
    nsrdb_freq = ctx.obj['NSRDB_FREQ']
    var_meta = ctx.obj['VAR_META']
    log_level = ctx.obj['LOG_LEVEL']

    log_file = 'all_sky/all_sky_{}.log'.format(date)
    fun_str = 'NSRDB.run_daily_all_sky'
    arg_str = ('"{}", {}, "{}", "{}", freq="{}", col_chunk={}, '
               'log_file="{}", log_level="{}", job_name="{}"'
               .format(out_dir, year, nsrdb_grid, date, nsrdb_freq, col_chunk,
                       log_file, log_level, name))
    if var_meta is not None:
        arg_str += ', var_meta="{}"'.format(var_meta)
    ctx.obj['IMPORT_STR'] = 'from nsrdb.nsrdb import NSRDB'
    ctx.obj['FUN_STR'] = fun_str
    ctx.obj['ARG_STR'] = arg_str
    ctx.obj['COMMAND'] = 'daily-all-sky'


@direct.group()
@click.option('--n_chunks', '-n', type=int, required=True,
              help='Number of chunks to collect into.')
@click.option('--i_chunk', '-ic', type=int, required=True,
              help='Chunk index.')
@click.option('--i_fname', '-if', type=int, required=True,
              help='Filename index: 0: ancillary_a, 1: ancillary_b, '
              '2: clearsky, 3: clouds, 4: csp, 5: irrad, 6: pv.')
@click.option('--n_writes', '-nw', type=int, default=1,
              help='Number of file list divisions to write per dataset. For '
              'example, if ghi and dni are being collected and n_writes is '
              'set to 2, half of the source ghi files will be collected at '
              'once and then written, then the second half of ghi files, '
              'then dni.')
@click.option('--max_workers', '-w', type=INT, default=None,
              help='Number of parallel workers to use.')
@click.option('-f', '--final', is_flag=True,
              help='Flag for final collection. Will put collected files in '
              'the final directory instead of in the collect directory.')
@click.option('--final_file_name', '-pn', type=STR, default=None,
              help='Final file name for filename outputs if this is the '
              'terminal job. None will default to the name in ctx which is '
              'usually the slurm job name.')
@click.pass_context
def collect_data_model(ctx, n_chunks, i_chunk, i_fname, n_writes,
                       max_workers, final, final_file_name):
    """Collect data model results into cohesive timseries file chunks."""

    name = ctx.obj['NAME']
    year = ctx.obj['YEAR']
    out_dir = ctx.obj['OUT_DIR']
    nsrdb_grid = ctx.obj['NSRDB_GRID']
    nsrdb_freq = ctx.obj['NSRDB_FREQ']
    var_meta = ctx.obj['VAR_META']
    log_level = ctx.obj['LOG_LEVEL']

    if final_file_name is None:
        final_file_name = name

    fnames = sorted(list(NSRDB.OUTS.keys()))
    fn_tag = fnames[i_fname].split('_')[1]
    log_file = ('collect/collect_{}_{}_{}.log'
                .format(i_fname, fn_tag, i_chunk))

    fun_str = 'NSRDB.collect_data_model'
    arg_str = ('"{}", {}, "{}", n_chunks={}, i_chunk={}, '
               'i_fname={}, freq="{}", n_writes={}, max_workers={}, '
               'log_file="{}", log_level="{}", job_name="{}", '
               'final={}, final_file_name="{}"'
               .format(out_dir, year, nsrdb_grid, n_chunks,
                       i_chunk, i_fname, nsrdb_freq, n_writes, max_workers,
                       log_file, log_level, name, final, final_file_name))
    if var_meta is not None:
        arg_str += ', var_meta="{}"'.format(var_meta)
    ctx.obj['IMPORT_STR'] = 'from nsrdb.nsrdb import NSRDB'
    ctx.obj['FUN_STR'] = fun_str
    ctx.obj['ARG_STR'] = arg_str
    ctx.obj['COMMAND'] = 'collect-data-model'


@direct.group(invoke_without_command=True)
@click.option('--collect_dir', '-cd', type=str, required=True,
              help='Directory containing chunked files to collect from.')
@click.option('--fn_out', '-fo', type=str, required=True,
              help='Output filename to be saved in out_dir.')
@click.option('--dsets', '-ds', type=STRLIST, required=True,
              help='List of dataset names to collect.')
@click.option('--n_writes', '-nw', type=int, default=1,
              help='Number of file list divisions to write per dataset. For '
              'example, if ghi and dni are being collected and n_writes is '
              'set to 2, half of the source ghi files will be collected at '
              'once and then written, then the second half of ghi files, '
              'then dni.')
@click.option('--max_workers', '-w', type=INT, default=None,
              help='Number of parallel workers to use.')
@click.option('-e', '--eagle', is_flag=True,
              help='Flag for that this is being used to pass commands to '
              'an Eagle call.')
@click.pass_context
def collect_daily(ctx, collect_dir, fn_out, dsets, n_writes, max_workers,
                  eagle):
    """Run the NSRDB file collection method on a specific daily directory
    for specific datasets to a single output file."""

    name = ctx.obj['NAME']
    out_dir = ctx.obj['OUT_DIR']
    var_meta = ctx.obj['VAR_META']
    log_level = ctx.obj['LOG_LEVEL']
    log_file = os.path.join(out_dir, 'collect_daily/{}.log'.format(name))

    fp_out = os.path.join(out_dir, fn_out)

    arg_str = ('"{}", "{}", {}, n_writes={}, max_workers={}, log_level="{}", '
               'log_file="{}", write_status=True, job_name="{}"'
               .format(collect_dir, fp_out, json.dumps(dsets), n_writes,
                       max_workers, log_level, log_file, name))
    if var_meta is not None:
        arg_str += ', var_meta="{}"'.format(var_meta)

    ctx.obj['IMPORT_STR'] = ('from nsrdb.file_handlers.collection '
                             'import Collector')
    ctx.obj['FUN_STR'] = 'Collector.collect_daily'
    ctx.obj['ARG_STR'] = arg_str
    ctx.obj['COMMAND'] = 'collect-daily'

    if ctx.invoked_subcommand is None and not eagle:
        init_logger('nsrdb.file_handlers', log_level=log_level,
                    log_file=log_file)
        Collector.collect_daily(collect_dir, fp_out, dsets,
                                max_workers=max_workers,
                                var_meta=var_meta)


@direct.group(invoke_without_command=True)
@click.option('--flist', '-fl', type=STRLIST, required=True,
              help='Explicit list of filenames in collect_dir to collect. '
              'Using this option will superscede the default behavior of '
              'collecting daily data model outputs in collect_dir.')
@click.option('--collect_dir', '-cd', type=str, required=True,
              help='Directory containing chunked files to collect from.')
@click.option('--fn_out', '-fo', type=str, required=True,
              help='Output filename to be saved in out_dir.')
@click.option('--dsets', '-ds', type=STRLIST, required=True,
              help='List of dataset names to collect.')
@click.option('-e', '--eagle', is_flag=True,
              help='Flag for that this is being used to pass commands to '
              'an Eagle call.')
@click.pass_context
def collect_flist(ctx, flist, collect_dir, fn_out, dsets, eagle):
    """Run the NSRDB file collection method with explicitly defined flist."""

    name = ctx.obj['NAME']
    out_dir = ctx.obj['OUT_DIR']
    var_meta = ctx.obj['VAR_META']
    log_level = ctx.obj['LOG_LEVEL']
    log_file = os.path.join(out_dir, 'collect/{}.log'.format(name))

    fp_out = os.path.join(out_dir, fn_out)

    arg_str = ('{}, "{}", "{}", {} log_level="{}", '
               'log_file="{}", write_status=True, job_name="{}"'
               .format(json.dumps(flist), collect_dir, fp_out,
                       json.dumps(dsets), log_level, log_file, name))
    if var_meta is not None:
        arg_str += ', var_meta="{}"'.format(var_meta)
    ctx.obj['IMPORT_STR'] = ('from nsrdb.file_handlers.collection '
                             'import Collector')
    ctx.obj['FUN_STR'] = 'Collector.collect_flist_lowmem'
    ctx.obj['ARG_STR'] = arg_str
    ctx.obj['COMMAND'] = 'collect-flist'

    if ctx.invoked_subcommand is None and not eagle:
        init_logger('nsrdb.file_handlers', log_level=log_level,
                    log_file=log_file)
        for dset in dsets:
            Collector.collect_flist_lowmem(flist, collect_dir, fp_out, dset,
                                           var_meta=var_meta)


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

    log_file = 'final/final_collection_{}.log'.format(i_fname)

    fun_str = 'NSRDB.collect_final'
    arg_str = ('"{}", "{}", {}, "{}", freq="{}", '
               'i_fname={}, log_file="{}", log_level="{}", '
               'tmp=False, job_name="{}"'
               .format(collect_dir, out_dir, year, nsrdb_grid, nsrdb_freq,
                       i_fname, log_file, log_level, name))
    if var_meta is not None:
        arg_str += ', var_meta="{}"'.format(var_meta)
    ctx.obj['IMPORT_STR'] = 'from nsrdb.nsrdb import NSRDB'
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
    """Eagle submission tool for the NSRDB cli."""

    name = ctx.obj['NAME']
    out_dir = ctx.obj['OUT_DIR']
    import_str = ctx.obj['IMPORT_STR']
    fun_str = ctx.obj['FUN_STR']
    arg_str = ctx.obj['ARG_STR']
    command = ctx.obj['COMMAND']

    if 'SLURM_MANAGER' not in ctx.obj:
        ctx.obj['SLURM_MANAGER'] = SLURM()

    slurm_manager = ctx.obj['SLURM_MANAGER']

    if stdout_path is None:
        stdout_path = os.path.join(out_dir, 'stdout/')

    status = Status.retrieve_job_status(out_dir, command, name,
                                        hardware='eagle',
                                        subprocess_manager=slurm_manager)

    msg = 'NSRDB CLI failed to submit jobs!'
    if status == 'successful':
        msg = ('Job "{}" is successful in status json found in "{}", '
               'not re-running.'.format(name, out_dir))
    elif 'fail' not in str(status).lower() and status is not None:
        msg = ('Job "{}" was found with status "{}", not resubmitting'
               'not re-running.'.format(name, status))
    else:
        cmd = ("python -c '{import_str};{f}({a})'"
               .format(import_str=import_str, f=fun_str, a=arg_str))
        slurm_id = None
        out = slurm_manager.sbatch(cmd,
                                   alloc=alloc,
                                   memory=memory,
                                   walltime=walltime,
                                   feature=feature,
                                   name=name,
                                   stdout_path=stdout_path)[0]

        if out:
            slurm_id = out
            msg = ('Kicked off job "{}" (SLURM jobid #{}) on Eagle.'
                   .format(name, slurm_id))

        Status.add_job(
            out_dir, command, name, replace=True,
            job_attrs={'job_id': slurm_id,
                       'hardware': 'eagle',
                       'out_dir': out_dir})

    click.echo(msg)
    logger.info(msg)


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
