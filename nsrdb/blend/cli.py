# -*- coding: utf-8 -*-
"""
NSRDB east-west blend command line interface (cli).
"""
import logging
import os
import time

import click
from rex.utilities.cli_dtypes import INT, STR
from rex.utilities.hpc import SLURM
from rex.utilities.loggers import init_logger

from nsrdb.blend.blend import Blender

logger = logging.getLogger(__name__)


@click.group(invoke_without_command=True)
@click.option('--name', '-n', default='blend', type=STR,
              help='Job name. Default is "blend".')
@click.option('--meta', '-m', type=STR, required=True,
              help='Filepath to final output blended meta data csv file.')
@click.option('--out_dir', '-od', type=STR, required=True,
              help='Directory to save blended output.')
@click.option('--east_dir', '-ed', type=STR, required=True,
              help='Source east directory.')
@click.option('--west_dir', '-wd', type=STR, required=True,
              help='Source west directory.')
@click.option('--out_fn', '-of', type=STR, default=None,
              help='Optional output filename.')
@click.option('--east_fn', '-ef', type=STR, default=None,
              help='Optional east filename (found in east_dir).')
@click.option('--west_fn', '-wf', type=STR, default=None,
              help='Optional west filename (found in west_dir).')
@click.option('--file_tag', '-t', type=STR, default=None,
              help='File tag found in files in east and west source dirs.')
@click.option('--map_col', '-mc', type=STR, default='gid_full',
              help='Column in the east and west meta data that map sites to '
              'the full meta_out gids.')
@click.option('--lon_seam', '-ls', type=float, default=-105.0,
              help='Vertical longitude seam at which data transitions from '
              'the western source to eastern, by default -105.0 (historical '
              'closest to nadir). 5min conus data (2019 onwards) is '
              'typically blended at -113.0 because the conus west satellite '
              'extent doesnt go that far east.')
@click.option('--chunk_size', '-cs', type=int, default=100000,
              help='Number of sites to read/write at a time.')
@click.option('--log_dir', '-ld', type=STR, default=None,
              help='Directory to save blend logs. Defaults to a logs/ '
              'directory in out_dir.')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def main(ctx, name, meta, out_dir, east_dir, west_dir, out_fn, east_fn,
         west_fn, file_tag, map_col, lon_seam, chunk_size, log_dir, verbose):
    """NSRDB East-West Blend CLI.

    Valid optional input combinations:
     - All filenames
     - Only file_tag
     - No filenames and no file_tag (for all file tags on HPC)

    Examples
    --------
    Here is an example shell script that calls the blend cli for 2020 full disc
    east (the largest dataset we have currently).

    ```
    #!/bin/bash
    META="/projects/pxs/reference_grids/nsrdb_meta_2km_full.csv"
    EDIR="/scratch/gbuster/nsrdb/nsrdb_full_east_2020/final/"
    WDIR="/scratch/gbuster/nsrdb/nsrdb_full_west_2020/final/"
    MAPCOL="gid_full"

    python -m nsrdb.blend.cli -n "blend0" -m $META -od "./" -ed $EDIR -wd $WDIR -t "ancillary_a" -mc $MAPCOL -ls -105.0 -cs 100000 -ld "./logs/" slurm -a "pxs" -wt 48.0 -l "--qos=normal" -mem "83" -sout "./logs/"
    python -m nsrdb.blend.cli -n "blend1" -m $META -od "./" -ed $EDIR -wd $WDIR -t "ancillary_b" -mc $MAPCOL -ls -105.0 -cs 100000 -ld "./logs/" slurm -a "pxs" -wt 48.0 -l "--qos=normal" -mem "83" -sout "./logs/"
    python -m nsrdb.blend.cli -n "blend2" -m $META -od "./" -ed $EDIR -wd $WDIR -t "clearsky" -mc $MAPCOL -ls -105.0 -cs 100000 -ld "./logs/" slurm -a "pxs" -wt 48.0 -l "--qos=normal" -mem "83" -sout "./logs/"
    python -m nsrdb.blend.cli -n "blend3" -m $META -od "./" -ed $EDIR -wd $WDIR -t "clouds" -mc $MAPCOL -ls -105.0 -cs 100000 -ld "./logs/" slurm -a "pxs" -wt 48.0 -l "--qos=normal" -mem "83" -sout "./logs/"
    python -m nsrdb.blend.cli -n "blend4" -m $META -od "./" -ed $EDIR -wd $WDIR -t "csp" -mc $MAPCOL -ls -105.0 -cs 100000 -ld "./logs/" slurm -a "pxs" -wt 48.0 -l "--qos=normal" -mem "83" -sout "./logs/"
    python -m nsrdb.blend.cli -n "blend5" -m $META -od "./" -ed $EDIR -wd $WDIR -t "irradiance" -mc $MAPCOL -ls -105.0 -cs 100000 -ld "./logs/" slurm -a "pxs" -wt 48.0 -l "--qos=normal" -mem "83" -sout "./logs/"
    python -m nsrdb.blend.cli -n "blend6" -m $META -od "./" -ed $EDIR -wd $WDIR -t "pv" -mc $MAPCOL -ls -105.0 -cs 100000 -ld "./logs/" slurm -a "pxs" -wt 48.0 -l "--qos=normal" -mem "83" -sout "./logs/"
    ```
    """

    if log_dir is None:
        log_dir = os.path.join(out_dir, 'logs/')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    ctx.ensure_object(dict)
    ctx.obj['NAME'] = name
    ctx.obj['META'] = meta
    ctx.obj['OUT_DIR'] = out_dir
    ctx.obj['EAST_DIR'] = east_dir
    ctx.obj['WEST_DIR'] = west_dir
    ctx.obj['OUT_FN'] = out_fn
    ctx.obj['EAST_FN'] = east_fn
    ctx.obj['WEST_FN'] = west_fn
    ctx.obj['FILE_TAG'] = file_tag
    ctx.obj['MAP_COL'] = map_col
    ctx.obj['LON_SEAM'] = lon_seam
    ctx.obj['CHUNK_SIZE'] = chunk_size
    ctx.obj['LOG_DIR'] = log_dir
    ctx.obj['VERBOSE'] = verbose

    if (out_fn is not None and east_fn is not None and west_fn is not None
            and file_tag is not None):
        logger.info('Filenames and file tags all specified. Using filenames.')
        file_tag = None

    if ctx.invoked_subcommand is None:
        t0 = time.time()
        log_level = 'DEBUG' if verbose else 'INFO'
        log_file = os.path.join(log_dir, '{}.log'.format(name))
        init_logger('nsrdb.blend', log_level=log_level, log_file=log_file)

        if (out_fn is None and east_fn is None and west_fn is None
                and file_tag is None):
            e = 'Filenames or file_tag must be specified for local blend job.'
            logger.error(e)
            raise RuntimeError(e)

        if out_fn is not None and east_fn is not None and west_fn is not None:
            out_fpath = os.path.join(out_dir, out_fn)
            east_fpath = os.path.join(east_dir, east_fn)
            west_fpath = os.path.join(west_dir, west_fn)
            Blender.blend_file(meta, out_fpath, east_fpath, west_fpath,
                               map_col=map_col, lon_seam=lon_seam,
                               chunk_size=chunk_size)
        else:
            Blender.blend_dir(meta, out_dir, east_dir, west_dir, file_tag,
                              out_fn=out_fn, map_col=map_col,
                              lon_seam=lon_seam, chunk_size=chunk_size)

        runtime = (time.time() - t0) / 60
        logger.info('NSRDB Blend complete. Time elapsed: {:.2f} min. Target '
                    'output dir: {}'.format(runtime, out_dir))


def get_node_cmds(name, meta, out_dir, east_dir, west_dir, out_fn,
                  east_fn, west_fn, file_tag, map_col, lon_seam,
                  chunk_size, log_dir, verbose,
                  default_tags=('ancillary', 'clouds', 'irradiance', 'sam')):
    """Get a CLI call command for the nsrdb blend cli."""

    if (out_fn is None and east_fn is None and west_fn is None
            and file_tag is None):
        names = []
        cmds = []
        for file_tag in default_tags:
            job_name = name + '_' + file_tag
            out = get_node_cmds(job_name, meta, out_dir, east_dir, west_dir,
                                out_fn, east_fn, west_fn, file_tag,
                                map_col, lon_seam, chunk_size, log_dir,
                                verbose)
            names += out[0]
            cmds += out[1]

    else:
        args = ('-n {name} '
                '-m {meta} '
                '-od {out_dir} '
                '-ed {east_dir} '
                '-wd {west_dir} '
                '-of {out_fn} '
                '-ef {east_fn} '
                '-wf {west_fn} '
                '-t {file_tag} '
                '-mc {map_col} '
                '-ls {lon_seam} '
                '-cs {chunk_size} '
                '-ld {log_dir} '
                )
        args = args.format(name=SLURM.s(name),
                           meta=SLURM.s(meta),
                           out_dir=SLURM.s(out_dir),
                           east_dir=SLURM.s(east_dir),
                           west_dir=SLURM.s(west_dir),
                           out_fn=SLURM.s(out_fn),
                           east_fn=SLURM.s(east_fn),
                           west_fn=SLURM.s(west_fn),
                           file_tag=SLURM.s(file_tag),
                           map_col=SLURM.s(map_col),
                           lon_seam=SLURM.s(lon_seam),
                           chunk_size=SLURM.s(chunk_size),
                           log_dir=SLURM.s(log_dir),
                           )
        if verbose:
            args += '-v '
        cmd = 'python -m nsrdb.blend.cli {}'.format(args)
        names = [name]
        cmds = [cmd]

    return names, cmds


@main.command()
@click.option('--alloc', '-a', default='pxs', type=str,
              help='SLURM allocation account name.')
@click.option('--walltime', '-wt', default=4.0, type=float,
              help='SLURM walltime request in hours. Default is 4.0')
@click.option('--feature', '-l', default=None, type=STR,
              help=('Additional flags for SLURM job. Format is "--qos=high" '
                    'or "--depend=[state:job_id]". Default is None.'))
@click.option('--memory', '-mem', default=None, type=INT,
              help='SLURM node memory request in GB. Default is None')
@click.option('--stdout_path', '-sout', default=None, type=STR,
              help='Subprocess standard output path. Default is in log_dir.')
@click.pass_context
def slurm(ctx, alloc, walltime, feature, memory, stdout_path):
    """Slurm (HPC) submission tool for reV supply curve aggregation."""

    name = ctx.obj['NAME']
    meta = ctx.obj['META']
    out_dir = ctx.obj['OUT_DIR']
    east_dir = ctx.obj['EAST_DIR']
    west_dir = ctx.obj['WEST_DIR']
    out_fn = ctx.obj['OUT_FN']
    east_fn = ctx.obj['EAST_FN']
    west_fn = ctx.obj['WEST_FN']
    file_tag = ctx.obj['FILE_TAG']
    map_col = ctx.obj['MAP_COL']
    lon_seam = ctx.obj['LON_SEAM']
    chunk_size = ctx.obj['CHUNK_SIZE']
    log_dir = ctx.obj['LOG_DIR']
    verbose = ctx.obj['VERBOSE']

    if stdout_path is None:
        stdout_path = os.path.join(log_dir, 'stdout/')

    names, cmds = get_node_cmds(name, meta, out_dir, east_dir, west_dir,
                                out_fn, east_fn, west_fn, file_tag, map_col,
                                lon_seam, chunk_size, log_dir, verbose)

    slurm_manager = SLURM()

    for name, cmd in zip(names, cmds):
        logger.info('Running NSRDB blend on SLURM with '
                    'node name "{}"'.format(name))
        out = slurm_manager.sbatch(cmd,
                                   alloc=alloc,
                                   memory=memory,
                                   walltime=walltime,
                                   feature=feature,
                                   name=name,
                                   stdout_path=stdout_path)[0]
        if out:
            msg = ('Kicked off nsrdb blend job "{}" (SLURM jobid #{}).'
                   .format(name, out))
        else:
            msg = ('Was unable to kick off nsrdb blend job "{}". '
                   'Please see the stdout error messages'
                   .format(name))
        click.echo(msg)
        logger.info(msg)


if __name__ == '__main__':
    try:
        main(obj={})
    except Exception:
        logger.exception('Error running nsrdb blend cli.')
        raise
