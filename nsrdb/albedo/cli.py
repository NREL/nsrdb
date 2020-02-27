"""
Command line interface for creating composite albedo data for a single day from
MODIS dry-land albedo and IMS snow data.

Mike Bannister
1/29/2020
"""
import click
import sys
import os
import logging
from datetime import datetime as dt
from datetime import timedelta

from nsrdb.utilities.execution import SLURM
from nsrdb.utilities.cli_dtypes import STR, INT

from nsrdb.utilities.loggers import init_logger
from nsrdb.albedo.albedo import CompositeAlbedoDay
from nsrdb.albedo.ims import get_dt

logger = logging.getLogger(__name__)


class Date(click.ParamType):
    """ Date argument parser and sanity checker """
    def convert(self, value, param, ctx):
        if len(value) == 7:
            # E.g., 2015001
            date = get_dt(int(value[:4]), int(value[4:]))
        elif len(value) == 8:
            # E.g., 20150531
            date = dt.strptime(value, '%Y%m%d')
        else:
            msg = ('Date must be provided in YYYYDDD or YYYYMMDD '
                   f'format (e.g. 2015012 or 20150305). {value} '
                   'was provided.')
            click.echo(msg)
            logger.error(msg)
            sys.exit(1)
        return date


def _setup_paths(ctx):
    """ Handle paths and path overrides """
    # Verify path is set
    if ctx.obj['path'] is None and (ctx.obj['mpath'] is None
                                    or ctx.obj['ipath'] is None
                                    or ctx.obj['apath'] is None):
        msg = ('Paths for MODIS, IMS, and composite albedo data '
               'must be set together using --path, or '
               'individually using --modis-path, --ims-path, and '
               '--albedo-path.')
        click.echo(msg)
        logger.error(msg)
        sys.exit(1)

    # Over ride general path with specifics
    if ctx.obj['path'] is not None:
        if ctx.obj['mpath'] is None:
            ctx.obj['mpath'] = ctx.obj['path']
        if ctx.obj['ipath'] is None:
            ctx.obj['ipath'] = ctx.obj['path']
        if ctx.obj['apath'] is None:
            ctx.obj['apath'] = ctx.obj['path']


@click.group()
@click.option('--path', '-p', type=click.Path(exists=True),
              help='Path for all data files. This may be partially '
              'overridden by the other path arguments.')
@click.option('--modis-path', '-m', type=click.Path(exists=True),
              help='Path of/for MODIS data files')
@click.option('--ims-path', '-i', type=click.Path(exists=True),
              help='Path of/for IMS data/metadata files')
@click.option('--albedo-path', '-a', type=click.Path(exists=True),
              help='Path to save composite albedo data files')
@click.option('--log-level',
              type=click.Choice(['DEBUG', 'INFO', 'WARNING',
                                 'ERROR', 'CRITICAL'],
                                case_sensitive=False),
              default='INFO',
              help='Logging level')
@click.option('--log-file', type=click.Path(), default='nsrdb.albedo.log',
              help='Logging output file.')
@click.option('--tiff', '-t', is_flag=True, default=False,
              help='Create TIFF and world file in addition to h5 file.')
@click.pass_context
def main(ctx, path, modis_path, ims_path, albedo_path, log_level, log_file,
         tiff):
    """
    Create composite albedo data for one day using MODIS and IMS data sets or
    convert existing albedo h5 file to TIFF with world file.
    """
    ctx.obj = {}
    ctx.obj['path'] = path
    ctx.obj['mpath'] = modis_path
    ctx.obj['ipath'] = ims_path
    ctx.obj['apath'] = albedo_path

    log_level = log_level.upper()
    ctx.obj['log_level'] = log_level
    ctx.obj['log_file'] = log_file

    ctx.obj['tiff'] = tiff

    init_logger('nsrdb.albedo', log_file=log_file, log_level=log_level)
    init_logger('nsrdb.utilities', log_file=log_file,
                log_level=log_level)


@main.command()
@click.argument('date', type=Date())
@click.option('--modis-shape', nargs=2, type=int, default=None,
              help='Shape of MODIS data, in format: XXX YYY')
@click.option('--ims-shape', nargs=2, type=int, default=None,
              help='Shape of IMS data, in format: XXX YYY')
@click.pass_context
def singleday(ctx, date, modis_shape, ims_shape):
    """
    Calculate composite albedo for a single day. Date is in YYYYDDD or
    YYYYMMDD format
    """
    _setup_paths(ctx)
    click.echo(f'Calculating single day composite albedo on {date}.')
    logger.info(f'Calculating single day composite albedo on {date}.')
    logger.debug(f'Click context: {ctx.obj}')

    # Override data shapes, used for testing
    _kwargs = {}
    if modis_shape:
        _kwargs['modis_shape'] = modis_shape
        logger.info(f'Using MODIS data shape of {modis_shape}')
    if ims_shape:
        _kwargs['ims_shape'] = ims_shape
        logger.info(f'Using IMS data shape of {ims_shape}')

    cad = CompositeAlbedoDay.run(date, ctx.obj['mpath'], ctx.obj['ipath'],
                                 ctx.obj['apath'], **_kwargs)
    cad.write_albedo()
    if ctx.obj['tiff']:
        cad.write_tiff()


@main.command()
@click.argument('start', type=Date())
@click.argument('end', type=Date())
@click.option('--alloc', required=True, type=STR,
              help='Eagle allocation account name.')
@click.option('--walltime', '-wt', default=1.0, type=float,
              help='Eagle walltime request in hours. Default is 1.0')
@click.option('--feature', '-l', default=None, type=STR,
              help=('Additional flags for SLURM job. Format is "--qos=high" '
                    'or "--depend=[state:job_id]". Default is None.'))
@click.option('--memory', '-mem', default=None, type=INT,
              help='Eagle node memory request in GB. Default is None')
@click.option('--stdout_path', '-sout', default=None, type=STR,
              help='Subprocess standard output path. Default is in out_dir.')
@click.pass_context
def multiday(ctx, start, end, alloc, walltime, feature, memory, stdout_path):
    """ Calculate composite albedo for a range of dates. Range is inclusive """

    _setup_paths(ctx)
    if stdout_path is None:
        stdout_path = os.path.join(ctx.obj['apath'], 'stdout')

    if start > end:
        click.echo('Start date must be before end date')
        logger.error('Start date must be before end date')
        sys.exit(1)

    logger.info(f'Calculating composite albedo from {start} to {end}.')

    for date in daterange(start, end):
        cmd = get_node_cmd(date, ctx.obj['ipath'], ctx.obj['mpath'],
                           ctx.obj['apath'], ctx.obj['tiff'])
        logger.debug(f'command for slurm: {cmd}')

        name = dt.strftime(date, 'albedo_%Y%j')
        logger.info('Running composite albedo processing on Eagle with '
                    f'name "{name}" for {date}')
        slurm = SLURM(cmd, alloc=alloc, memory=memory,
                      walltime=walltime, feature=feature,
                      name=name, stdout_path=stdout_path)
        if slurm.id:
            msg = (f'Kicked off reV SC aggregation job "{name}" (SLURM '
                   f'jobid #{slurm.id}) on Eagle.')
        else:
            msg = (f'Was unable to kick off reV SC job "{name}". '
                   'Please see the stdout error messages')
        click.echo(msg)
        logger.info(msg)


@main.command()
@click.argument('albedo-file')
def h5totiff(albedo_file):
    """ Convert composite data in H5 file to TIFF """
    click.echo(f'Creating TIFF from {albedo_file}')
    CompositeAlbedoDay.write_tiff_from_h5(albedo_file)


def get_node_cmd(date, ipath, mpath, apath, tiff):
    """ Create shell command for single day CLI call """
    sdate = date.strftime('%Y%m%d')
    args = f'-i {ipath} -m {mpath} -a {apath}'
    if tiff:
        args += ' --tiff'
    args += f' singleday {sdate}'

    cmd = f'python -m nsrdb.albedo.cli {args}'
    return cmd


def daterange(start_date, end_date):
    """
    Create a range of dates.

    From https://stackoverflow.com/questions/1060279/
    iterating-through-a-range-of-dates-in-python
    """
    for n in range(int((end_date - start_date).days) + 1):
        yield start_date + timedelta(n)


if __name__ == '__main__':
    main()
