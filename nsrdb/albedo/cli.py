import click
import sys
import os
from datetime import datetime as dt

from ims import get_dt  # , ImsDay
# from modis import ModisDay
from albedo import CompositeAlbedoDay


@click.command()
@click.option('--path', '-p', type=click.Path(exists=True),
              help='Path for all data files. This may be partially ' +
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
@click.option('--log-file', type=click.Path(), default=None,
              help='Logging output file.')
@click.option('--modis-shape', nargs=2, type=int, default=None,
              help='Shape of MODIS data')
@click.option('--ims-shape', nargs=2, type=int, default=None,
              help='Shape of IMS data')
@click.argument('date')  # help='Desired date for albedo data. In YYYYDDD ' +
def main(path, modis_path, ims_path, albedo_path, date, log_level, log_file,
         modis_shape, ims_shape):
    """
    Create composite albedo data for one day using MODIS and IMS data sets.
    """
    # Date sanity check
    if len(date) == 7:
        date = get_dt(int(date[:4]), int(date[4:]))
    elif len(date) == 8:
        date = dt.strptime(date, '%Y%m%d')
    else:
        print('Date must be provided in YYYYDDD or YYYYMMDD ' +
              f'format (e.g. 2015012 or 20150305). {date} ' +
              'was provided.')
        sys.exit(1)

    # Verify path is set
    if path is None and (modis_path is None or ims_path is None or
                         albedo_path is None):
        print('Paths for MODIS, IMS, and composite albedo data ' +
              'must be set together using --path, or ' +
              'individually using --modis-path, --ims-path, and ' +
              '--albedo-path.')
        sys.exit(1)

    # Over ride general path with specifics
    if path is not None:
        if modis_path is None:
            modis_path = path
        if ims_path is None:
            ims_path = path
        if albedo_path is None:
            albedo_path = path

    print(f'modis: {modis_path}, ims: {ims_path}, albedo: {albedo_path}, ' +
          f'date: {date}, path: {path}, log_level: {log_level}, ' +
          f'log_file: {log_file}')
    print(f'modis_shape {modis_shape}, ims_shape {ims_shape}')

    _kwargs = {}
    if modis_shape:
        print('modis shape!')
        _kwargs['modis_shape'] = modis_shape
    if ims_shape:
        print('ims shape!')
        _kwargs['ims_shape'] = ims_shape

    cad = CompositeAlbedoDay.run(date, modis_path, ims_path, albedo_path,
                                 log_level=log_level, log_file=log_file,
                                 **_kwargs)
    cad.write_albedo()


if __name__ == '__main__':
    main()
