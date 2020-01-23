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
@click.argument('date')  # help='Desired date for albedo data. In YYYYDDD ' +
def main(path, modis_path, ims_path, albedo_path, date):
    """
    Create composite albedo data for one day using MODIS and IMS data sets.
    """
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

    if path is not None:
        if modis_path is None:
            modis_path = path
        if ims_path is None:
            ims_path = path
        if albedo_path is None:
            albedo_path = path

    print(date)

    print(f'modis: {modis_path}, ims: {ims_path}, albedo: {albedo_path}, ' +
          f'date: {date}, path: {path}')

    cad = CompositeAlbedoDay.run(date, modis_path, ims_path, albedo_path)
    day = date.timetuple().tm_yday
    year = date.year
    outfilename = os.path.join(albedo_path, f'nsrdb_albedo_{day}_{year}.h5')
    cad.write_albedo(outfilename)


if __name__ == '__main__':
    main()
