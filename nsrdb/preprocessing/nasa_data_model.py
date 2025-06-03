"""Convert NASA data to UWISC format."""

import argparse
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import cached_property
from glob import glob

import numpy as np
import pandas as pd
import xarray as xr
from rex import init_logger

init_logger('nsrdb', log_level='DEBUG')
init_logger(__name__, log_level='DEBUG')

logger = logging.getLogger(__name__)

DROP_VARS = ['relative_time']

NAME_MAP = {
    'BT_3.75um': 'temp_3_75um_nom',
    'BT_10.8um': 'temp_11_0um_nom',
    'ref_0.63um': 'refl_0_65um_nom',
    'cloud_optical_depth': 'cld_opd_dcomp',
    'cloud_eff_particle_size': 'cld_reff_dcomp',
    'cloud_eff_pressure': 'cld_press_acha',
    'cloud_eff_height': 'cld_height_acha',
    'cloud_phase': 'cloud_type',
    'solar_zenith': 'solar_zenith_angle',
    'view_zenith': 'sensor_zenith_angle',
    'relative_azimuth': 'solar_azimuth_angle',
}

UWISC_CLOUD_TYPE = {
    'N/A': -15,
    'Clear': 0,
    'Probably Clear': 1,
    'Fog': 2,
    'Water': 3,
    'Super-Cooled Water': 4,
    'Mixed': 5,
    'Opaque Ice': 6,
    'Cirrus': 7,
    'Overlapping': 8,
    'Overshooting': 9,
    'Unknown': 10,
    'Dust': 11,
    'Smoke': 12,
}

NASA_CLOUD_TYPE = {
    'Clear sky snow/ice': 0,
    'Water cloud': 1,
    'Ice cloud': 2,
    'No cloud property retrievals': 3,
    'Clear sky land/water': 4,
    'Bad input data': 5,
    'Possible water cloud': 6,
    'Possible ice cloud': 7,
    'Cleaned data': 13,
}

CLOUD_TYPE_MAP = {
    0: 'Clear',
    1: 'Water',
    2: 'Opaque Ice',
    3: 'Unknown',
    4: 'Clear',
    5: 'N/A',
    6: 'Water',
    7: 'Opaque Ice',
    13: 'Unknown',
}


class NasaDataModel:
    """Class to handle conversion of nasa data to standard uwisc style format
    for NSRDB pipeline"""

    def __init__(self, input_file, output_pattern):
        """
        Parameters
        ----------
        input_file : str
            e.g. "./2017/01/01/nacomposite_2017.001.0000.nc"
        output_pattern : str
            Needs to include year, doy, and timestamp format keys.
            e.g. "./{year}/{doy}/nacomposite_{timestamp}.nc"
        """
        self.input_file = input_file
        self.output_pattern = output_pattern

    @cached_property
    def timestamp(self):
        """Get year, doy, hour from input file name.

        TODO: Should get this from relative_time variables to be more precise
        """
        match_pattern = r'.*_([0-9]+).([0-9]+).([0-9]+).\w+'
        ts = re.match(match_pattern, self.input_file).groups()
        year, doy, hour = ts
        secs = '000'
        return year, doy, hour, secs

    @cached_property
    def output_file(self):
        """Get output file name for given output pattern."""
        year, doy, _, _ = self.timestamp
        return self.output_pattern.format(
            year=year, doy=doy, timestamp=f's{"".join(self.timestamp)}'
        )

    @cached_property
    def ds(self):
        """Get xarray dataset for raw input file"""
        return xr.open_mfdataset(
            self.input_file,
            **{'group': 'map_data', 'decode_times': False},
            format='NETCDF4',
            engine='h5netcdf',
        )

    @classmethod
    def rename_vars(cls, ds):
        """Rename variables to uwisc conventions"""
        for k, v in NAME_MAP.items():
            if k in ds.data_vars:
                ds = ds.rename({k: v})
        return ds

    @classmethod
    def drop_vars(cls, ds):
        """Drop list of variables"""
        for v in DROP_VARS:
            if v in ds.data_vars:
                ds = ds.drop_vars(v)
        return ds

    @classmethod
    def remap_dims(cls, ds):
        """Rename dims and coords to standards. Make lat / lon into 2d arrays,
        as expected by cloud regridding routine."""

        sdims = ('south_north', 'west_east')
        for var in ds.data_vars:
            single_ts = (
                'time' in ds[var].dims
                and ds[var].transpose('time', ...).shape[0] == 1
            )
            if single_ts and var != 'reference_time':
                ds[var] = (sdims, ds[var].isel(time=0).data)

        ref_time = ds.attrs.get('reference_time', None)
        if ref_time is not None:
            ti = pd.DatetimeIndex([ref_time]).values
            ds = ds.assign_coords({'time': ('time', ti)})
        if 'Lines' in ds.dims:
            ds = ds.swap_dims({'Lines': 'south_north', 'Pixels': 'west_east'})
        if 'lat' in ds.coords:
            ds = ds.rename({'lat': 'latitude', 'lon': 'longitude'})

        ds['south_north'] = ds['latitude']
        ds['west_east'] = ds['longitude']

        lons, lats = np.meshgrid(ds['longitude'], ds['latitude'])
        ds = ds.assign_coords({
            'latitude': (sdims, lats),
            'longitude': (sdims, lons),
        })
        return ds

    @classmethod
    def remap_cloud_phase(cls, ds):
        """Map nasa cloud phase flags to uwisc values."""
        ct_name = NAME_MAP['cloud_phase']
        cloud_type = ds[ct_name].values.copy()
        for val, cs_type in CLOUD_TYPE_MAP.items():
            cloud_type = np.where(
                ds[ct_name].values.astype(int) == int(val),
                UWISC_CLOUD_TYPE[cs_type],
                cloud_type,
            )
        ds[ct_name] = (ds[ct_name].dims, cloud_type)
        return ds

    @classmethod
    def derive_stdevs(cls, ds):
        """Derive standard deviations of some variables, which are used as
        training features."""
        for var_name in ('refl_0_65um_nom', 'temp_11_0um_nom'):
            stddev = (
                ds[var_name]
                .rolling(
                    south_north=3, west_east=3, center=True, min_periods=1
                )
                .std()
            )
            ds[f'{var_name}_stddev_3x3'] = stddev
        return ds

    @classmethod
    def write_output(cls, ds, output_file):
        """Write converted dataset to output_file."""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        ds = ds.transpose('south_north', 'west_east', ...)
        ds.load().to_netcdf(output_file, format='NETCDF4', engine='h5netcdf')

    @classmethod
    def run(cls, input_file, output_pattern):
        """Run conversion routine and write converted dataset."""
        dm = cls(input_file, output_pattern)

        if os.path.exists(dm.output_file):
            logger.info(
                '%s already exists. Skipping conversion.', dm.output_file
            )
        else:
            logger.info('Geting xarray dataset for %s', input_file)
            ds = dm.ds

            logger.info('Remapping dimensions.')
            ds = dm.remap_dims(ds)

            logger.info('Renaming variables.')
            ds = dm.rename_vars(ds)

            logger.info('Dropping some variables.')
            ds = dm.drop_vars(ds)

            logger.info('Remapping cloud type values.')
            ds = dm.remap_cloud_phase(ds)

            logger.info('Deriving some stddev variables.')
            ds = dm.derive_stdevs(ds)

            logger.info('Writing converted file to %s', dm.output_file)
            dm.write_output(ds, dm.output_file)


def run_jobs(input_pattern, output_pattern, max_workers=None):
    """Run multiple file conversion jobs"""

    files = glob(input_pattern)

    if max_workers == 1:
        for file in files:
            NasaDataModel.run(input_file=file, output_pattern=output_pattern)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for file in files:
                fut = executor.submit(
                    NasaDataModel.run,
                    input_file=file,
                    output_pattern=output_pattern,
                )
                futures[fut] = file

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error('Error processing file: %s', futures[future])
                    logger.exception(e)

    logger.info('Finished converting %s files.', len(files))


if __name__ == '__main__':
    default_output_pattern = '/projects/pxs/nasa_polar/standardized/{year}'
    default_output_pattern += '/{doy}/nacomposite_{timestamp}.nc'
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input_pattern',
        type=str,
        help="""File pattern for input_files. e.g.
             /projects/pxs/nasa_polar/2023/*/*/*.nc""",
    )
    parser.add_argument(
        '-output_pattern',
        type=str,
        default=default_output_pattern,
        help='File pattern for output files.',
    )
    parser.add_argument(
        '-max_workers',
        type=int,
        default=10,
        help='Number of workers to use for parallel file conversion',
    )
    args = parser.parse_args()
    run_jobs(
        input_pattern=args.input_pattern,
        output_pattern=args.output_pattern,
        max_workers=args.max_workers,
    )
