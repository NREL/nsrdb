# -*- coding: utf-8 -*-
"""Map one year of IMS snow data to the MODIS dataset.

Created on Wed Apr 10 08:44:30 2019

@author: gbuster
"""

import os
import pandas as pd
import h5py
import numpy as np
import logging
import psutil
from scipy.spatial import cKDTree
from warnings import warn

from nsrdb.utilities.loggers import init_logger
from nsrdb.data_model.base_handler import AncillaryVarHandler
from nsrdb import CONFIGDIR


logger = logging.getLogger(__name__)


def mem_str():
    """Get a string to log memory status."""
    mem = psutil.virtual_memory()
    msg = ('{0:.3f} GB used of {1:.3f} GB total ({2:.1f}% used) '
           '({3:.3f} GB free) ({4:.3f} GB available).'
           .format(mem.used / 1e9,
                   mem.total / 1e9,
                   100 * mem.used / mem.total,
                   mem.free / 1e9,
                   mem.available / 1e9))
    return msg


def get_modis_fmap(year=2015,
                   modis_dir='/scratch/gbuster/albedo/modis/source_2015'):
    """Get a df mapping modis files to each day of year.

    Parameters
    ----------
    year : int
        Modis year of interest. It is expected that the modis files
        end in "_year.h5"
    modis_dir : str
        Directory containing modis .h5 files.

    Returns
    -------
    df : pd.DataFrame
        Dataframe describing and mapping the modis files. The index is a
        daily DatetimeIndex for the given year (len = 365 or 366).
        Column 'doy' is the closest day of year that modis has data for.
        Column 'fpath' is the full file path to the corresponding modis file.
    """

    year_str = '_{}'.format(year)
    flist = os.listdir(modis_dir)
    doy_year = []
    doy = []
    for f in flist:
        f = f.replace(year_str + '.h5', '')
        splitlist = f.split('_')
        doy_year.append(splitlist[-1] + year_str)
        doy.append(int(splitlist[-1]))

    doy_year = sorted(doy_year, key=lambda x: x.split('_')[0])
    doy = sorted(doy, key=int)

    time_index = pd.to_datetime(doy_year, format='%j_%Y')

    df = pd.DataFrame({'doy': doy}, index=time_index)
    full_ti = pd.date_range(start='1-1-{}'.format(year),
                            end='1-1-{}'.format(year + 1), freq='1D')[:-1]

    df = df.reindex(full_ti).interpolate(method='nearest', axis=0)\
        .fillna(method='ffill').fillna(method='bfill')

    df['doy'] = df['doy'].astype(int).astype(str).str.zfill(3)
    df['path'] = modis_dir
    df['fname'] = None
    df['fpath'] = None
    flist = os.listdir(modis_dir)
    for i in df.index:
        for fname in flist:
            if '_{}_'.format(df.loc[i, 'doy']) in fname and year_str in fname:
                df.loc[i, 'fname'] = fname
                df.loc[i, 'fpath'] = os.path.join(modis_dir, fname)
                break

    return df


def get_modis_lat_lon(fpath):
    """Get lat/lon data arrays from a modis file."""
    with h5py.File(fpath, 'r') as f:
        lat = f['Latitude'][...]
        lon = f['Longitude'][...]

    # Offset longitude
    lon[lon > 0] = lon[lon > 0] - 360

    return lat, lon


def get_modis_meta(fpath):
    """Get the modis meta data with latitude longitude columns."""
    lat, lon = get_modis_lat_lon(fpath)
    lat_mesh, lon_mesh = np.meshgrid(lat, lon)
    meta = pd.DataFrame({'latitude': lat_mesh.ravel(),
                         'longitude': lon_mesh.ravel()})
    return meta


def get_albedo(fpath, group1='MCD43GF_30arc_second', group2='Data Fields',
               dset='Albedo_Map_0.3-5.0'):
    """Get Albedo data array from modis file."""
    with h5py.File(fpath, 'r') as f:
        attrs = dict(f[group1][group2][dset].attrs)
        data = f[group1][group2][dset][...]

    scale_factor = attrs.get('scale_factor', 1)
    add_offset = attrs.get('add_offset', 0)
    if add_offset != 0:
        warn('Warning, add offset = {}'.format(add_offset))
    data = data.ravel().astype(np.float32)
    data *= scale_factor
    data += add_offset
    data[data > 1] = np.nan

    return data


def get_ims_meta(f_ims):
    """Get the IMS meta dataframe with latitude longitude columns."""
    with h5py.File(f_ims, 'r') as f:
        meta = pd.DataFrame(f['meta'][...])
    return meta


def get_snow(f_ims, day_index):
    """Get an array of snow data from an IMS file."""
    with h5py.File(f_ims, 'r') as f:
        snow_data = f['snow_cover'][day_index, :]
    return snow_data


def map_modis(day_index_range, year, f_ims, dir_out, modis_year=2015,
              modis_dir='/scratch/gbuster/albedo/modis/source_2015',
              snow_albedo=0.8669):
    """Map IMS snow data to albedo data and save to disk.

    Parameters
    ----------
    day_index_range : iterable
        Iterable (list, range) of zero-indexed day indices
        (0 through 364 or 365[leap] for full year).
    year : int
        NSRDB and IMS year of interest.
    f_ims : str
        Full file path to a full year IMS snow data .h5 file.
    dir_out : str
        Output directory to save final NSRDB albedo data.
    modis_year : int
        Year of MODIS albedo data to pull from (2015 is most recent as of 4/19)
    modis_dir : str
        Directory containing MODIS .h5 albedo files for the given modis year.
    snow_albedo : float
        Albedo value to assign to snowy data.
    """

    log_file = os.path.join(dir_out, 'nsrdb_albedo_{}.log'.format(year))
    init_logger(__name__, log_file=log_file, log_level='DEBUG')

    logger.info('Making {} NSRDB albedo from IMS and MODIS:\n\t{}\n\t{}'
                .format(year, f_ims, modis_dir))

    # get NSRDB meta data for albedo
    albedo_meta = AncillaryVarHandler(
        os.path.join(CONFIGDIR, 'nsrdb_vars.csv'), 'albedo', None)
    albedo_attrs = {'units': albedo_meta.units,
                    'scale_factor': albedo_meta.scale_factor}

    # get modis file listing and map
    modis_fmap = get_modis_fmap(year=modis_year, modis_dir=modis_dir)

    # get modis data
    logger.info('Getting MODIS meta data.')
    mds_meta = get_modis_meta(
        modis_fmap.loc[modis_fmap.index[0], 'fpath'])

    # make tree
    logger.info('Making kdTree from MODIS meta data.')
    logger.debug(mem_str())
    tree = cKDTree(mds_meta.values)
    logger.debug('Completed kdTree from MODIS meta data.')
    logger.debug(mem_str())

    for day_index in day_index_range:
        time_index = modis_fmap.index[day_index]
        logger.info('Working on time index {}'.format(time_index))
        doy = modis_fmap.loc[time_index, 'doy']
        f_modis = modis_fmap.loc[time_index, 'fpath']

        # get modis data for current time index
        logger.info('Getting MODIS albedo data.')
        albedo = get_albedo(f_modis)

        # get ims data
        logger.info('Getting IMS snow data.')
        logger.debug(mem_str())
        ims_meta = get_ims_meta(f_ims)
        snow_data = get_snow(f_ims, day_index)

        # extract ims latitude and ims longitudes where there is snow
        where_snow = np.where(snow_data == 1)[0]
        ims_meta = ims_meta.iloc[where_snow, :].values

        # Find the modis indices corresponding to ims coords where snow
        logger.info('Finding MODIS indices where IMS has snow (tree query).')
        ind = tree.query(ims_meta)[1]

        # write snow albedo value to indices in modis where snow in IMS
        logger.info('Updating snowy MODIS cells to albedo of {}'
                    .format(snow_albedo))
        logger.debug(mem_str())
        albedo[ind] = snow_albedo

        # apply scale and dtype conversion
        albedo *= albedo_attrs['scale_factor']
        albedo = albedo.astype(albedo_meta.dtype)

        # write to output file
        f_out = os.path.join(dir_out, 'nsrdb_albedo_{}_{}.h5'
                             .format(doy, year))
        logger.info('Writing NSRDB albedo data to {}'.format(f_out))
        logger.debug(mem_str())
        with h5py.File(f_out, 'w') as f:
            f.create_dataset('albedo', shape=albedo.shape, dtype=albedo.dtype,
                             data=albedo)


if __name__ == '__main__':
    day_index_range = range(365)
    year = 2018
    f_ims = '/scratch/gbuster/albedo/ims/ims_2018_1k.h5'
    dir_out = '/scratch/gbuster/albedo/combined'
    map_modis(day_index_range, year, f_ims, dir_out)
