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
from nsrdb.utilities.execution import PBS
from nsrdb.qa.plots import Spatial


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
    lon = np.where(lon > 180, lon - 360, lon)

    return lat, lon


def get_modis_meta(fpath):
    """Get the modis meta data with latitude longitude columns."""
    lat, lon = get_modis_lat_lon(fpath)
    lat_mesh, lon_mesh = np.meshgrid(lat, lon)
    meta = pd.DataFrame({'latitude': lat_mesh.ravel(),
                         'longitude': lon_mesh.ravel()})
    return meta


def get_albedo(fpath, valid_lat_ind=None, valid_lon_ind=None,
               group1='MCD43GF_30arc_second', group2='Data Fields',
               dset='Albedo_Map_0.3-5.0'):
    """Get Albedo data array from modis file."""

    with h5py.File(fpath, 'r') as f:
        attrs = dict(f[group1][group2][dset].attrs)
        data = f[group1][group2][dset][...]
        logger.debug('Native MODIS albedo h5 dataset has shape {}'
                     .format(data.shape))

    if valid_lat_ind is not None:
        data = data[valid_lat_ind, :]
    if valid_lon_ind is not None:
        data = data[:, valid_lon_ind]

    scale_factor = attrs.get('scale_factor', 1)
    add_offset = attrs.get('add_offset', 0)
    if add_offset != 0:
        warn('Warning, add offset = {}'.format(add_offset))
    data = data.astype(np.float32)
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


def map_modis(day_index, year, f_ims, dir_out, modis_year=2015,
              modis_dir='/scratch/gbuster/albedo/modis/source_2015',
              snow_albedo=0.8669):
    """Map IMS snow data to albedo data and save to disk.

    Parameters
    ----------
    day_index : int
        Zero-indexed day indices.
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
    log_file = os.path.join(dir_out, 'nsrdb_albedo_{}_{}.log'
                            .format(str(day_index).zfill(3), year))
    init_logger(__name__, log_file=log_file, log_level='DEBUG')

    logger.info('Making {} NSRDB albedo from IMS and MODIS:\n\t{}\n\t{}'
                .format(year, f_ims, modis_dir))
    logger.info('Updating snowy MODIS cells to albedo of {}'
                .format(snow_albedo))

    # get NSRDB meta data for albedo
    albedo_meta = AncillaryVarHandler(
        os.path.join(CONFIGDIR, 'nsrdb_vars.csv'), 'surface_albedo', None)
    albedo_attrs = {'units': albedo_meta.units,
                    'scale_factor': albedo_meta.scale_factor}

    # get modis file listing and map
    modis_fmap = get_modis_fmap(year=modis_year, modis_dir=modis_dir)

    # get modis data
    logger.debug('Getting MODIS meta data.')
    mds_lat, mds_lon = get_modis_lat_lon(
        modis_fmap.loc[modis_fmap.index[0], 'fpath'])
    mds_lat = mds_lat.astype(np.float32)
    mds_lon = mds_lon.astype(np.float32)

    logger.debug('Latitude and longitude have shapes {} and {}, '
                 'after exclusions, respectively.'
                 .format(mds_lat.shape, mds_lon.shape))
    logger.debug('MODIS latitude ranges from {} to {}'
                 .format(np.min(mds_lat), np.max(mds_lat)))
    logger.debug('MODIS longitude ranges from {} to {}'
                 .format(np.min(mds_lon), np.max(mds_lon)))

    # make tree(s)
    logger.info('Making kdTree from MODIS meta data.')
    logger.debug(mem_str())
    lat_tree = cKDTree(np.expand_dims(mds_lat, axis=1))
    lon_tree = cKDTree(np.expand_dims(mds_lon, axis=1))
    logger.debug('Completed kdTree from MODIS meta data.')
    logger.debug(mem_str())

    time_index = modis_fmap.index[day_index]
    logger.info('Working on time index {}'.format(time_index))
    doy = modis_fmap.loc[time_index, 'doy']
    f_modis = modis_fmap.loc[time_index, 'fpath']

    # get ims data
    logger.debug('Getting IMS snow data.')
    logger.debug(mem_str())
    ims_meta = get_ims_meta(f_ims)
    snow_data = get_snow(f_ims, day_index)

    # extract ims latitude and ims longitudes where there is snow
    where_snow = np.where(snow_data == 1)[0]
    ims_meta = ims_meta.iloc[where_snow, :]
    logger.debug('There are {} sites with snow.'.format(len(ims_meta)))

    logger.debug('IMS latitude ranges from {} to {}'
                 .format(np.min(ims_meta['latitude'].values),
                         np.max(ims_meta['latitude'].values)))
    logger.debug('IMS longitude ranges from {} to {}'
                 .format(np.min(ims_meta['longitude'].values),
                         np.max(ims_meta['longitude'].values)))

    # get modis data for current time index
    logger.debug('Getting MODIS albedo data from file for doy {}'.format(doy))
    logger.debug(mem_str())
    albedo = get_albedo(f_modis)
    logger.debug('Imported MODIS albedo array has shape {}'
                 .format(albedo.shape))

    # work through IMS snow sites in sub iterations to limit memory spike
    # reduce increment if memory errors are breaking the code
    # small increments (10k) do not seem to slow down the code substantially
    i0 = 0
    i1 = 0
    count = 0
    increment = 10000
    while True:
        i0 = i1
        i1 = np.min((len(ims_meta), i0 + increment))
        if i0 == i1:
            break
        count += 1

        if count % 100 == 0:
            logger.debug('Performing sub iteration #{} from IMS meta index '
                         '{} to {} (limit is {}).'
                         .format(count, i0, i1, len(ims_meta)))
            logger.debug(mem_str())

        ims_meta_sub = ims_meta.iloc[i0:i1, :]

        # Find the modis indices corresponding to ims coords where snow
        lat_dist, lat_ind = lat_tree.query(
            np.expand_dims(ims_meta_sub['latitude'].values, axis=1))
        lon_dist, lon_ind = lon_tree.query(
            np.expand_dims(ims_meta_sub['longitude'].values, axis=1))

        if count % 100 == 0:
            logger.debug('{} {} {} {}'.format(np.mean(lat_dist),
                                              np.median(lat_dist),
                                              np.max(lat_dist),
                                              np.mean(lon_dist),
                                              np.median(lon_dist),
                                              np.max(lon_dist)))

        # write snow albedo value to indices in modis where snow in IMS
        albedo[lat_ind, lon_ind] = snow_albedo

    # plot for QA
    logger.debug('Making pretty pictures.')
    mesh_lon, mesh_lat = np.meshgrid(mds_lon, mds_lat)
    mesh_lat = mesh_lat.ravel().astype(np.float32)
    mesh_lon = mesh_lon.ravel().astype(np.float32)
    plot_df = pd.DataFrame({'latitude': mesh_lat, 'longitude': mesh_lon,
                            'albedo': albedo.ravel()})
    Spatial.plot_geo_df(plot_df.iloc[range(0, len(plot_df), 50), :],
                        'albedo_{}_{}'.format(str(day_index).zfill(3), year),
                        dir_out, xlim=(-180, 180), ylim=(-90, 90))
    del plot_df, mesh_lat, mesh_lon

    # apply scale and dtype conversion
    albedo *= albedo_attrs['scale_factor']
    albedo = albedo.astype(albedo_meta.dtype)

    logger.debug('The final albedo range for date index {} in {} is {} to {}'
                 .format(day_index, year, np.min(albedo), np.max(albedo)))

    # write to output file
    f_out = os.path.join(dir_out, 'nsrdb_albedo_{}_{}.h5'
                         .format(str(day_index).zfill(3), year))
    logger.info('Writing NSRDB albedo data to {}'.format(f_out))
    logger.debug(mem_str())
    with h5py.File(f_out, 'w') as f:
        f.create_dataset('surface_albedo', shape=albedo.shape,
                         dtype=albedo.dtype, data=albedo)
        for k, v in albedo_attrs.items():
            f['surface_albedo'].attrs[k] = v

        f.create_dataset('latitude', shape=mds_lat.shape,
                         dtype=mds_lat.dtype, data=mds_lat)
        f.create_dataset('longitude', shape=mds_lon.shape,
                         dtype=mds_lon.dtype, data=mds_lon)


def peregrine(index, year, f_ims, dir_out, alloc='pxs', queue='short',
              feature=None):
    """Run the map_modis method on a peregrine node."""

    name = 'alb_{}_{}'.format(str(index).zfill(3), year)
    cmd = ('python -c '
           '\'from nsrdb.albedo.map_ims_to_modis import map_modis; '
           'map_modis({index}, {year}, "{f_ims}", "{dir_out}")\''
           .format(index=index, year=year, f_ims=f_ims, dir_out=dir_out))

    pbs = PBS(cmd, alloc=alloc, queue=queue, name=name,
              stdout_path=dir_out, feature=feature)

    print('\ncmd:\n{}\n'.format(cmd))

    if pbs.id:
        msg = ('Kicked off job "{}" (PBS jobid #{}) on '
               'Peregrine.'.format(name, pbs.id))
    else:
        msg = ('Was unable to kick off job "{}". '
               'Please see the stdout error messages'
               .format(name))
    print(msg)


if __name__ == '__main__':
    for index in range(0, 365, 10):
        year = 2018
        f_ims = '/scratch/gbuster/albedo/ims/ims_2018_1k.h5'
        dir_out = '/scratch/gbuster/albedo/combined'
        f_out = 'nsrdb_albedo_{}_{}.h5'.format(str(index).zfill(3), year)
        dir_list = os.listdir(dir_out)
        if f_out not in dir_list:
            peregrine(index, year, f_ims, dir_out, queue='short',
                      feature='feature=haswell')
