# -*- coding: utf-8 -*-
"""
Cloud Properties filling using phgynn
"""
import pandas as pd
import numpy as np
import time
import logging
from scipy.interpolate import interp1d

from nsrdb.all_sky import ICE_TYPES, WATER_TYPES

logger = logging.getLogger(__name__)


def clean_cloud_df(cloud_raw, filter_daylight=True, filter_clear=True,
                   add_feature_flag=True, sza_lim=89, nan_option='interp'):
    """ Clean up cloud data """
    t0 = time.time()
    cloud_data = cloud_raw.copy()
    day = (cloud_data['solar_zenith_angle'] < sza_lim)

    cloud_type = cloud_data['cloud_type']
    day_missing_ctype = day & (cloud_type < 0)
    mask = cloud_type < 0
    cloud_type[mask] = np.nan
    cloud_type = \
        pd.DataFrame(cloud_type).interpolate('nearest').ffill().bfill().values
    cloud_data['cloud_type'] = cloud_type

    cloudy = cloud_type.isin(ICE_TYPES + WATER_TYPES)
    day_clouds = day & cloudy
    day_missing_opd = day_clouds & (cloud_data['cld_opd_dcomp'] <= 0)
    day_missing_reff = day_clouds & (cloud_data['cld_reff_dcomp'] <= 0)

    mask = cloud_data['cld_opd_dcomp'] <= 0
    cloud_data['cld_opd_dcomp'][mask] = np.nan
    mask = cloud_data['cld_reff_dcomp'] <= 0
    cloud_data['cld_reff_dcomp'][mask] = np.nan

    logger.info('{:.2f}% of timesteps are daylight'
                .format(100 * day.sum() / len(day)))
    logger.info('{:.2f}% of daylight timesteps are cloudy'
                .format(100 * day_clouds.sum() / day.sum()))
    logger.info('{:.2f}% of daylight timesteps are missing cloud type'
                .format(100 * day_missing_ctype.sum() / day.sum()))
    logger.info('{:.2f}% of cloudy daylight timesteps are missing cloud opd'
                .format(100 * day_missing_opd.sum() / day_clouds.sum()))
    logger.info('{:.2f}% of cloudy daylight timesteps are missing cloud reff'
                .format(100 * day_missing_reff.sum() / day_clouds.sum()))

    logger.debug('Column NaN values:')
    for c, d in cloud_data.items():
        pnan = 100 * np.isna(d).sum() / len(d)
        logger.debug('\t"{}" has {:.2f}% NaN values'.format(c, pnan))

    if 'interp' in nan_option.lower():
        logger.debug('Interpolating opd and reff')
        for c, d in cloud_data.items():
            cloud_data[c] = \
                pd.DataFrame(d).interpolate('nearest').ffill().bfill().values

        cloud_data['cld_opd_dcomp'][~cloudy] = 0.0
        cloud_data['cld_reff_dcomp'][~cloudy] = 0.0
    elif 'drop' in nan_option.lower():
        l0 = len(cloudy)
        cloud_data = cloud_data.dropna(axis=0, how='any')
        day = (cloud_data['solar_zenith_angle'] < sza_lim)
        cloudy = cloud_data['cloud_type'].isin(ICE_TYPES + WATER_TYPES)
        logger.debug('Dropped {} rows with NaN values.'
                     .format(l0 - len(cloud_data)))

    assert ~any(cloud_data['cloud_type'] < 0)
    assert ~any(np.isna(d) for d in cloud_data.values())
    assert ~any(cloudy & (cloud_data['cld_opd_dcomp'] <= 0))

    if add_feature_flag:
        logger.debug('Adding feature flag')
        ice_clouds = cloud_data['cloud_type'].isin(ICE_TYPES)
        water_clouds = cloud_data['cloud_type'].isin(WATER_TYPES)
        flag = np.full(ice_clouds.shape, 'night')
        flag[day] = 'clear'
        flag[ice_clouds] = 'ice_cloud'
        flag[water_clouds] = 'water_cloud'
        flag[day_missing_ctype] = 'bad_cloud'
        flag[day_missing_opd] = 'bad_cloud'
        flag[day_missing_reff] = 'bad_cloud'

    mask = np.full(day.shape, True)
    if filter_daylight:
        mask &= day

    if filter_clear:
        mask &= cloudy

    if filter_daylight or filter_clear:
        logger.info('Data reduced from '
                    '{} rows to {} after filters ({:.2f}% of original)'
                    .format(len(mask), mask.sum(),
                            100 * mask.sum() / len(cloud_data)))
        for c, d in cloud_data.items():
            cloud_data[c] = d[mask]

    logger.debug('Feature flag column has these values: {}'
                 .format(np.unique(cloud_data['flag'])))
    logger.info('Cleaning took {:.1f} seconds'.format(time.time() - t0))

    return cloud_data
