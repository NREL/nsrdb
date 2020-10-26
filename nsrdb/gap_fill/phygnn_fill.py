# -*- coding: utf-8 -*-
"""
Cloud Properties filling using phgynn
"""
import logging
import numpy as np
import pandas as pd
import os
import time

from nsrdb.all_sky import ICE_TYPES, WATER_TYPES
from phgynn import PhygnnModel, PreProcess
from rex import MultiFileNSRDB

logger = logging.getLogger(__name__)


class PhygnnCloudFill:
    def __init__(self, phgynn_dir, h5_dir, day, filter_daylight=True,
                 filter_clear=True, add_feature_flag=True, sza_lim=89,
                 nan_option='interp'):
        self._phygnn_model = PhygnnModel.load(phgynn_dir)
        feature_data, self._dset_map = self._parse_data(h5_dir, day)
        self._feature_data = self._pre_process_features(
            feature_data,
            filter_daylight=filter_daylight,
            filter_clear=filter_clear,
            add_feature_flag=add_feature_flag,
            sza_lim=sza_lim, nan_option=nan_option)

    @staticmethod
    def _clean_feature_data(feature_raw, filter_daylight=True,
                            filter_clear=True, add_feature_flag=True,
                            sza_lim=89, nan_option='interp'):
        """ Clean up feature data """
        t0 = time.time()
        feature_data = feature_raw.copy()
        day = (feature_data['solar_zenith_angle'] < sza_lim)

        cloud_type = feature_data['cloud_type']
        day_missing_ctype = day & (cloud_type < 0)
        mask = cloud_type < 0
        if all(mask):
            cloud_type[...] = 0
        else:
            cloud_type[mask] = np.nan
            cloud_type = pd.DataFrame(cloud_type).interpolate(
                'nearest').ffill().bfill().values
            feature_data['cloud_type'] = cloud_type

        cloudy = cloud_type.isin(ICE_TYPES + WATER_TYPES)
        day_clouds = day & cloudy
        day_missing_opd = day_clouds & (feature_data['cld_opd_dcomp'] <= 0)
        day_missing_reff = day_clouds & (feature_data['cld_reff_dcomp'] <= 0)

        mask = feature_data['cld_opd_dcomp'] <= 0
        feature_data['cld_opd_dcomp'][mask] = np.nan
        mask = feature_data['cld_reff_dcomp'] <= 0
        feature_data['cld_reff_dcomp'][mask] = np.nan

        logger.info('{:.2f}% of timesteps are daylight'
                    .format(100 * day.sum() / len(day)))
        logger.info('{:.2f}% of daylight timesteps are cloudy'
                    .format(100 * day_clouds.sum() / day.sum()))
        logger.info('{:.2f}% of daylight timesteps are missing cloud type'
                    .format(100 * day_missing_ctype.sum() / day.sum()))
        logger.info('{:.2f}% of cloudy daylight timesteps are missing cloud '
                    'opd'
                    .format(100 * day_missing_opd.sum() / day_clouds.sum()))
        logger.info('{:.2f}% of cloudy daylight timesteps are missing cloud '
                    'reff'
                    .format(100 * day_missing_reff.sum() / day_clouds.sum()))

        logger.debug('Column NaN values:')
        for c, d in feature_data.items():
            pnan = 100 * np.isna(d).sum() / len(d)
            logger.debug('\t"{}" has {:.2f}% NaN values'.format(c, pnan))

        if 'interp' in nan_option.lower():
            logger.debug('Interpolating opd and reff')
            for c, d in feature_data.items():
                feature_data[c] = pd.DataFrame(d).interpolate(
                    'nearest').ffill().bfill().values

            feature_data['cld_opd_dcomp'][~cloudy] = 0.0
            feature_data['cld_reff_dcomp'][~cloudy] = 0.0
        elif 'drop' in nan_option.lower():
            l0 = len(cloudy)
            feature_data = feature_data.dropna(axis=0, how='any')
            day = (feature_data['solar_zenith_angle'] < sza_lim)
            cloudy = feature_data['cloud_type'].isin(ICE_TYPES + WATER_TYPES)
            logger.debug('Dropped {} rows with NaN values.'
                         .format(l0 - len(feature_data)))

        assert ~any(feature_data['cloud_type'] < 0)
        assert ~any(np.isna(d) for d in feature_data.values())
        assert ~any(cloudy & (feature_data['cld_opd_dcomp'] <= 0))

        if add_feature_flag:
            logger.debug('Adding feature flag')
            ice_clouds = feature_data['cloud_type'].isin(ICE_TYPES)
            water_clouds = feature_data['cloud_type'].isin(WATER_TYPES)
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
                                100 * mask.sum() / len(feature_data)))
            for c, d in feature_data.items():
                feature_data[c] = d[mask]

        logger.debug('Feature flag column has these values: {}'
                     .format(np.unique(feature_data['flag'])))
        logger.info('Cleaning took {:.1f} seconds'.format(time.time() - t0))

        return feature_data

    @classmethod
    def _pre_process_features(cls, feature_raw, filter_daylight=True,
                              filter_clear=True, add_feature_flag=True,
                              sza_lim=89, nan_option='interp'):
        feature_data = cls._clean_feature_data(
            feature_raw,
            filter_daylight=filter_daylight,
            filter_clear=filter_clear,
            add_feature_flag=add_feature_flag,
            sza_lim=sza_lim, nan_option=nan_option)

        feature_data = PreProcess.one_hot(feature_data,
                                          convert_int=False,
                                          categories=None,
                                          return_ind=False)

        return feature_data

    def _parse_data(self, h5_dir, day):
        res_path = os.path.join(h5_dir, '{}_*.h5'.format(day))
        with MultiFileNSRDB(res_path) as res:
            dset_map = res.h5._dset_map
            feature_data = {}
            for dset in self._phygnn_model.feature_names:
                feature_data[dset] = res[dset]

            return feature_data, dset_map
