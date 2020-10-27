# -*- coding: utf-8 -*-
"""
Cloud Properties filling using phgynn
"""
import h5py
import logging
import numpy as np
import pandas as pd
import os
import shutil
import time

from nsrdb.all_sky import ICE_TYPES, WATER_TYPES
from phgynn import PhygnnModel, PreProcess
from rex import MultiFileNSRDB

logger = logging.getLogger(__name__)


class PhygnnCloudFill:
    """
    Use phygnn to fill missing cloud data
    """
    def __init__(self, phgynn_dir, h5_dir, day, filter_daylight=True,
                 filter_clear=True, sza_lim=89,
                 nan_option='interp'):
        """
        Parameters
        ----------
        phgynn_dir : str
            Directory to load phygnn model from
        h5_dir : str
            Directory containing NSRDB .h5 files
        day : str
            Date for day to be gap filled
        filter_daylight : bool, optional
            Flag to filter daylight timesteps, by default True
        filter_clear : bool, optional
            Flag to filter clear sky timesteps, by default True
        sza_lim : int, optional
            Solar zenith angle limit, by default 89
        nan_option : str, optional
            Method to handle nan values, by default 'interp'
        """
        self._phygnn_model = PhygnnModel.load(phgynn_dir)
        feature_data, self._dset_map = self._parse_data(h5_dir, day)
        self._feature_data = self._pre_process_features(
            feature_data,
            filter_daylight=filter_daylight,
            filter_clear=filter_clear,
            sza_lim=sza_lim, nan_option=nan_option)

    @property
    def phygnn_model(self):
        """
        Pre-trained PhygnnModel instance

        Returns
        -------
        PhygnnModel
        """
        return self._phygnn_model

    @property
    def dset_map(self):
        """
        Mapping of datasets to .h5 files

        Returns
        -------
        dict
        """
        return self._dset_map

    @property
    def feature_data(self):
        """
        Feature data arrays

        Returns
        -------
        dict
        """
        return self._feature_data

    @staticmethod
    def _clean_feature_data(feature_raw, filter_daylight=True,
                            filter_clear=True, add_feature_flag=True,
                            sza_lim=89, nan_option='interp'):
        """
        Clean feature data

        Parameters
        ----------
        feature_raw : dict
            Dictionary of feature data arrays
        filter_daylight : bool, optional
            Flag to filter daylight timesteps, by default True
        filter_clear : bool, optional
            Flag to filter clear sky timesteps, by default True
        add_feature_flag : bool, optional
            Flag to add cloud type flag dataset, by default True
        sza_lim : int, optional
            Solar zenith angle limit, by default 89
        nan_option : str, optional
            Method to handle nan values, by default 'interp'

        Returns
        -------
        feature_data : ndarray
            Clean feature data
        """
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
            feature_data['flag'] = flag

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
                              filter_clear=True, sza_lim=89,
                              nan_option='interp'):
        """
        Clean feature data and convert to one-hot as needed

        Parameters
        ----------
        feature_raw : dict
            Dictionary of feature data arrays
        filter_daylight : bool, optional
            Flag to filter daylight timesteps, by default True
        filter_clear : bool, optional
            Flag to filter clear sky timesteps, by default True
        sza_lim : int, optional
            Solar zenith angle limit, by default 89
        nan_option : str, optional
            Method to handle nan values, by default 'interp'

        Returns
        -------
        feature_data : ndarray
            Clean and one-hotted feature data
        """
        feature_data = cls._clean_feature_data(
            feature_raw,
            filter_daylight=filter_daylight,
            filter_clear=filter_clear,
            add_feature_flag=True,
            sza_lim=sza_lim, nan_option=nan_option)

        for c, d in feature_data.items():
            if c != 'flag':
                feature_data[c] = PreProcess.one_hot(d,
                                                     convert_int=False,
                                                     categories=None,
                                                     return_ind=False)

        return feature_data

    def _parse_data(self, h5_dir, day):
        """
        Parse feature data from .h5 files

        Parameters
        ----------
        h5_dir : str
            Directory containing NSRDB .h5 files
        day : str
            Date for day to be gap filled

        Returns
        -------
        feature_data : dict
            Dictionary of feature data arrays
        dset_map : dict
            Mapping of datasets to source .h5 files
        """
        res_path = os.path.join(h5_dir, '{}_*.h5'.format(day))
        with MultiFileNSRDB(res_path) as res:
            dset_map = res.h5._dset_map
            feature_data = {}
            for dset in self._phygnn_model.feature_names:
                logger.debug('Loading {} data'.format(dset))
                feature_data[dset] = res[dset]

            return feature_data, dset_map

    def _archive_cld_properties(self):
        """
        Archive original cloud property (cld_*) .h5 files
        """
        cld_dsets = ['cld_opd_dcomp', 'cld_reff_dcomp', 'cld_press_acha', ]
        for dset in cld_dsets:
            src_fpath = self.dset_map[dset]
            src_dir, f_name = os.path.split(src_fpath)
            dst_dir = os.path.join(src_dir, 'raw')
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)

            dst_fpath = os.path.join(dst_dir, f_name)
            if os.path.exists(dst_fpath):
                msg = ("{} already exists, this suggests gap fill "
                       "has already been run!")
                logger.error(msg)
                raise RuntimeError(msg)
            else:
                logger.debug('Archiving {} to {}'
                             .format(src_fpath, dst_fpath))
                shutil.copy(src_fpath, dst_fpath + '.tmp')

    def _predict_cld_properties(self):
        """
        Predict cloud properties with phygnn

        Returns
        -------
        predicted_data : dict
            Dictionary of predicted cloud properties
        """
        shape = (self.feature_data['flag'].size,
                 self.phygnn_model.feature_dims)
        features = np.full(shape, np.nan, dtype=np.float32)
        logger.debug('Feature data shape: {}'.format(shape))
        for i, dset in enumerate(self.phygnn_model.feature_names):
            features[:, i] = self.feature_data[dset].flatten(order='F')

        labels = self.phygnn_model.predict(features,
                                           table=False)
        logger.debug('Label data shape: {}'.format(labels.shape))

        shape = self.feature_data['flag'].shape
        predicted_data = {}
        for i, dset in enumerate(self.phygnn_model.label_names):
            logger.debug('Reshaping predicted {} to {}'
                         .format(dset, shape))
            predicted_data[dset] = labels[:, i].reshape(shape, order='F')

        return predicted_data

    def _fill_bad_cld_properties(self, predicted_data):
        """
        Fill bad cloud properties from predicted data

        Parameters
        ----------
        predicted_data : dict
            Dictionary of phygnn predicted cloud properties

        Returns
        -------
        filled_data : dict
            Dictionary of filled cld property data
        """
        mask = self.feature_data['flag'] == 'bad_cloud'
        logger.debug('Filling {} values using phygnn predictions'
                     .format(np.sum(mask)))
        filled_data = {}
        for dset, arr in predicted_data.items():
            logger.debug('Filling {} data'.format(dset))
            cld_data = self.feature_data[dset]
            cld_data[mask] = arr[mask]
            filled_data[dset] = cld_data

        return filled_data

    def fill_cld_properties(self):
        """
        Fill bad cloud properties using phygnn predicitons and save to disc
        in original files. Original files will be archived to a new "raw/"
        sub-directory
        """
        logger.info('Filling bad cloud properties using phygnn predicitons')
        self._archive_cld_properties()
        filled_data = \
            self._fill_bad_cld_properties(self._predict_cld_properties())

        for dset, arr in filled_data.items():
            fpath = self.dset_map[dset]
            logger.info('Updating {} data in {} with gap-filled data'
                        .format(dset, fpath))
            with h5py.File(fpath, mode='a') as f:
                f[dset][...] = arr

            src_dir, f_name = os.path.split(fpath)
            raw_path = os.path.join(src_dir, 'raw', f_name)
            logger.debug('renaming .tmp raw file to {}'
                         .format(raw_path))
            os.rename(raw_path + '.tmp', raw_path)

    @classmethod
    def fill(cls, phgynn_dir, h5_dir, day, filter_daylight=True,
             filter_clear=True, sza_lim=89, nan_option='interp'):
        """
        Fill cloud properties using phygnn predictions

        Parameters
        ----------
        phgynn_dir : str
            Directory to load phygnn model from
        h5_dir : str
            Directory containing NSRDB .h5 files
        day : str
            Date for day to be gap filled
        filter_daylight : bool, optional
            Flag to filter daylight timesteps, by default True
        filter_clear : bool, optional
            Flag to filter clear sky timesteps, by default True
        sza_lim : int, optional
            Solar zenith angle limit, by default 89
        nan_option : str, optional
            Method to handle nan values, by default 'interp'
        """
        p_fill = cls(phgynn_dir, h5_dir, day, filter_daylight=filter_daylight,
                     filter_clear=filter_clear, sza_lim=sza_lim,
                     nan_option=nan_option)
        p_fill.fill_cld_properties()
