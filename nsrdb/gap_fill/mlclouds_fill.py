# -*- coding: utf-8 -*-
"""
Cloud Properties filling using phgynn
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import numpy as np
import pandas as pd
import os
import shutil
import psutil
import time
from warnings import warn

from nsrdb.all_sky import ICE_TYPES, WATER_TYPES
from nsrdb.gap_fill.cloud_fill import CloudGapFill
from nsrdb.file_handlers.outputs import Outputs, Resource
from nsrdb.data_model.variable_factory import VarFactory
from phygnn import PhygnnModel
from mlclouds import MODEL_FPATH
from rex import MultiFileNSRDB

logger = logging.getLogger(__name__)


class MLCloudsFill:
    """
    Use the MLClouds algorith with phygnn model to fill missing cloud data
    """
    DEFAULT_MODEL = MODEL_FPATH

    def __init__(self, h5_source, fill_all=False, model_path=None,
                 var_meta=None):
        """
        Parameters
        ----------
        h5_source : str
            Path to directory containing multi-file resource file sets.
            Available formats:
                /h5_dir/
                /h5_dir/prefix*suffix
        fill_all : bool
            Flag to fill all cloud properties for all timesteps where
            cloud_type is cloudy.
        model_path : str | None
            Directory to load phygnn model from. This is typically a fpath to
            a .pkl file with an accompanying .json file in the same directory.
            None will try to use the default model path from the mlclouds
            project directory.
        var_meta : str | pd.DataFrame | None
            CSV file or dataframe containing meta data for all NSRDB variables.
            Defaults to the NSRDB var meta csv in git repo.
        """

        self._dset_map = None
        self._h5_source = h5_source
        self._fill_all = fill_all
        self._var_meta = var_meta
        if model_path is None:
            model_path = self.DEFAULT_MODEL

        logger.debug('Initializing MLCloudsFill with h5_source: {}'
                     .format(self._h5_source))
        logger.debug('Initializing MLCloudsFill with model: {}'
                     .format(model_path))
        logger.debug('MLCloudsFill fill filling all cloud properties: {}'
                     .format(self._fill_all))
        self._phygnn_model = PhygnnModel.load(model_path)

        with MultiFileNSRDB(self.h5_source) as res:
            self._dset_map = res.h5._dset_map
            self._res_shape = res.shape

        missing = []
        for dset in self._phygnn_model.feature_names:
            if (dset not in ('clear', 'ice_cloud', 'water_cloud', 'bad_cloud')
                    and dset not in self._dset_map):
                missing.append(dset)

        if any(missing):
            msg = ('The following datasets were missing in the h5_source '
                   'directory: {}'.format(missing))
            logger.error(msg)
            raise FileNotFoundError(msg)

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
    def h5_source(self):
        """
        Path to directory containing multi-file resource file sets.
        Available formats:
            /h5_dir/
            /h5_dir/prefix*suffix

        Returns
        -------
        str
        """
        return self._h5_source

    @property
    def dset_map(self):
        """
        Mapping of datasets to .h5 files

        Returns
        -------
        dict
        """
        return self._dset_map

    def parse_feature_data(self, feature_data=None, col_slice=slice(None)):
        """
        Parse raw feature data from .h5 files (will have gaps!)

        Parameters
        ----------
        feature_data : dict | None
            Pre-loaded feature data to add to (optional). Keys are the feature
            names (nsrdb dataset names), values are 2D numpy arrays
            (time x sites). Any dsets already in this input won't be re-read.
        col_slice : slice
            Column slice of the resource data to work on. This is a result of
            chunking the columns to reduce memory load. Use slice(None) for
            no chunking.

        Returns
        -------
        feature_data : dict
            Raw feature data with gaps. keys are the feature names (nsrdb
            dataset names), values are 2D numpy arrays (time x sites).
        """

        logger.info('Loading feature data.')
        if feature_data is None:
            feature_data = {}

        dsets = (self._phygnn_model.feature_names
                 + self._phygnn_model.label_names)

        with MultiFileNSRDB(self.h5_source) as res:
            for dset in dsets:
                if dset not in feature_data and dset in res.dsets:
                    logger.debug('Loading {} data'.format(dset))
                    feature_data[dset] = res[dset, :, col_slice]

        mem = psutil.virtual_memory()
        logger.info('Feature data loaded for column slice {}. '
                    'Memory utilization is {:.3f} GB out of {:.3f} GB total '
                    '({:.2f}% used).'
                    .format(col_slice, mem.used / 1e9, mem.total / 1e9,
                            100 * mem.used / mem.total))

        return feature_data

    def _init_clean_arrays(self):
        """Initialize a dict of numpy arrays for clean data.
        """
        arr = np.zeros(self._res_shape, dtype=np.float32)
        dsets = ('cloud_type', 'cld_press_acha', 'cld_opd_dcomp',
                 'cld_reff_dcomp')
        clean_arrays = {d: arr.copy() for d in dsets}
        fill_flag_array = arr.copy()
        return clean_arrays, fill_flag_array

    @staticmethod
    def _clean_array(dset, array):
        """Clean a dataset array using temporal nearest neighbor interpolation.

        Parameters
        ----------
        dset : str
            NSRDB dataset name
        array : np.ndarray
            2D (time x sites) float numpy array of data for dset. Missing
            values should be set to NaN.

        Returns
        -------
        array : np.ndarray
            2D (time x sites) float numpy array of data for dset. Missing
            values should be filled
        """
        any_bad = np.isnan(array).any()
        all_bad = (~np.isnan(array)).sum(axis=0) < 3
        if any(all_bad):
            mean_impute = np.nanmean(array)
            count = all_bad.sum()
            msg = ('Feature dataset "{}" has {} columns with all NaN '
                   'values out of {} ({:.2f}%). Filling with '
                   'mean value of {}.'
                   .format(dset, count, array.shape[1],
                           100 * count / array.shape[1], mean_impute))
            logger.warning(msg)
            warn(msg)
            array[:, all_bad] = mean_impute

        if any_bad:
            array = pd.DataFrame(array).interpolate(
                'nearest').ffill().bfill().values

        return array

    def clean_feature_data(self, feature_raw, fill_flag, sza_lim=90,
                           max_workers=1):
        """
        Clean feature data

        Parameters
        ----------
        feature_raw : dict
            Raw feature data with gaps. keys are the feature names (nsrdb
            dataset names), values are 2D numpy arrays (time x sites).
        fill_flag : np.ndarray
            Integer array of flags showing what data was filled and why.
        sza_lim : int, optional
            Solar zenith angle limit below which missing cloud property data
            will be gap filled. By default 90 to fill all missing daylight data
        max_workers : None | int
            Maximum workers to clean data in parallel. 1 is serial and None
            uses all available workers.

        Returns
        -------
        feature_data : dict
            Clean feature data without gaps. keys are the feature names
            (nsrdb dataset names), values are 2D numpy arrays (time x sites).
        fill_flag : np.ndarray
            Integer array of flags showing what data was filled and why.
        """

        t0 = time.time()
        feature_data = feature_raw.copy()
        day = (feature_data['solar_zenith_angle'] < sza_lim)

        cloud_type = feature_data['cloud_type']
        assert cloud_type.shape == fill_flag.shape
        day_missing_ctype = day & (cloud_type < 0)
        mask = cloud_type < 0
        full_missing_ctype_mask = mask.all(axis=0)
        if any(full_missing_ctype_mask):
            msg = ('Some sites ({} out of {}) have full timeseries of missing '
                   'cloud type!'
                   .format(full_missing_ctype_mask, mask.shape[1]))
            warn(msg)
            logger.warning(msg)
            mask[:, full_missing_ctype_mask] = False
            cloud_type[:, full_missing_ctype_mask] = 0
            fill_flag[:, full_missing_ctype_mask] = 2

        if mask.any():
            logger.info('There are {} missing cloud type observations '
                        'out of {}. Interpolating with Nearest Neighbor.'
                        .format(mask.sum(), mask.shape[0] * mask.shape[1]))
            cloud_type[mask] = np.nan
            fill_flag[mask] = 1
            cloud_type = pd.DataFrame(cloud_type).interpolate(
                'nearest').ffill().bfill().values
            feature_data['cloud_type'] = cloud_type

        cloudy = np.isin(cloud_type, ICE_TYPES + WATER_TYPES)
        day_clouds = day & cloudy
        day_missing_opd = day_clouds & (feature_data['cld_opd_dcomp'] <= 0)
        day_missing_reff = day_clouds & (feature_data['cld_reff_dcomp'] <= 0)

        mask_fill_flag_opd = day_missing_opd & (fill_flag == 0)
        mask_fill_flag_reff = day_missing_reff & (fill_flag == 0)
        mask_all_bad_opd = ((day_missing_opd | ~day).all(axis=0)
                            & (fill_flag < 2).all(axis=0))
        mask_all_bad_reff = ((day_missing_reff | ~day).all(axis=0)
                             & (fill_flag < 2).all(axis=0))
        fill_flag[mask_fill_flag_opd] = 7  # fill_flag=7 is mlcloud fill
        fill_flag[mask_fill_flag_reff] = 7  # fill_flag=7 is mlcloud fill
        fill_flag[:, mask_all_bad_opd] = 4
        fill_flag[:, mask_all_bad_reff] = 4

        logger.info('{:.2f}% of timesteps are daylight'
                    .format(100 * day.sum() / (day.shape[0] * day.shape[1])))
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

        mask = feature_data['cld_opd_dcomp'] <= 0
        feature_data['cld_opd_dcomp'][mask] = np.nan
        mask = feature_data['cld_reff_dcomp'] <= 0
        feature_data['cld_reff_dcomp'][mask] = np.nan

        logger.debug('Column NaN values:')
        for c, d in feature_data.items():
            pnan = 100 * np.isnan(d).sum() / (d.shape[0] * d.shape[1])
            logger.debug('\t"{}" has {:.2f}% NaN values'.format(c, pnan))

        logger.debug('Interpolating feature data using nearest neighbor.')

        if max_workers == 1:
            for c, d in feature_data.items():
                if c not in self.phygnn_model.label_names:
                    feature_data[c] = self._clean_array(c, d)
        else:
            logger.info('Interpolating feature data with {} max workers'
                        .format(max_workers))
            futures = {}
            with ProcessPoolExecutor(max_workers=max_workers) as exe:
                for c, d in feature_data.items():
                    if c not in self.phygnn_model.label_names:
                        future = exe.submit(self._clean_array, c, d)
                        futures[future] = c

                for future in as_completed(futures):
                    c = futures[future]
                    feature_data[c] = future.result()

        logger.debug('Feature data interpolation is complete.')

        feature_data['cld_opd_dcomp'][~cloudy] = 0.0
        feature_data['cld_reff_dcomp'][~cloudy] = 0.0

        assert ~(feature_data['cloud_type'] < 0).any()
        assert ~any([np.isnan(d).any() for d in feature_data.values()])
        assert ~(cloudy & (feature_data['cld_opd_dcomp'] <= 0)).any()
        assert ~(cloudy & (feature_data['cld_reff_dcomp'] <= 0)).any()

        logger.debug('Adding feature flag')
        day_ice_clouds = day & np.isin(feature_data['cloud_type'],
                                       ICE_TYPES)
        day_water_clouds = day & np.isin(feature_data['cloud_type'],
                                         WATER_TYPES)
        flag = np.full(day_ice_clouds.shape, 'night', dtype=object)
        flag[day] = 'clear'
        flag[day_ice_clouds] = 'ice_cloud'
        flag[day_water_clouds] = 'water_cloud'
        flag[day_missing_ctype] = 'bad_cloud'
        flag[day_missing_opd] = 'bad_cloud'
        flag[day_missing_reff] = 'bad_cloud'
        feature_data['flag'] = flag
        logger.debug('Created the "flag" dataset with the following unique '
                     'values: {}'.format(np.unique(flag)))

        mem = psutil.virtual_memory()
        logger.debug('Cleaned feature data dict has these keys: {}'
                     .format(feature_data.keys()))
        logger.debug('Cleaned feature data dict values have these shapes: {}'
                     .format([d.shape for d in feature_data.values()]))
        logger.debug('Feature flag column has these values: {}'
                     .format(np.unique(feature_data['flag'])))
        logger.info('Cleaning took {:.1f} seconds'.format(time.time() - t0))
        logger.info('Memory utilization is '
                    '{:.3f} GB out of {:.3f} GB total ({:.2f}% used).'
                    .format(mem.used / 1e9, mem.total / 1e9,
                            100 * mem.used / mem.total))

        return feature_data, fill_flag

    def archive_cld_properties(self):
        """
        Archive original cloud property (cld_*) .h5 files. This method creates
        .tmp files in a ./raw/ sub directory. mark_complete_archived_files()
        should be run at the end to remove the .tmp designation. This will
        signify that the cloud fill was completed successfully.
        """
        cld_dsets = ['cld_opd_dcomp', 'cld_reff_dcomp', 'cld_press_acha',
                     'cloud_type']
        for dset in cld_dsets:
            src_fpath = self.dset_map[dset]
            src_dir, f_name = os.path.split(src_fpath)
            dst_dir = os.path.join(src_dir, 'raw')
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)

            dst_fpath = os.path.join(dst_dir, f_name)
            if os.path.exists(dst_fpath):
                msg = ("A raw cloud file already exists, this suggests "
                       "MLClouds gap fill has already been run: {}"
                       .format(dst_fpath))
                logger.error(msg)
                raise RuntimeError(msg)
            else:
                logger.debug('Archiving {} to {}'
                             .format(src_fpath, dst_fpath + '.tmp'))
                shutil.copy(src_fpath, dst_fpath + '.tmp')

    def mark_complete_archived_files(self):
        """Remove the .tmp marker from the archived files once MLCloudsFill
        is complete"""
        cld_dsets = ['cld_opd_dcomp', 'cld_reff_dcomp', 'cld_press_acha',
                     'cloud_type']
        for dset in cld_dsets:
            fpath = self.dset_map[dset]
            src_dir, f_name = os.path.split(fpath)
            raw_path = os.path.join(src_dir, 'raw', f_name)
            logger.debug('Renaming .tmp raw file to {}'.format(raw_path))
            if os.path.exists(raw_path + '.tmp'):
                os.rename(raw_path + '.tmp', raw_path)
            else:
                msg = ('Something went wrong. The .tmp file created at the '
                       'beginning of MLCloudsFill no longer exists: {}'
                       .format(raw_path + '.tmp'))
                logger.error(msg)
                raise FileNotFoundError(msg)

    def predict_cld_properties(self, feature_data, col_slice=None):
        """
        Predict cloud properties with phygnn

        Parameters
        ----------
        feature_data : dict
            Clean feature data without gaps. keys are the feature names
            (nsrdb dataset names), values are 2D numpy arrays (time x sites).
        col_slice : slice
            Column slice of the resource data to work on. This is a result of
            chunking the columns to reduce memory load. Just used for logging
            in this method.

        Returns
        -------
        predicted_data : dict
            Dictionary of predicted cloud properties. Keys are nsrdb dataset
            names, values are 2D arrays of phygnn-predicted values
            (time x sites).
        """

        L = feature_data['flag'].shape[0] * feature_data['flag'].shape[1]
        cols = [k for k in feature_data.keys()
                if k in self.phygnn_model.feature_names
                or k == 'flag']
        feature_df = pd.DataFrame(index=np.arange(L), columns=cols)
        for dset, arr in feature_data.items():
            if dset in feature_df:
                feature_df[dset] = arr.flatten(order='F')

        # Predict on night timesteps as if they were clear
        # the phygnn model wont be trained with the night category
        # so this is an easy way to keep the full data shape and
        # cooperate with the feature names that phygnn expects
        night_mask = feature_df['flag'] == 'night'
        feature_df.loc[night_mask, 'flag'] = 'clear'

        logger.debug('Predicting gap filled cloud data for column slice {}'
                     .format(col_slice))
        labels = self.phygnn_model.predict(feature_df,
                                           table=False)
        mem = psutil.virtual_memory()
        logger.debug('Prediction complete for column slice {}. Memory '
                     'utilization is {:.3f} GB out of {:.3f} GB total '
                     '({:.2f}% used).'
                     .format(col_slice, mem.used / 1e9, mem.total / 1e9,
                             100 * mem.used / mem.total))
        logger.debug('Label data shape: {}'.format(labels.shape))

        shape = feature_data['flag'].shape
        predicted_data = {}
        for i, dset in enumerate(self.phygnn_model.label_names):
            logger.debug('Reshaping predicted {} to {}'
                         .format(dset, shape))
            predicted_data[dset] = labels[:, i].reshape(shape, order='F')

        for dset, arr in predicted_data.items():
            nnan = np.isnan(arr).sum()
            ntot = arr.shape[0] * arr.shape[1]
            logger.debug('Raw predicted data for {} for column slice {} '
                         'has mean: {:.2f}, median: {:.2f}, range: '
                         '({:.2f}, {:.2f}) and {} NaN values out of '
                         '{} ({:.2f}%)'
                         .format(dset, col_slice, np.nanmean(arr),
                                 np.median(arr), np.nanmin(arr),
                                 np.nanmax(arr), nnan, ntot,
                                 100 * nnan / ntot))

        return predicted_data

    def fill_bad_cld_properties(self, predicted_data, feature_data,
                                col_slice=None):
        """
        Fill bad cloud properties in the feature data from predicted data

        Parameters
        ----------
        predicted_data : dict
            Dictionary of predicted cloud properties. Keys are nsrdb dataset
            names, values are 2D arrays of phygnn-predicted values
            (time x sites).
        feature_data : dict
            Clean feature data without gaps. keys are the feature names
            (nsrdb dataset names), values are 2D numpy arrays (time x sites).
        col_slice : slice
            Column slice of the resource data to work on. This is a result of
            chunking the columns to reduce memory load. Just used for logging
            in this method.

        Returns
        -------
        filled_data : dict
            Dictionary of filled cloud properties. Keys are nsrdb dataset
            names, values are 2D arrays of phygnn-predicted values
            (time x sites). The filled data is a combination of the input
            predicted_data and feature_data. The datasets in the predicted_data
            input are used to fill the feature_data input where:
            (feature_data['flag'] == "bad_cloud")
        """

        fill_mask = feature_data['flag'] == 'bad_cloud'
        night_mask = feature_data['flag'] == 'night'
        clear_mask = feature_data['flag'] == 'clear'
        cloud_mask = ((feature_data['flag'] != 'night')
                      | (feature_data['flag'] != 'clear'))
        if fill_mask.sum() == 0:
            msg = ('No "bad_cloud" flags were detected in the feature_data '
                   '"flag" dataset. Something went wrong! '
                   'The cloud data is never perfect...')
            logger.warning(msg)
            warn(msg)

        if self._fill_all:
            logger.debug('Filling {} values (all cloudy timesteps) using '
                         'MLClouds predictions for column slice {}'
                         .format(np.sum(cloud_mask), col_slice))
        else:
            logger.debug('Filling {} values using MLClouds predictions for '
                         'column slice {}'
                         .format(np.sum(fill_mask), col_slice))

        filled_data = {}
        for dset, arr in predicted_data.items():
            varobj = VarFactory.get_base_handler(dset, var_meta=self._var_meta)
            arr = np.maximum(arr, varobj.physical_min)
            arr = np.minimum(arr, varobj.physical_max)

            cld_data = feature_data[dset]
            if self._fill_all:
                cld_data[cloud_mask] = arr[cloud_mask]
            else:
                cld_data[fill_mask] = arr[fill_mask]

            cld_data[night_mask | clear_mask] = 0
            filled_data[dset] = cld_data

            nnan = np.isnan(filled_data[dset]).sum()
            ntot = filled_data[dset].shape[0] * filled_data[dset].shape[1]
            logger.debug('Final cleaned data for {} for column slice {} '
                         'has mean: {:.2f}, median: {:.2f}, range: '
                         '({:.2f}, {:.2f}) and {} NaN values out of '
                         '{} ({:.2f}%)'
                         .format(dset, col_slice,
                                 np.nanmean(filled_data[dset]),
                                 np.median(filled_data[dset]),
                                 np.nanmin(filled_data[dset]),
                                 np.nanmax(filled_data[dset]),
                                 nnan, ntot,
                                 100 * nnan / ntot))

            if nnan > 0:
                msg = ('Gap filled cloud property "{}" still had '
                       '{} NaN values!'.format(dset, nnan))
                logger.error(msg)
                raise RuntimeError(msg)

        return filled_data

    @staticmethod
    def fill_ctype_press(h5_source, col_slice=slice(None)):
        """Fill cloud type and pressure using simple temporal nearest neighbor.

        Parameters
        ----------
        h5_source : str
            Path to directory containing multi-file resource file sets.
            Available formats:
                /h5_dir/
                /h5_dir/prefix*suffix
        col_slice : slice
            Column slice of the resource data to work on. This is a result of
            chunking the columns to reduce memory load. Use slice(None) for
            no chunking.

        Returns
        -------
        cloud_type : np.ndarray
            2D array (time x sites) of gap-filled cloud type data.
        cloud_pres : np.ndarray
            2D array (time x sites) of gap-filled cld_press_acha data.
        sza : np.ndarray
            2D array (time x sites) of solar zenith angle data.
        fill_flag : np.ndarray
            Integer array of flags showing what data was filled and why.
        """

        fill_flag = None
        with MultiFileNSRDB(h5_source) as f:
            cloud_type = f['cloud_type', :, col_slice]
            cloud_pres = f['cld_press_acha', :, col_slice]
            sza = f['solar_zenith_angle', :, col_slice]

        cloud_type, fill_flag = CloudGapFill.fill_cloud_type(
            cloud_type, fill_flag=fill_flag)

        cloud_pres, fill_flag = CloudGapFill.fill_cloud_prop(
            'cld_press_acha', cloud_pres, cloud_type, sza,
            fill_flag=fill_flag, cloud_type_is_clean=True)

        return cloud_type, cloud_pres, sza, fill_flag

    def write_filled_data(self, filled_data, col_slice=slice(None)):
        """Write gap filled cloud data to disk

        Parameters
        ----------
        filled_data : dict
            Dictionary of filled cloud properties. Keys are nsrdb dataset
            names, values are 2D arrays of phygnn-predicted values
            (time x sites). The filled data is a combination of the input
            predicted_data and feature_data. The datasets in the predicted_data
            input are used to fill the feature_data input where:
            (feature_data['flag'] == "bad_cloud")
        col_slice : slice
            Column slice of the resource data to work on. This is a result of
            chunking the columns to reduce memory load. Use slice(None) for
            no chunking.
        """

        for dset, arr in filled_data.items():
            fpath = self.dset_map[dset]
            with Outputs(fpath, mode='a') as f:
                logger.info('Writing filled "{}" to: {}'
                            .format(dset, os.path.basename(fpath)))
                f[dset, :, col_slice] = arr
                logger.debug('Finished writing "{}".'.format(dset))

    def write_fill_flag(self, fill_flag, col_slice=slice(None)):
        """Write the fill flag dataset to its daily file next to the cloud
        property files.

        Parameters
        ----------
        fill_flag : np.ndarray
            Integer array of flags showing what data was filled and why.
        col_slice : slice
            Column slice of the resource data to work on. This is a result of
            chunking the columns to reduce memory load. Use slice(None) for
            no chunking.
        """
        fpath_opd = self._dset_map['cld_opd_dcomp']
        fpath = fpath_opd.replace('cld_opd_dcomp', 'cloud_fill_flag')
        var_obj = VarFactory.get_base_handler('cloud_fill_flag',
                                              var_meta=self._var_meta)

        with Resource(fpath_opd) as res:
            ti = res.time_index
            meta = res.meta

        logger.info('Writing cloud_fill_flag to: {}'
                    .format(os.path.basename(fpath)))

        if not os.path.exists(fpath):
            with Outputs(fpath, mode='w') as fout:
                fout.time_index = ti
                fout.meta = meta
                init_data = np.zeros((len(ti), len(meta)))
                fout._add_dset(dset_name='cloud_fill_flag',
                               dtype=var_obj.final_dtype, data=init_data,
                               chunks=var_obj.chunks, attrs=var_obj.attrs)

        with Outputs(fpath, mode='a') as fout:
            fout['cloud_fill_flag', :, col_slice] = fill_flag

        logger.debug('Write complete')
        logger.info('Final cloud_fill_flag counts:')
        ntot = fill_flag.shape[0] * fill_flag.shape[1]
        for n in range(10):
            count = (fill_flag == n).sum()
            logger.info('\tFlag {} has {} counts out of {} ({:.2f}%)'
                        .format(n, count, ntot, 100 * count / ntot))

    @classmethod
    def _prep_chunk(cls, h5_source, model_path=None, var_meta=None, sza_lim=90,
                    col_slice=slice(None)):
        """Prepare a column chunk (slice) of data for phygnn prediction.

        Parameters
        ----------
        h5_source : str
            Path to directory containing multi-file resource file sets.
            Available formats:
                /h5_dir/
                /h5_dir/prefix*suffix
        model_path : str | None
            Directory to load phygnn model from. This is typically a fpath to
            a .pkl file with an accompanying .json file in the same directory.
            None will try to use the default model path from the mlclouds
            project directory.
        var_meta : str | pd.DataFrame | None
            CSV file or dataframe containing meta data for all NSRDB variables.
            Defaults to the NSRDB var meta csv in git repo.
        sza_lim : int, optional
            Solar zenith angle limit below which missing cloud property data
            will be gap filled. By default 90 to fill all missing daylight data
        col_slice : slice
            Column slice of the resource data to work on. This is a result of
            chunking the columns to reduce memory load. Use slice(None) for
            no chunking.

        Returns
        -------
        feature_data : dict
            Clean feature data without gaps. keys are the feature names
            (nsrdb dataset names), values are 2D numpy arrays (time x sites).
            This is just for the col_slice being worked on.
        clean_data : dict
            Dictionary of filled cloud properties. Keys are nsrdb dataset
            names, values are 2D arrays (time x sites) and have
            nearest-neighbor cleaned values for cloud pressure and type
            This is just for the col_slice being worked on.
        fill_flag : np.ndarray
            Integer array of flags showing what data was filled and why.
            This is just for the col_slice being worked on.
        """
        obj = cls(h5_source, model_path=model_path, var_meta=var_meta)

        logger.debug('Preparing data for MLCloudsFill for column slice {}'
                     .format(col_slice))
        ctype, cpres, sza, fill_flag = obj.fill_ctype_press(
            obj.h5_source, col_slice=col_slice)
        clean_data = {'cloud_type': ctype, 'cld_press_acha': cpres}
        feature_data = {'cloud_type': ctype,
                        'cld_press_acha': cpres,
                        'solar_zenith_angle': sza}
        feature_data = obj.parse_feature_data(feature_data=feature_data,
                                              col_slice=col_slice)
        feature_data, fill_flag = obj.clean_feature_data(
            feature_data, fill_flag, sza_lim=sza_lim)
        logger.debug('Completed MLClouds data prep for column slice {}'
                     .format(col_slice))

        return feature_data, clean_data, fill_flag

    def _process_chunk(self, i_features, i_clean, i_flag, col_slice,
                       clean_data, fill_flag):
        """Use cleaned and prepared data to run phygnn predictions and create
        final filled data for a single column chunk.

        Parameters
        ----------
        i_features : dict
            Clean feature data without gaps. keys are the feature names
            (nsrdb dataset names), values are 2D numpy arrays (time x sites).
            This is just for a single column chunk (col_slice).
        i_clean : dict
            Dictionary of filled cloud properties. Keys are nsrdb dataset
            names, values are 2D arrays (time x sites) of phygnn-predicted
            values (cloud opd and reff) or nearest-neighbor cleaned values
            (cloud pressure and type).
            This is just for a single column chunk (col_slice).
        i_flag : np.ndarray
            Integer array of flags showing what data was filled and why.
            This is just for a single column chunk (col_slice).
        col_slice : slice
            Column slice of the resource data to work on. This is a result of
            chunking the columns to reduce memory load.
        clean_data : dict
            Dictionary of filled cloud properties. Keys are nsrdb dataset
            names, values are 2D arrays (time x sites).
            This is for ALL chunks (full resource shape).
        fill_flag : np.ndarray
            Integer array of flags showing what data was filled and why.
            This is for ALL chunks (full resource shape).

        Returns
        -------
        clean_data : dict
            Dictionary of filled cloud properties. Keys are nsrdb dataset
            names, values are 2D arrays (time x sites). This has been updated
            with phygnn-predicted values (cloud opd and reff) or
            nearest-neighbor cleaned values (cloud pressure and type)
            This is for ALL chunks (full resource shape).
        fill_flag : np.ndarray
            Integer array of flags showing what data was filled and why.
            This is for ALL chunks (full resource shape).
        """

        i_predicted = self.predict_cld_properties(i_features,
                                                  col_slice=col_slice)
        i_filled = self.fill_bad_cld_properties(i_predicted, i_features,
                                                col_slice=col_slice)
        i_clean.update(i_filled)

        fill_flag[:, col_slice] = i_flag
        for k, v in clean_data.items():
            v[:, col_slice] = i_clean[k]

        return clean_data, fill_flag

    @classmethod
    def run(cls, h5_source, fill_all=False, model_path=None, var_meta=None,
            sza_lim=90, col_chunk=None, max_workers=None):
        """
        Fill cloud properties using phygnn predictions. Original files will be
        archived to a new "raw/" sub-directory

        Parameters
        ----------
        h5_source : str
            Path to directory containing multi-file resource file sets.
            Available formats:
                /h5_dir/
                /h5_dir/prefix*suffix
        fill_all : bool
            Flag to fill all cloud properties for all timesteps where
            cloud_type is cloudy.
        model_path : str | None
            Directory to load phygnn model from. This is typically a fpath to
            a .pkl file with an accompanying .json file in the same directory.
            None will try to use the default model path from the mlclouds
            project directory.
        var_meta : str | pd.DataFrame | None
            CSV file or dataframe containing meta data for all NSRDB variables.
            Defaults to the NSRDB var meta csv in git repo.
        sza_lim : int, optional
            Solar zenith angle limit below which missing cloud property data
            will be gap filled. By default 90 to fill all missing daylight data
        col_chunk : None | int
            Optional chunking method to gap fill one column chunk at a time
            to reduce memory requirements. If provided, this should be an
            integer specifying how many columns to work on at one time.
        max_workers : None | int
            Maximum workers for running mlclouds in parallel. 1 is serial and
            None uses all available workers.
        """

        logger.info('Running MLCloudsFill with h5_source: {}'
                    .format(h5_source))
        logger.info('Running MLCloudsFill with model: {}'
                    .format(model_path))
        obj = cls(h5_source, fill_all=fill_all, model_path=model_path,
                  var_meta=var_meta)
        obj.archive_cld_properties()
        clean_data, fill_flag = obj._init_clean_arrays()

        if col_chunk is None:
            slices = [slice(None)]
        else:
            columns = np.arange(obj._res_shape[1])
            N = np.ceil(len(columns) / col_chunk)
            arrays = np.array_split(columns, N)
            slices = [slice(a[0], 1 + a[-1]) for a in arrays]
            logger.info('Gap fill will be run across the full data column '
                        'shape {} in {} column chunks with approximately {} '
                        'sites per chunk'
                        .format(len(columns), len(slices), col_chunk))

        if max_workers == 1:
            for col_slice in slices:
                out = obj._prep_chunk(h5_source, model_path=model_path,
                                      var_meta=var_meta, sza_lim=sza_lim,
                                      col_slice=col_slice)
                i_features, i_clean, i_flag = out
                out = obj._process_chunk(i_features, i_clean, i_flag,
                                         col_slice, clean_data, fill_flag)
                clean_data, fill_flag = out
                del i_features, i_clean, i_flag

        else:
            futures = {}
            logger.info('Starting process pool for parallel phygnn cloud '
                        'fill with {} workers.'.format(max_workers))
            with ProcessPoolExecutor(max_workers=max_workers) as exe:
                for col_slice in slices:
                    future = exe.submit(obj._prep_chunk, h5_source,
                                        model_path=model_path,
                                        var_meta=var_meta,
                                        sza_lim=sza_lim,
                                        col_slice=col_slice)
                    futures[future] = col_slice

                logger.info('Kicked off {} futures.'.format(len(futures)))
                for i, future in enumerate(as_completed(futures)):
                    col_slice = futures[future]
                    i_features, i_clean, i_flag = future.result()
                    out = obj._process_chunk(i_features, i_clean, i_flag,
                                             col_slice, clean_data, fill_flag)
                    clean_data, fill_flag = out
                    del i_features, i_clean, i_flag, future

                    mem = psutil.virtual_memory()
                    logger.info('MLCloudsFill futures completed: '
                                '{0} out of {1}. '
                                'Current memory usage is '
                                '{2:.3f} GB out of {3:.3f} GB total.'
                                .format(i + 1, len(futures),
                                        mem.used / 1e9, mem.total / 1e9))

        obj.write_filled_data(clean_data, col_slice=slice(None))
        obj.write_fill_flag(fill_flag, col_slice=slice(None))
        obj.mark_complete_archived_files()
        logger.info('Completed MLCloudsFill for h5_source: {}'
                    .format(h5_source))