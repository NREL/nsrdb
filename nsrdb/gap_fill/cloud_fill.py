# -*- coding: utf-8 -*-
"""
Created on Fri April 26 2019

@author: gbuster
"""
from copy import deepcopy
import logging
import numpy as np
import pandas as pd
from warnings import warn

from nsrdb.all_sky import (WATER_TYPES, ICE_TYPES, CLEAR_TYPES, CLOUD_TYPES,
                           SZA_LIM)
from nsrdb.file_handlers.resource import Resource
from nsrdb.file_handlers.outputs import Outputs


logger = logging.getLogger(__name__)


class CloudGapFill:
    """Framework to fill gaps in cloud properties."""

    # cloud categories
    CATS = {'water': WATER_TYPES, 'ice': ICE_TYPES}

    # hard fill for sites that have full year w no data.
    FILL = {'water': {'cld_opd_dcomp': 10.0,
                      'cld_reff_dcomp': 8.0,
                      'cld_press_acha': 800.0},
            'ice': {'cld_opd_dcomp': 5.0,
                    'cld_reff_dcomp': 20.0,
                    'cld_press_acha': 250.0}}

    @staticmethod
    def fill_cloud_cat(category, cloud_prop, cloud_type):
        """Fill cloud properties of a given category (water, ice).

        Parameters
        ----------
        category : str
            'water' or 'ice'
        cloud_prop : pd.DataFrame
            DataFrame of cloud properties (single-property) with shape
            (time x sites). Missing values must be NaN.
        cloud_type : pd.DataFrame
            Integer cloud type data with no missing values.

        Returns
        -------
        cloud_prop : pd.DataFrame
            DataFrame of cloud properties with missing values filled for
            cloud types in the specified category.
        """

        types = CloudGapFill.CATS[category]

        # make a copy of master cloud_prop for gap filling one category
        cloud_prop_fill = deepcopy(cloud_prop)

        # find locations for current cloud property
        type_mask = cloud_type.isin(types)

        # set other locations to nan to not impede gap fill
        cloud_prop_fill[~type_mask] = np.nan

        # patch sites with all NaN or just one non-NaN but warn
        all_na = (np.isnan(cloud_prop_fill).sum(axis=0) >=
                  (cloud_prop_fill.shape[0] - 1))
        if any(all_na):
            cloud_prop_fill.loc[:, all_na] = -999

        # gap fill all missing values
        cloud_prop_fill = cloud_prop_fill.interpolate(
            method='nearest', axis=0)\
            .fillna(method='ffill').fillna(method='bfill')

        # Make sure sites with only NaN props stay as nan
        cloud_prop_fill.loc[:, all_na] = np.nan

        # fill values for the specified cloud type
        cloud_prop[type_mask] = cloud_prop_fill[type_mask].values

        return cloud_prop

    @staticmethod
    def make_zeros(cloud_prop, cloud_type, sza, sza_lim=SZA_LIM):
        """set clear and night cloud properties to zero

        Parameters
        ----------
        cloud_prop : pd.DataFrame
            DataFrame of cloud properties (single-property) with shape
            (time x sites).
        cloud_type : pd.DataFrame
            Integer cloud type data with no missing values.
        sza : pd.DataFrame
            DataFrame of solar zenith angle values to determine nighttime.
        sza_lim : int | float
            Value above which sza indicates nighttime (sun below horizon).

        Returns
        -------
        cloud_prop : pd.DataFrame
            DataFrame of cloud properties with properties during clear and
            night time timesteps set to zero.
        """

        # set clear and night cloud properties to zero
        cloud_prop[cloud_type.isin(CLEAR_TYPES)] = 0.0
        cloud_prop[sza > sza_lim] = 0.0

        return cloud_prop

    @staticmethod
    def handle_persistent_nan(dset, cloud_prop, cloud_type):
        """Handle any remaining NaN property values and warn.

        Parameters
        ----------
        dset : str
            Name of the cloud property being filled.
        cloud_prop : pd.DataFrame
            DataFrame of cloud property values. Fill should have been
            attempted, this method catches any remaining NaN values.
        cloud_type : pd.DataFrame
            Integer cloud type data with no missing values.

        Returns
        -------
        cloud_prop : pd.DataFrame
            DataFrame of cloud property values with no remaining NaN's.
        """

        if np.sum(np.isnan(cloud_prop.values)) > 0:
            loc = np.where(np.sum(np.isnan(cloud_prop.values), axis=0) > 0)[0]
            warn('NaN values persist at {} sites.'.format(len(loc)))

            # do a hard fix of remaining nan values
            for cat, types in CloudGapFill.CATS.items():
                mask = (cloud_type.isin(types) & cloud_prop.isnull())
                cloud_prop[mask] = CloudGapFill.FILL[cat][dset]

        return cloud_prop

    @staticmethod
    def fill_cloud_type(cloud_type, fill_flag=None, missing=-15):
        """Fill the cloud type data.

        Parameters
        ----------
        cloud_type : pd.DataFrame
            Integer cloud type data with missing flags.
        missing : int
            Flag for missing cloud types.
        fill_flag : np.ndarray
            Integer array of flags showing what data was filled and why.

        Returns
        -------
        cloud_type : pd.DataFrame
            Integer cloud type data with missing values filled using the
            temporal nearest neighbor.
        fill_flag : np.ndarray
            Integer array of flags showing what data was filled and why.
        """

        df_convert = False
        if isinstance(cloud_type, np.ndarray):
            df_convert = True
            cloud_type = pd.DataFrame(cloud_type)

        missing_mask = (cloud_type.values == missing)

        if fill_flag is None:
            fill_flag = np.zeros(cloud_type.shape, dtype=np.int8)
        fill_flag[missing_mask] = 1

        if missing_mask.all(axis=0).any():
            # full timeseries with no cloud type. set to clear and warn
            fill_flag[:, missing_mask.all(axis=0)] = 2
            cloud_type.loc[:, missing_mask.all(axis=0)] = 0
            warn('{} sites have missing cloud types for the '
                 'entire year.'.format(np.sum(missing_mask.all(axis=0))))
            # reset missing mask
            missing_mask = (cloud_type.values == missing)

        if missing_mask.any():
            cloud_type = cloud_type.astype(np.float32)
            cloud_type[missing_mask] = np.nan
            cloud_type = cloud_type.interpolate(method='nearest', axis=0)\
                .fillna(method='ffill').fillna(method='bfill')

            cloud_type = cloud_type.astype(np.int8)

        if df_convert:
            cloud_type = cloud_type.values

        return cloud_type, fill_flag

    @classmethod
    def fill_cloud_prop(cls, prop_name, cloud_prop, cloud_type, sza,
                        fill_flag=None):
        """Perform full cloud property fill.

        Parameters
        ----------
        prop_name : str
            Name of the cloud property being filled.
        cloud_prop : pd.DataFrame
            DataFrame of cloud property values.
        cloud_type : pd.DataFrame
            Integer cloud type data with no missing values.
        sza : pd.DataFrame
            DataFrame of solar zenith angle values to determine nighttime.
        fill_flag : np.ndarray
            Integer array of flags showing what data was filled and why.

        Returns
        -------
        cloud_prop : pd.DataFrame
            DataFrame of cloud property values with no remaining NaN's.
        fill_flag : np.ndarray
            Integer array of flags showing what data was filled and why.
        """

        logger.debug('Gap filling "{}".'.format(prop_name))

        float_convert = False
        if np.issubdtype(cloud_prop.dtype, np.integer):
            float_convert = True
            native_dtype = cloud_prop.dtype
            cloud_prop = cloud_prop.astype(np.float32)

        # make dataframes
        df_convert = False
        if isinstance(cloud_prop, np.ndarray):
            df_convert = True
            cloud_prop = pd.DataFrame(cloud_prop)
        if isinstance(cloud_type, np.ndarray):
            cloud_type = pd.DataFrame(cloud_type)
        if isinstance(sza, np.ndarray):
            sza = pd.DataFrame(sza)

        if fill_flag is None:
            fill_flag = np.zeros(cloud_type.shape, dtype=np.int8)

        # fill cloud types.
        cloud_type, fill_flag = cls.fill_cloud_type(cloud_type,
                                                    fill_flag=fill_flag)

        # find location of missing properties
        missing_prop = (cloud_type.isin(CLOUD_TYPES) & (cloud_prop <= 0))
        fill_flag[(missing_prop.values & (fill_flag == 0))] = 3

        if missing_prop.all(axis=0).any():
            # if full timeseries is missing properties but not type, set 4
            all_missing_ctype = (fill_flag == 2).all(axis=0)
            fill_flag[:, (missing_prop.all(axis=0) & ~all_missing_ctype)] = 4

        # set missing property values to NaN. Clear will be reset later.
        cloud_prop[(cloud_prop <= 0)] = np.nan

        # perform gap fill for each cloud category seperately
        for category, _ in cls.CATS.items():
            # make property fill for given cloud type set
            cloud_prop = cls.fill_cloud_cat(category, cloud_prop, cloud_type)

        # set clear and night cloud properties to zero
        cloud_prop = cls.make_zeros(cloud_prop, cloud_type, sza)

        # handle persistent nan properties
        cloud_prop = cls.handle_persistent_nan(prop_name, cloud_prop,
                                               cloud_type)

        if df_convert:
            cloud_prop = cloud_prop.values

        if float_convert:
            cloud_prop = cloud_prop.astype(native_dtype)

        return cloud_prop, fill_flag

    @classmethod
    def fill_file(cls, f_cloud, rows=slice(None), cols=slice(None),
                  col_chunk=None):
        """Gap fill cloud properties in an h5 file.

        Parameters
        ----------
        f_cloud : str
            File path to a cloud file with datasets 'cloud_type',
            'solar_zenith_angle', 'fill_flag', and some cloud property
            dataset(s) with prefix 'cld_'.
        rows : slice
            Subset of rows to gap fill.
        cols : slice
            Subset of columns to gap fill.
        col_chunks : None
            Optional chunking method to gap fill a few chunks at a time
            to reduce memory requirements.
        """
        logger.info('Patching cloud properties in file: "{}"'.format(f_cloud))

        with Resource(f_cloud) as f:
            dsets = f.dsets
            shape = f.shape

        start = cols.start
        stop = cols.stop
        if start is None:
            start = 0
        if stop is None:
            stop = shape[1]

        if col_chunk is None:
            col_chunk = stop - start

        last = start

        while True:
            i0 = last
            i1 = np.min((i0 + col_chunk, shape[1]))
            cols = slice(i0, i1)
            last = i1
            if i0 == i1:
                logger.debug('Job complete at index {}'.format(i0))
                break
            else:
                logger.debug('Patching cloud properties: {} through {}'
                             .format(i0, i1))

            with Resource(f_cloud) as f:
                cloud_type = f['cloud_type', rows, cols]
                sza = f['solar_zenith_angle', rows, cols]

            cloud_type, fill_flag = cls.fill_cloud_type(cloud_type)

            with Outputs(f_cloud, mode='a') as f:
                f['cloud_type', rows, cols] = cloud_type

            for dset in dsets:
                if 'cld_' in dset:
                    with Resource(f_cloud) as f:
                        cloud_prop = f[dset, rows, cols]

                    cloud_prop, fill_flag = cls.fill_cloud_prop(
                        dset, cloud_prop, cloud_type, sza, fill_flag=fill_flag)

                    with Outputs(f_cloud, mode='a') as f:
                        f[dset, rows, cols] = cloud_prop

            with Outputs(f_cloud, mode='a') as f:
                f['fill_flag', rows, cols] = fill_flag
