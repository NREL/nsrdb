# -*- coding: utf-8 -*-
"""
Created on Fri April 26 2019

@author: gbuster
"""
from copy import deepcopy
import numpy as np
import pandas as pd
from warnings import warn
from nsrdb.all_sky import WATER_TYPES, ICE_TYPES, CLEAR_TYPES


class CloudGapFill:
    """Framework to fill gaps in cloud properties."""

    # cloud categories
    CATS = {'water': WATER_TYPES, 'ice': ICE_TYPES}

    # hard fill for sites that have full year w no data.
    FILL = {'water': {'cld_opd_dcomp': 10.0,
                      'cld_reff_dcomp': 8.0,
                      'cloud_press_acha': 800.0},
            'ice': {'cld_opd_dcomp': 5.0,
                    'cld_reff_dcomp': 20.0,
                    'cloud_press_acha': 250.0}}

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

        # gap fill all missing values
        cloud_prop_fill = cloud_prop_fill.interpolate(
            method='nearest', axis=0)\
            .fillna(method='ffill').fillna(method='bfill')

        # fill values for the specified cloud type
        cloud_prop[type_mask] = cloud_prop_fill[type_mask].values

        return cloud_prop

    @staticmethod
    def make_zeros(cloud_prop, cloud_type, sza, sza_lim=89):
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
            loc = list(loc)
            warn('NaN values persist at the following sites: {}'
                 .format(loc))

            # do a hard fix of remaining nan values
            for cat, types in CloudGapFill.CATS.items():
                mask = (cloud_type.isin(types) & cloud_prop.isnull())
                cloud_prop[mask] = CloudGapFill.FILL[cat][dset]

        return cloud_prop

    @staticmethod
    def fill_cloud_type(cloud_type, missing=-15):
        """Fill the cloud type data.

        Parameters
        ----------
        cloud_type : pd.DataFrame
            Integer cloud type data with missing flags.
        missing : int
            Flag for missing cloud types.

        Returns
        -------
        cloud_type : pd.DataFrame
            Integer cloud type data with missing values filled using the
            temporal nearest neighbor.
        """

        cloud_type = cloud_type.astype(np.float32)
        cloud_type[cloud_type == missing] = np.nan
        cloud_type = cloud_type.interpolate(method='nearest', axis=0)\
            .fillna(method='ffill').fillna(method='bfill')

        # if bad data remains, it means full year is bad, set to clear.
        if any(pd.isnull(cloud_type)) is True:
            loc = np.where(pd.isnull(cloud_type).any(axis=0) is True)[0]
            warn('The following sites have missing cloud types for the '
                 'entire year: {}'.format(list(loc)))
            cloud_type[pd.isnull(cloud_type)] = 0

        cloud_type = cloud_type.astype(np.int8)

        return cloud_type

    @classmethod
    def fill_cloud_prop(cls, prop_name, cloud_prop, cloud_type, sza):
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

        Returns
        -------
        cloud_prop : pd.DataFrame
            DataFrame of cloud property values with no remaining NaN's.
        """

        # fill cloud types.
        cloud_type = cls.fill_cloud_type(cloud_type)

        # set missing property values to NaN
        cloud_prop[cloud_prop <= 0] = np.nan

        # perform gap fill for each cloud category seperately
        for category, _ in cls.CATS.items():
            # make property fill for given cloud type set
            cloud_prop = cls.fill_cloud_cat(category, cloud_prop, cloud_type)

        # set clear and night cloud properties to zero
        cloud_prop = cls.make_zeros(cloud_prop, cloud_type, sza)

        # handle persistent nan properties
        cloud_prop = cls.handle_persistent_nan(prop_name, cloud_prop,
                                               cloud_type)
        return cloud_prop
