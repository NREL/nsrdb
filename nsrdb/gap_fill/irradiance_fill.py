# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 2018

@author: gbuster
"""
import numpy as np
import pandas as pd
from warnings import warn

from farms import CLOUD_TYPES, SZA_LIM


def missing_cld_props(cloud_type, cld_opd_dcomp, cld_reff_dcomp):
    """Make a boolean mask showing where there are missing cloud properties.

    Parameters
    ----------
    cloud_type : np.ndarray
        Array of integer cloud types.
    cloud_opd_dcomp : np.ndarray
        Array of cloud optical depths. Expected range is 0 - 160 with
        missing values <= 0.
    cld_reff_dcomp : np.ndarray
        Array of cloud effective partical radii. Expected range is 0 - 160
        with missing values <= 0.

    Returns
    -------
    missing_props : np.ndarray
        Boolean array with True values where there are clouds but no cloud
        properties.
    """

    missing_props = ((np.isin(cloud_type, CLOUD_TYPES))
                     & ((cld_opd_dcomp <= 0) | (cld_reff_dcomp <= 0)
                        | (np.isnan(cld_opd_dcomp)
                        | (np.isnan(cld_reff_dcomp)))))

    return missing_props


def make_fill_flag(irrad, cs_irrad, cloud_type, missing_cld_props,
                   cloud_fill_flag=None):
    """Make a dataset indicating where to fill bad irradiance data.

    Parameters
    ----------
    irrad : np.ndarray
        All-sky (cloudy + clear) irradiance data.
    cs_irrad : np.ndarray
        Clearsky (REST) irradiance data.
    cloud_type : np.ndarray
        Array of numerical cloud types.
    missing_cld_props : np.ndarray
        Boolean array flagging timesteps with missing cloud properties.
    cloud_fill_flag : None | np.ndarray
        Integer array of flags showing what data was previously filled and why.
        None will create a new fill flag initialized as all zeros.
        An array input will be interpreted as flags showing which cloud
        properties have already been filled.

    Returns
    -------
    fill_flag : np.ndarray
        Array of integers signifying whether to fill the irradiance
        data and the reason why.
    """

    # Make a categorical numerical fill flag, uint8 is the default dtype of
    # fill_flag as of 2021
    new_fill_flag = np.zeros_like(irrad).astype(np.uint8)

    # make fill flags
    new_fill_flag[(cloud_type == -15)] = 1
    new_fill_flag[:, (cloud_type == -15).all(axis=0)] = 2
    new_fill_flag[missing_cld_props & (cs_irrad > 0)] = 3
    new_fill_flag[:, missing_cld_props.all(axis=0)] = 4
    # clearsky limit (fill flag 5) is filled in enforce_clearsky()
    new_fill_flag[np.isnan(irrad) | (irrad < 0)] = 6

    if cloud_fill_flag is not None:
        return np.where(new_fill_flag == 0, cloud_fill_flag, new_fill_flag)
    else:
        return new_fill_flag


def gap_fill_irrad(irrad, cs_irrad, fill_flag, return_csr=False,
                   flags_to_fill=None):
    """Fill bad irradiance data using clearsky and nearest good cloudy data.

    Parameters
    ----------
    irrad : np.ndarray
        Full FARMS + REST2 merged irradiance 2D array.
    cs_irrad : np.ndarray
        REST2 clearsky irradiance without bad or missing data.
    fill_flag : np.ndarray
        2D array with nonzero values where bad data must be filled.
    return_csr : bool
        If set to true, this function will return the clearsky ratio array.
    flags_to_fill : None | list
        List of fill_flag values to fill. None defaults to all nonzero flags.

    Returns
    -------
    irrad : np.ndarray
        Updated and patched irradiance numpy array.
    csr : np.ndarray
        Irradiance cloudy/clearsky ratio numpy array.
    """
    # disable divide by zero warnings
    np.seterr(divide='ignore', invalid='ignore')

    # Default flags to fill to all nonzero values
    if flags_to_fill is None:
        flags_to_fill = np.unique(fill_flag)
        flags_to_fill = list(flags_to_fill).remove(0)

    # convert fill flag to boolean
    fill_flag = np.isin(fill_flag, flags_to_fill)

    # get the merged div by clearsky ratio dataframe
    csr = irrad / cs_irrad

    # replace to-fill values with nan
    csr[fill_flag] = np.nan

    # assign sites csr=1 with all NaN or just one non-NaN but warn
    all_na = (np.isnan(csr).sum(axis=0) >= (csr.shape[0] - 1))
    if any(all_na):
        warn('{} sites exist with full NaN csr timeseries.'
             .format(np.sum(all_na)))
        csr[:, all_na] = 1.0

    # fill nan ratio values with nearest good ratio values
    csr = pd.DataFrame(csr).interpolate(method='nearest', axis=0)\
        .fillna(method='ffill', axis=0)\
        .fillna(method='bfill', axis=0).values

    # Set the cloud/clear ratio when it's nighttime
    csr[(cs_irrad == 0)] = 0

    # fill values with (closest good ratio values X
    # good clearsky data at the bad data index)
    irrad[fill_flag] = csr[fill_flag] * cs_irrad[fill_flag]

    if return_csr:
        return irrad, csr
    else:
        return irrad


def enforce_clearsky(dni, ghi, cs_dni, cs_ghi, sza, fill_flag,
                     sza_lim=SZA_LIM):
    """Enforce a maximum irradiance equal to the clearsky irradiance.

    Parameters
    ----------
    dni : np.ndarray
        All-sky (cloudy + clear) DNI.
    ghi : np.ndarray
        All-sky (cloudy + clear) GHI.
    cs_dni : np.ndarray
        Clearsky (REST) DNI.
    cs_ghi : np.ndarray
        Clearsky (REST) GHI.
    sza : np.ndarray
        Solar zenith angle in degrees.
    fill_flag : None
        Integer array of flags showing what data was previously filled and why.
    sza_lim : int | float
        Upper limit of SZA in degrees.

    Returns
    -------
    dni : np.ndarray
        All-sky (cloudy + clear) DNI with max of clearsky values.
    ghi : np.ndarray
        All-sky (cloudy + clear) GHI with max of clearsky values.
    fill_flag : np.ndarray
        Array of integers signifying whether to fill the irradiance
        data and the reason why.
    """

    mask = ((dni > cs_dni) | (ghi > cs_ghi)) & (sza < sza_lim)
    dni[mask] = cs_dni[mask]
    ghi[mask] = cs_ghi[mask]
    fill_flag[mask] = 5

    return dni, ghi, fill_flag
