"""Common utilities for NSRDB all-sky module.
"""

import pandas as pd
import numpy as np
from warnings import warn
from nsrdb.all_sky import RADIUS, CLEAR_TYPES, SZA_LIM


def check_range(data, name, rang=(0, 1)):
    """Ensure that data values are in correct range."""
    if np.nanmin(data) < rang[0] or np.nanmax(data) > rang[1]:
        raise ValueError('Variable "{n}" is out of expected '
                         'transmittance/reflectance range. Recommend checking '
                         'solar zenith angle to ensure cos(sza) is '
                         'non-negative and non-zero. '
                         'Max/min of {n} = {mx}/{mn}'
                         .format(n=name,
                                 mx=np.nanmax(data),
                                 mn=np.nanmin(data)))


def ti_to_radius(time_index, n_cols=1):
    """Convert a time index to radius.

    Parameters
    ----------
    time_index : pandas.core.indexes.datetimes.DatetimeIndex
        NSRDB time series. Can extract from h5 as follows:
        time_index = pd.to_datetime(h5['time_index'][...].astype(str))
    n_cols : int
        Number of columns to output. The radius vertical 1D array will be
        copied this number of times horizontally (np.tile).

    Returns
    -------
    radius : np.array
        Array of radius values matching the time index.
        Shape is (len(time_index), n_cols).
    """
    doy = pd.DataFrame(index=time_index.dayofyear)
    radius = doy.join(RADIUS)
    radius = np.tile(radius.values, n_cols)
    return radius


def calc_beta(aod, alpha):
    """Calculate the Angstrom turbidity coeff. (beta).

    Parameters
    ----------
    aod : np.ndarray
        Array of aerosol optical depth (AOD) values. Shape must match alpha.
    alpha : np.ndarray
        Array of angstrom wavelength exponent values. Shape must match aod.

    Returns
    -------
    beta : np.ndarray
        Array of Angstrom turbidity coeff., i.e. AOD at 1000 nm.
        Shape will be same as aod and alpha. Will be tested for compliance
        with the mandatory interval [0, 2.2].
    """
    if aod.shape != alpha.shape:
        raise ValueError('To calculate beta, aod and alpha inputs must be of '
                         'the same shape. Received arrays of shape {} and {}'
                         .format(aod.shape, alpha.shape))
    beta = aod * np.power(0.55, alpha)
    if np.max(beta) > 2.2 or np.min(beta) < 0:
        warn('Calculation of beta resulted in values outside of '
             'expected range [0, 2.2]. Min/max of beta are: {}/{}'
             .format(np.min(beta), np.max(beta)))
    return beta


def merge_rest_farms(clearsky_irrad, cloudy_irrad, cloud_type):
    """Combine clearsky and rest data into all-sky irradiance array.

    This also ensures that the cloudy irradiance is always less
    than the clearsky irradiance.

    Parameters
    ----------
    clearsky_irrad : np.ndarray
        Clearsky irradiance data from REST.
    cloudy_irrad : np.ndarray
        Cloudy irradiance data from FARMS.
    cloud_type : np.ndarray
        Cloud type array which acts as a mask specifying where to take
        cloud/clear data.

    Returns
    -------
    all_sky_irrad : np.ndarray
        All-sky (cloudy + clear) irradiance data, merged dataset from
        FARMS and REST.
    """
    # disable nan warnings
    np.seterr(divide='ignore', invalid='ignore')

    # Don't let cloudy irradiance be greater than the clearsky irradiance.
    cloudy_irrad = np.where(cloudy_irrad > clearsky_irrad,
                            clearsky_irrad, cloudy_irrad)

    # combine clearsky and farms according to the cloud types.
    all_sky_irrad = np.where(np.isin(cloud_type, CLEAR_TYPES),
                             clearsky_irrad, cloudy_irrad)

    return all_sky_irrad


def calc_dhi(dni, ghi, sza):
    """Calculate the diffuse horizontal irradiance and correct the direct.

    Parameters
    ----------
    dni : np.ndarray
        Direct normal irradiance.
    ghi : np.ndarray
        Global horizontal irradiance.
    sza : np.ndarray
        Solar zenith angle (degrees).

    Returns
    -------
    dni : np.ndarray
        Direct normal irradiance. This is set to zero where dhi < 0
    dhi : np.ndarray
        Diffuse horizontal irradiance. This is ensured to be non-negative.
    """

    dhi = ghi - dni * np.cos(np.radians(sza))
    if np.min(dhi) < 0:
        # patch for negative DHI values. Set DNI to zero, set DHI to GHI
        pos = np.where(dhi < 0)
        dni[pos] = 0
        dhi[pos] = ghi[pos]

    return dni, dhi


def screen_cld(cld_data, rng=(0, 160)):
    """Enforce a numeric range on cloud property data.

    Parameters
    ----------
    cld_data : np.ndarray
        Cloud property data (cld_opd_dcomp, cld_reff_dcomp).
    rng : list | tuple
        Inclusive intended range of the cloud data.

    Parameters
    ----------
    cld_data : np.ndarray
        Cloud property data (cld_opd_dcomp, cld_reff_dcomp)
        with min/max values equal to rng.
    """

    cld_data[(cld_data < rng[0])] = rng[0]
    cld_data[(cld_data > rng[1])] = rng[1]
    return cld_data


def screen_sza(sza, lim=SZA_LIM):
    """Enforce an upper limit on the solar zenith angle.

    Parameters
    ----------
    sza : np.ndarray
        Solar zenith angle in degrees.
    lim : int | float
        Upper limit of SZA in degrees.

    Returns
    ----------
    sza : np.ndarray
        Solar zenith angle in degrees with max value = lim.
    """
    sza[(sza > lim)] = lim
    return sza


def dark_night(irrad_data, sza, lim=SZA_LIM):
    """Enforce zero irradiance when solar angle >= threshold.

    Parameters
    ----------
    irrad_data : np.ndarray
        DHI, DNI, or GHI.
    sza : np.ndarray
        Solar zenith angle in degrees.
    lim : int | float
        Upper limit of SZA in degrees.

    Returns
    -------
    irrad_data : np.ndarray
        DHI, DNI, or GHI with zero values when sza >= lim.
    """
    night_mask = np.where(sza >= lim)
    irrad_data[night_mask] = 0
    return irrad_data
