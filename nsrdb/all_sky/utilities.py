"""Common utilities for NSRDB all-sky module.
"""

import pandas as pd
import numpy as np
import os
from warnings import warn
from nsrdb.all_sky import RADIUS, CLEAR_TYPES, CLOUD_TYPES, SZA_LIM


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


def ti_to_radius_csv(time_index, n_cols=1):
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


def ti_to_radius(time_index, n_cols=1):
    """Calculates Earth-Sun Radius Vector.

    Reference:
    http://www.nrel.gov/docs/fy08osti/34302.pdf

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

    # load earth periodic table
    path = os.path.dirname(os.path.realpath(__file__))
    df = pd.read_csv(os.path.join(path, 'earth_periodic_terms.csv'))
    df['key'] = 1
    # 3.1.1 (4). Julian Date.
    j = time_index.to_julian_date().values
    # 3.1.2 (5). Julian Ephermeris Date
    j = j + 64.797 / 86400
    # 3.1.3 (7). Julian Century Ephemeris
    j = (j - 2451545) / 36525
    # 3.1.4 (8). Julian Ephemeris Millennium
    j = j / 10
    df_jme = pd.DataFrame({'uid': range(len(j)), 'jme': j, 'key': 1})
    # Merge JME with Periodic Table
    df_merge = pd.merge(df_jme, df, on='key')
    # 3.2.1 (9). Heliocentric radius vector.
    df_merge['r'] = df_merge['a'] * np.cos(df_merge['b'] + df_merge['c'] *
                                           df_merge['jme'])
    # 3.2.2 (10).
    dfs = df_merge.groupby(by=['uid', 'term'])['r'].sum().unstack()
    # 3.2.4 (11). Earth Heliocentric radius vector
    radius = ((dfs['R0'] + dfs['R1'] * j + dfs['R2'] * np.power(j, 2) +
               dfs['R3'] * np.power(j, 3) + dfs['R4'] * np.power(j, 4) +
               dfs['R5'] * np.power(j, 5)) / np.power(10, 8)).values
    radius = radius.reshape((len(time_index), 1))
    radius = np.tile(radius, n_cols)
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


def rayleigh_check(dhi, dni, ghi, cs_dhi, cs_dni, cs_ghi, fill_flag,
                   rayleigh_flag=7):
    """Perform the rayleigh violation check.

    This check ensures that all-sky diffuse irradiance >= clearsky diffuse.
    If the condition is not met, all irradiances are set to clearsky values.

    Parameters
    ----------
    dhi : np.ndarray
        All-sky diffuse irradiance.
    dni : np.ndarray
        All-sky direct normal irradiance.
    ghi : np.ndarray
        All-sky global horizontal irradiance.
    cs_dhi : np.ndarray
        Clearsky (rest) diffuse irradiance.
    cs_dni : np.ndarray
        Clearsky (rest) direct normal irradiance.
    cs_ghi : np.ndarray
        Clearsky (rest) global horizontal irradiance.
    fill_flag : np.ndarray
        Array of integers signifying whether irradiance has been filled.
    rayleigh_flag : int
        Fill flag for rayleigh violation.

    Returns
    -------
    dhi : np.ndarray
        All-sky diffuse irradiance. Rayleigh violations are set to cs_dhi.
    dni : np.ndarray
        All-sky direct normal irradiance. Rayleigh violations are set to
        cs_dni.
    ghi : np.ndarray
        All-sky global horizontal irradiance. Rayleigh violations are set to
        cs_ghi.
    fill_flag : np.ndarray
        Array of integers signifying whether irradiance has been filled, with
        rayleigh violations marked with the rayleigh flag.
    """

    # boolean mask where the rayleigh check fails
    failed = (dhi < cs_dhi)

    if failed.any():
        # set irradiance values to clearsky, set fill flag.
        dhi[failed] = cs_dhi[failed]
        dni[failed] = cs_dni[failed]
        ghi[failed] = cs_ghi[failed]
        fill_flag[failed] = rayleigh_flag

    return dhi, dni, ghi, fill_flag


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

    cld_data[np.isnan(cld_data)] = 0
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


def cloud_variability(irrad, cs_irrad, cloud_type, var_frac=0.05,
                      option='tri', random_seed=123):
    """Add syntehtic variability to irradiance when it's cloudy.

    Parameters
    ----------
    irrad : np.ndarray
        Full FARMS + REST2 merged irradiance 2D array.
    cs_irrad : np.ndarray
        REST2 clearsky irradiance without bad or missing data.
    cloud_type : np.ndarray
        Array of numerical cloud types.
    var_frac : float
        Maximum variability fraction.
    option : str
        Variability function option ('tri' or 'linear').
    random_seed : int | NoneType
        Number to seed the numpy random number generator. Used to generate
        reproducable psuedo-random cloud variability. Numpy random will be
        seeded with the system time if this is None.

    Returns
    -------
    irrad : np.ndarray
        Full FARMS + REST2 merged irradiance 2D array with variability added
        to cloudy timesteps.
    """

    # disable divide by zero warnings
    np.seterr(divide='ignore', invalid='ignore')

    if var_frac:
        # set a seed for psuedo-random but repeatable results
        np.random.seed(seed=random_seed)

        # update the clearsky ratio (1 is clear, 0 is cloudy or dark)
        csr = irrad / cs_irrad
        # Set the cloud/clear ratio to zero when it's nighttime
        csr[(cs_irrad == 0)] = 0

        if option == 'linear':
            var_frac_arr = linear_variability(csr, var_frac)
        elif option == 'tri':
            var_frac_arr = tri_variability(csr, var_frac)

        # get a uniform random scalar array 0 to 1 with data shape
        rand_arr = np.random.rand(irrad.shape[0], irrad.shape[1])
        # Center the random array at 1 +/- var_perc_arr (with csr scaling)
        rand_arr = 1 + var_frac_arr * (rand_arr * 2 - 1)

        # only apply rand to the applicable cloudy timesteps
        rand_arr = np.where(np.isin(cloud_type, CLOUD_TYPES), rand_arr, 1)
        irrad *= rand_arr

    return irrad


def linear_variability(csr, var_frac):
    """Return an array with a linear relation between clearsky ratio and
    maximum variability fraction.

    Parameters
    ----------
    csr : np.ndarray
        REST2 clearsky irradiance without bad or missing data.
    var_frac : float
        Maximum variability fraction.

    Returns
    -------
    out : np.ndarray
        Array with shape matching csr with maximum variability (var_frac)
        when the csr = 1 (clear or thin clouds).
    """

    return var_frac * csr


def tri_variability(csr, var_frac, center=0.9):
    """Return an array with a triangular distribution between clearsky ratio
    and maximum variability fraction.

    The max variability occurs when csr==center, and zero variability when
    csr==0 or csr==1

    Parameters
    ----------
    csr : np.ndarray
        REST2 clearsky irradiance without bad or missing data.
    var_frac : float
        Maximum variability fraction.
    center : float
        Value of the clearsky ratio at which there is maximum variability.

    Returns
    -------
    tri : np.ndarray
        Array with shape matching csr with maximum variability (var_frac)
        when the csr==center.
    """

    tri_left = var_frac * csr * 1.11111
    slope = -1 / (1 - center)
    yint = center * 10 + 1
    tri_right = var_frac * (slope * csr + yint)
    tri = np.where(csr < center, tri_left, tri_right)
    return tri
