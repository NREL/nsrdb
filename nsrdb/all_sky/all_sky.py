# -*- coding: utf-8 -*-
"""NSRDB all-sky module.
"""

import numpy as np
import pandas as pd
from warnings import warn
from nsrdb.all_sky.disc import disc
from nsrdb.all_sky.rest2 import rest2, rest2_tuuclr
from nsrdb.all_sky.farms import farms
from nsrdb.all_sky.gap_fill import make_fill_flag, gap_fill_irrad
from nsrdb.all_sky import CLOUD_TYPES, SZA_LIM
from nsrdb.all_sky.utilities import (ti_to_radius, calc_beta, merge_rest_farms,
                                     calc_dhi, screen_sza, screen_cld,
                                     dark_night, cloud_variability)


def all_sky(alpha, aod, asymmetry, cloud_type, cld_opd_dcomp, cld_reff_dcomp,
            ozone, solar_zenith_angle, ssa, surface_albedo, surface_pressure,
            time_index, total_precipitable_water, ghi_variability=None):
    """Calculate the all-sky irradiance.

    Parameters
    ----------
    alpha : np.ndarray
        Angstrom wavelength exponent, ideally obtained by linear
        regression of all available spectral AODs between 380 and 1020 nm.
        Input value will be tested for compliance with the mandatory
        interval [0, 2.5], and corrected if necessary.
    aod : np.ndarray
        Array of aerosol optical depth (AOD) values.
    asymmetry : np.ndarray
        Aerosol asymmetry parameter. Since it tends to vary with wavelength
        and alpha, use a representative value for a wavelength of about
        700 nm and alpha about 1. Will default to 0.7 if a NEGATIVE value is
        input (when exact value is unknown).
    cloud_type : np.ndarray
        Array of integer cloud types.
    cloud_opd_dcomp : np.ndarray
        Array of cloud optical depths. Expected range is 0 - 160 with
        missing values < 0.
    cld_reff_dcomp : np.ndarray
        Array of cloud effective partical radii. Expected range is 0 - 160
        with missing values < 0.
    ozone : np.ndarray
        reduced ozone vertical pathlength (atm-cm)
        [Note: 1 atm-cm = 1000 DU]
    solar_zenith_angle : np.ndarray
        Solar zenith angle (degrees). Must represent the average value over the
        integration period (e.g. hourly) under scrutiny.
    ssa : np.ndarray
        aerosol single-scattering albedo at a representative wavelength of
        about 700 nm. Use -9.99 or any other NEGATIVE value if unknown
        (will default to 0.92 if a negative value is input)
    surface_albedo : np.ndarray
        Ground albedo.
    surface_pressure : np.ndarray
        Surface pressure (mbar).
    time_index : pd.DatetimeIndex
        Time index.
    total_precipitable_water : np.ndarray
        Total precip. water (cm).
    ghi_variability : NoneType | float
        Variability fraction to apply to GHI. Provides synthetic variability
        for cloudy irradiance when downscaling data.

    Returns
    -------
    output : dict
        Namespace of all-sky irradiance output variables with the
        following keys:
            'clearsky_dhi'
            'clearsky_dni'
            'clearsky_ghi'
            'dhi'
            'dni'
            'ghi'
            'fill_flag'
    """

    if isinstance(time_index, np.ndarray):
        time_index = pd.to_datetime(time_index.astype(str))

    # calculate derived variables
    radius = ti_to_radius(time_index, n_cols=alpha.shape[1])
    beta = calc_beta(aod, alpha)

    # make boolean flag for where there were missing cloud properties
    # need to do this before the cloud property ranges are screened
    missing_cld_props = ((np.isin(cloud_type, CLOUD_TYPES)) &
                         ((cld_opd_dcomp < 0) | (cld_reff_dcomp < 0) |
                          (np.isnan(cld_opd_dcomp) |
                          (np.isnan(cld_reff_dcomp)))))

    # screen variables based on expected ranges
    solar_zenith_angle = screen_sza(solar_zenith_angle, lim=SZA_LIM)
    cld_opd_dcomp = screen_cld(cld_opd_dcomp)
    cld_reff_dcomp = screen_cld(cld_reff_dcomp)

    # run REST2 to get clearsky data
    rest_data = rest2(surface_pressure, surface_albedo, ssa, asymmetry,
                      solar_zenith_angle, radius, alpha, beta, ozone,
                      total_precipitable_water)
    Tuuclr = rest2_tuuclr(surface_pressure, surface_albedo, ssa, radius, alpha,
                          ozone, total_precipitable_water, parallel=False)

    # Ensure that clearsky irradiance is zero when sun is below horizon
    rest_data.dhi = dark_night(rest_data.dhi, solar_zenith_angle, lim=SZA_LIM)
    rest_data.dni = dark_night(rest_data.dni, solar_zenith_angle, lim=SZA_LIM)
    rest_data.ghi = dark_night(rest_data.ghi, solar_zenith_angle, lim=SZA_LIM)

    # use FARMS to calculate cloudy GHI
    ghi = farms(tau=cld_opd_dcomp,
                cloud_type=cloud_type,
                cloud_effective_radius=cld_reff_dcomp,
                solar_zenith_angle=solar_zenith_angle,
                radius=radius,
                Tuuclr=Tuuclr,
                Ruuclr=rest_data.Ruuclr,
                Tddclr=rest_data.Tddclr,
                Tduclr=rest_data.Tduclr,
                albedo=surface_albedo,
                )

    # merge the clearsky and cloudy irradiance into all-sky irradiance
    ghi = merge_rest_farms(rest_data.ghi, ghi, cloud_type)

    # option to add synthetic variability to cloudy ghi
    if ghi_variability:
        ghi = cloud_variability(ghi, rest_data.ghi, cloud_type,
                                var_frac=ghi_variability, option='tri')

    # calculate the DNI using the DISC model
    dni = disc(ghi, solar_zenith_angle, doy=time_index.dayofyear.values,
               pressure=surface_pressure, sza_lim=SZA_LIM)

    # merge the clearsky and cloudy irradiance into all-sky irradiance
    dni = merge_rest_farms(rest_data.dni, dni, cloud_type)

    # make a fill flag where bad data exists in the GHI irradiance
    fill_flag = make_fill_flag(ghi, rest_data.ghi, cloud_type,
                               missing_cld_props)

    # Gap fill bad data in ghi and dni using the fill flag and cloud type
    ghi = gap_fill_irrad(ghi, rest_data.ghi, fill_flag, return_csr=False)
    dni = gap_fill_irrad(dni, rest_data.dni, fill_flag)

    # calculate the DHI, patching DNI for negative DHI values
    dni, dhi = calc_dhi(dni, ghi, solar_zenith_angle)

    # ensure that final irradiance is zero when sun is below horizon.
    dhi = dark_night(dhi, solar_zenith_angle, lim=SZA_LIM)
    dni = dark_night(dni, solar_zenith_angle, lim=SZA_LIM)
    ghi = dark_night(ghi, solar_zenith_angle, lim=SZA_LIM)

    # check for NaN and negative irradiance values, raise warning
    for name, var in [['dhi', dhi], ['dni', dni], ['ghi', ghi]]:
        if np.sum(np.isnan(var)):
            warn('NaN values are present in "{}" after all-sky irradiance '
                 'calculation.'.format(name))
        if np.min(var) < 0:
            warn('Negative values are present in "{}" after all-sky '
                 'irradiance calculation.'.format(name))

    output = {'clearsky_dhi': rest_data.dhi,
              'clearsky_dni': rest_data.dni,
              'clearsky_ghi': rest_data.ghi,
              'dhi': dhi,
              'dni': dni,
              'ghi': ghi,
              'fill_flag': fill_flag}

    return output
