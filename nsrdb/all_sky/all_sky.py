# -*- coding: utf-8 -*-
"""NSRDB all-sky module.
"""

import numpy as np
from nsrdb.all_sky.disc import disc
from nsrdb.all_sky.rest2 import rest2, rest2_tuuclr
from nsrdb.all_sky.farms import farms
from nsrdb.all_sky.gap_fill import make_fill_flag, gap_fill_irrad
from nsrdb.all_sky import CLOUD_TYPES, SZA_LIM
from nsrdb.all_sky.utilities import (ti_to_radius, calc_beta, merge_rest_farms,
                                     calc_dhi, screen_sza, screen_cld,
                                     dark_night)


def all_sky(albedo, alpha, aod, asym, ozone, p, ssa, sza, ti, w,
            cloud_type, cld_opd_dcomp, cld_reff_dcomp, debug=False):
    """Calculate the all-sky irradiance.

    Parameters
    ----------
    albedo : np.ndarray
        Ground albedo.
    alpha : np.ndarray
        Angstrom wavelength exponent, ideally obtained by linear
        regression of all available spectral AODs between 380 and 1020 nm.
        Input value will be tested for compliance with the mandatory
        interval [0, 2.5], and corrected if necessary.
    aod : np.ndarray
        Array of aerosol optical depth (AOD) values.
    asym : np.ndarray
        Aerosol asymmetry parameter. Since it tends to vary with wavelength
        and alpha, use a representative value for a wavelength of about
        700 nm and alpha about 1. Will default to 0.7 if a NEGATIVE value is
        input (when exact value is unknown).
    ozone : np.ndarray
        reduced ozone vertical pathlength (atm-cm)
        [Note: 1 atm-cm = 1000 DU]
    p : np.ndarray
        Surface pressure (mbar).
    ssa : np.ndarray
        aerosol single-scattering albedo at a representative wavelength of
        about 700 nm. Use -9.99 or any other NEGATIVE value if unknown
        (will default to 0.92 if a negative value is input)
    sza : np.ndarray
        Solar zenith angle (degrees). Must represent the average value over the
        integration period (e.g. hourly) under scrutiny.
    ti : pd.DatetimeIndex
        Time index.
    w : np.ndarray
        Total precip. water (cm).
    cloud_type : np.ndarray
        Array of integer cloud types.
    cloud_opd_dcomp : np.ndarray
        Array of cloud optical depths. Expected range is 0 - 160 with
        missing values < 0.
    cld_reff_dcomp : np.ndarray
        Array of cloud effective partical radii. Expected range is 0 - 160
        with missing values < 0.
    debug : bool
        Flag to return extra variables.

    Returns
    -------
    dhi : np.ndarray
        Diffuse horizontal irradiance.
    dni
        Direct normal irradiance.
    ghi
        Global horizontal irradiance.
    """

    # calculate derived variables
    radius = ti_to_radius(ti, n_cols=p.shape[1])
    beta = calc_beta(aod, alpha)

    # make boolean flag for where there were missing cloud properties
    # need to do this before the cloud property ranges are screened
    missing_cld_props = ((np.isin(cloud_type, CLOUD_TYPES)) &
                         ((cld_opd_dcomp < 0) | (cld_reff_dcomp < 0) |
                          (np.isnan(cld_opd_dcomp) |
                          (np.isnan(cld_reff_dcomp)))))

    # screen variables based on expected ranges
    sza = screen_sza(sza, lim=SZA_LIM)
    cld_opd_dcomp = screen_cld(cld_opd_dcomp)
    cld_reff_dcomp = screen_cld(cld_reff_dcomp)

    # run REST2 to get clearsky data
    rest_data = rest2(p, albedo, ssa, asym, sza, radius, alpha, beta, ozone, w)
    Tuuclr = rest2_tuuclr(p, albedo, ssa, radius, alpha, ozone, w,
                          parallel=False)

    # Ensure that clearsky irradiance is zero when sun is below horizon
    rest_data.dhi = dark_night(rest_data.dhi, sza, lim=SZA_LIM)
    rest_data.dni = dark_night(rest_data.dni, sza, lim=SZA_LIM)
    rest_data.ghi = dark_night(rest_data.ghi, sza, lim=SZA_LIM)

    # use FARMS to calculate cloudy GHI
    ghi = farms(tau=cld_opd_dcomp,
                cloud_type=cloud_type,
                cloud_effective_radius=cld_reff_dcomp,
                solar_zenith_angle=sza,
                radius=radius,
                Tuuclr=Tuuclr,
                Ruuclr=rest_data.Ruuclr,
                Tddclr=rest_data.Tddclr,
                Tduclr=rest_data.Tduclr,
                albedo=albedo,
                )

    # merge the clearsky and cloudy irradiance into all-sky irradiance
    ghi = merge_rest_farms(rest_data.ghi, ghi, cloud_type)

    # calculate the DNI using the DISC model
    dni = disc(ghi, sza, doy=ti.dayofyear.values,
               pressure=p, sza_lim=SZA_LIM)

    # merge the clearsky and cloudy irradiance into all-sky irradiance
    dni = merge_rest_farms(rest_data.dni, dni, cloud_type)

    # make a fill flag where bad data exists in the GHI irradiance
    fill_flag = make_fill_flag(ghi, rest_data.ghi, cloud_type,
                               missing_cld_props)

    # Gap fill bad data in ghi and dni using the fill flag and cloud type
    ghi, csr = gap_fill_irrad(ghi, rest_data.ghi, fill_flag, return_csr=True)
    dni = gap_fill_irrad(dni, rest_data.dni, fill_flag)

    # calculate the DHI, patching DNI for negative DHI values
    dni, dhi = calc_dhi(dni, ghi, sza)

    # ensure that final irradiance is zero when sun is below horizon.
    dhi = dark_night(dhi, sza, lim=SZA_LIM)
    dni = dark_night(dni, sza, lim=SZA_LIM)
    ghi = dark_night(ghi, sza, lim=SZA_LIM)

    if debug:
        # return extra debugging variables.
        return (dhi, dni, ghi, rest_data.dhi, rest_data.dni, rest_data.ghi,
                fill_flag, csr)
    else:
        # return all-sky irradiance (no debugging variables)
        return dhi, dni, ghi
