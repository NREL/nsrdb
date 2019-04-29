# -*- coding: utf-8 -*-
"""NSRDB all-sky module.
"""

from concurrent.futures import ProcessPoolExecutor
import os
import numpy as np
import pandas as pd
import psutil
import time
import logging
from warnings import warn

from nsrdb.file_handlers.resource import Resource
from nsrdb.all_sky.disc import disc
from nsrdb.all_sky.rest2 import rest2, rest2_tuuclr
from nsrdb.all_sky.farms import farms
from nsrdb.all_sky import SZA_LIM
from nsrdb.all_sky.utilities import (ti_to_radius, calc_beta, merge_rest_farms,
                                     calc_dhi, screen_sza, screen_cld,
                                     dark_night, cloud_variability)
from nsrdb.gap_fill.irradiance_fill import (make_fill_flag, gap_fill_irrad,
                                            missing_cld_props)


logger = logging.getLogger(__name__)


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
        missing values <= 0.
    cld_reff_dcomp : np.ndarray
        Array of cloud effective partical radii. Expected range is 0 - 160
        with missing values <= 0.
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
    missing_props = missing_cld_props(cloud_type, cld_opd_dcomp,
                                      cld_reff_dcomp)

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
    fill_flag = make_fill_flag(ghi, rest_data.ghi, cloud_type, missing_props)

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


def all_sky_h5(f_ancillary, f_cloud, rows=slice(None), cols=slice(None)):
    """Run all-sky from .h5 files.

    Parameters
    ----------
    f_ancillary : str
        File path to ancillary data file.
    f_cloud : str
        File path the cloud data file.
    rows : slice
        Subset of rows to run.
    cols : slice
        Subset of columns to run.

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

    logger.info('Running all-sky from the following files:\n\t{}\n\t{}'
                .format(f_ancillary, f_cloud))
    logger.info('Running only for rows: "{}" and columns: "{}"'
                .format(rows, cols))

    with Resource(f_ancillary) as fa:
        with Resource(f_cloud) as fc:
            out = all_sky(
                alpha=fa['alpha', rows, cols],
                aod=fa['aod', rows, cols],
                asymmetry=fa['asymmetry', rows, cols],
                cloud_type=fc['cloud_type', rows, cols],
                cld_opd_dcomp=fc['cld_opd_dcomp', rows, cols],
                cld_reff_dcomp=fc['cld_reff_dcomp', rows, cols],
                ozone=fa['ozone', rows, cols],
                solar_zenith_angle=fc['solar_zenith_angle', rows, cols],
                ssa=fa['ssa', rows, cols],
                surface_albedo=fa['surface_albedo', rows, cols],
                surface_pressure=fa['surface_pressure', rows, cols],
                time_index=fc.time_index[rows],
                total_precipitable_water=fa['total_precipitable_water',
                                            rows, cols])
    return out


def all_sky_h5_parallel(f_ancillary, f_cloud, rows=slice(None),
                        cols=slice(None)):
    """Run all-sky from .h5 files.

    Parameters
    ----------
    f_ancillary : str
        File path to ancillary data file.
    f_cloud : str
        File path the cloud data file.
    rows : slice
        Subset of rows to run.
    cols : slice
        Subset of columns to run.

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

    if rows.start is None:
        rows = slice(0, rows.stop)
    if rows.stop is None:
        with Resource(f_cloud) as res:
            rows = slice(rows.start, res.shape[0])

    if cols.start is None:
        cols = slice(0, cols.stop)
    if cols.stop is None:
        with Resource(f_cloud) as res:
            cols = slice(cols.start, res.shape[1])

    out_shape = (rows.stop - rows.start, cols.stop - cols.start)
    c_range = range(cols.start, cols.stop)

    # start a local cluster
    max_workers = int(os.cpu_count())
    futures = {}
    logger.info('Running all-sky in parallel on {} workers.'
                .format(max_workers))
    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        # submit a future for each NSRDB site
        for c in c_range:
            futures[c] = exe.submit(
                all_sky_h5, f_ancillary, f_cloud,
                rows=rows, cols=slice(c, c + 1))

        # watch memory during futures to get max memory usage
        logger.debug('Waiting on parallel futures...')
        max_mem = 0
        running = len(futures)
        while running > 0:
            mem = psutil.virtual_memory()
            max_mem = np.max((mem.used / 1e9, max_mem))
            time.sleep(5)
            running = 0
            keys = []
            for key, future in futures.items():
                if future.running():
                    running += 1
                    keys += [key]
            logger.debug('{} sites are being processed by all-sky futures.'
                         .format(running))

        logger.info('Futures finished, maximum memory usage was '
                    '{0:.3f} GB out of {1:.3f} GB total.'
                    .format(max_mem, mem.total / 1e9))

        # gather results
        for k, v in futures.items():
            futures[k] = v.result()

    out = {}
    for var, arr in futures[cols.start].items():
        out[var] = np.ndarray(out_shape, dtype=arr.dtype)

    for c, as_out_dict in futures.items():
        for var, arr in as_out_dict.items():
            out[var][:, c] = arr

    return out
