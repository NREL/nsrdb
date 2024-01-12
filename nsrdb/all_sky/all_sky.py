# -*- coding: utf-8 -*-
"""NSRDB all-sky module.
"""
import logging
import os
from concurrent.futures import as_completed
from warnings import warn

import numpy as np
import pandas as pd
import psutil
from farms import SZA_LIM
from farms.disc import disc
from farms.farms import farms
from farms.utilities import (
    calc_beta,
    calc_dhi,
    cloud_variability,
    dark_night,
    merge_rest_farms,
    screen_cld,
    screen_sza,
    ti_to_radius,
)
from rest2.rest2 import rest2, rest2_tuuclr
from rex import MultiFileResource, Resource
from rex.utilities import SpawnProcessPool

from nsrdb.all_sky.utilities import scale_all_sky_outputs
from nsrdb.gap_fill.irradiance_fill import (
    enforce_clearsky,
    gap_fill_irrad,
    make_fill_flag,
    missing_cld_props,
)

logger = logging.getLogger(__name__)


# Spatiotemporal all sky variable input names. Does not include time_index
# which is not Spatiotemporal.
ALL_SKY_ARGS = ('alpha',
                'aod',
                'asymmetry',
                'cloud_type',
                'cld_opd_dcomp',
                'cld_reff_dcomp',
                'ozone',
                'solar_zenith_angle',
                'ssa',
                'surface_albedo',
                'surface_pressure',
                'total_precipitable_water',
                'cloud_fill_flag')


def all_sky(alpha, aod, asymmetry, cloud_type, cld_opd_dcomp, cld_reff_dcomp,
            ozone, solar_zenith_angle, ssa, surface_albedo, surface_pressure,
            time_index, total_precipitable_water, cloud_fill_flag=None,
            variability_kwargs=None, scale_outputs=True, disc_on=False):
    """Calculate the all-sky irradiance.
    Updated by Yu Xie on 3/29/2023 to compute DNI by FARMS-DNI.

    Variables
    ---------
    dni_farmsdni: np.ndarray
        DNI computed by FARMS-DNI (Wm-2).
    dni0: np.ndarray
        DNI computed by the Lambert law (Wm-2). It only includes the narrow
         beam in the circumsolar region.
    Updated by Yu Xie on 3/29/2023 to compute DNI by FARMS-DNI.

    Variables
    ---------
    dni_farmsdni: np.ndarray
        DNI computed by FARMS-DNI (Wm-2).

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
        Surface pressure (mbar).(mbar is same as hPa)
    time_index : pd.DatetimeIndex
        Time index.
    total_precipitable_water : np.ndarray
        Total precip. water (cm).
    cloud_fill_flag : None | np.ndarray
        Integer array of flags showing what data was previously filled and why.
        None will create a new fill flag initialized as all zeros.
        An array input will be interpreted as flags showing which cloud
        properties have already been filled.
    variability_kwargs : NoneType | dict
        Variability key word args to apply to GHI. Provides synthetic
        variability for cloudy irradiance when downscaling data. See
        nsrdb.all_sky.utilities.cloud_variability for kwarg definitions.
    scale_outputs : bool
        Flag to safely scale and dtype convert output arrays.
    disc_on : bool
        Compute cloudy-sky DNI using FARMS-DNI (False) or DISC (True).

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

    # do not all-sky irradiance gap fill previously filled cloud props
    flags_to_fill = list(range(1, 100))
    if cloud_fill_flag is not None:
        already_filled = np.unique(cloud_fill_flag)
        flags_to_fill = list(set(flags_to_fill) - set(already_filled))

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
    ghi, dni_farmsdni, _ = farms(tau=cld_opd_dcomp,
                                 cloud_type=cloud_type,
                                 cloud_effective_radius=cld_reff_dcomp,
                                 solar_zenith_angle=solar_zenith_angle,
                                 radius=radius,
                                 Tuuclr=Tuuclr,
                                 Ruuclr=rest_data.Ruuclr,
                                 Tddclr=rest_data.Tddclr,
                                 Tduclr=rest_data.Tduclr,
                                 albedo=surface_albedo)

    # merge the clearsky and cloudy irradiance into all-sky irradiance
    ghi = merge_rest_farms(rest_data.ghi, ghi, cloud_type)

    # option to add synthetic variability to cloudy ghi
    if variability_kwargs is not None:
        ghi = cloud_variability(ghi, rest_data.ghi, cloud_type,
                                **variability_kwargs)

    # merge the clearsky and cloudy irradiance into all-sky irradiance
    if disc_on:
        dni = disc(ghi, solar_zenith_angle, doy=time_index.dayofyear.values,
                   pressure=surface_pressure)
        dni = merge_rest_farms(rest_data.dni, dni, cloud_type)
    else:
        dni = merge_rest_farms(rest_data.dni, dni_farmsdni, cloud_type)

    # make a fill flag where bad data exists in the GHI irradiance
    fill_flag = make_fill_flag(ghi, rest_data.ghi, cloud_type, missing_props,
                               cloud_fill_flag=cloud_fill_flag)

    # Gap fill bad data in ghi and dni using the fill flag and cloud type
    ghi = gap_fill_irrad(ghi, rest_data.ghi, fill_flag, return_csr=False,
                         flags_to_fill=flags_to_fill)
    dni = gap_fill_irrad(dni, rest_data.dni, fill_flag, return_csr=False,
                         flags_to_fill=flags_to_fill)

    # enforce a max irradiance of clearsky irradiance
    dni, ghi, fill_flag = enforce_clearsky(dni, ghi, rest_data.dni,
                                           rest_data.ghi, solar_zenith_angle,
                                           fill_flag, sza_lim=SZA_LIM)

    # calculate the DHI, patching DNI for negative DHI values
    dhi, dni = calc_dhi(dni, ghi, solar_zenith_angle)

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

    if scale_outputs:
        output = scale_all_sky_outputs(output)

    return output


def all_sky_h5(f_source, rows=slice(None), cols=slice(None), disc_on=False):
    """Run all-sky from .h5 files.

    Parameters
    ----------
    f_source : str
        File path to source data file containing all sky inputs. Can be a
        single h5 file or MultiFileResource with format: /dir/prefix*suffix.h5
    rows : slice
        Subset of rows to run.
    cols : slice
        Subset of columns to run.
    disc_on : bool
        Compute cloudy sky dni with disc model or farms-dni model.

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

    if os.path.exists(f_source):
        Handler = Resource
    else:
        Handler = MultiFileResource

    with Handler(f_source) as source:
        all_sky_input = {dset: source[dset, rows, cols]
                         for dset in ALL_SKY_ARGS}
        all_sky_input['time_index'] = source.time_index[rows].tz_convert(None)
        all_sky_input['disc_on'] = disc_on

    try:
        out = all_sky(**all_sky_input)
    except Exception as e:
        logger.exception('All-Sky failed!')
        raise e

    return out


def all_sky_h5_parallel(f_source, rows=slice(None), cols=slice(None),
                        col_chunk=10, max_workers=None, disc_on=False):
    """Run all-sky from .h5 files.

    Parameters
    ----------
    f_source : str
        File path to source data file containing all sky inputs. Can be a
        single h5 file or MultiFileResource with format: /dir/prefix*suffix.h5
    rows : slice
        Subset of rows to run.
    cols : slice
        Subset of columns to run.
    col_chunk : int
        Number of columns to process on a single core. Larger col_chunk will
        increase the REST2 memory spike substantially, but will be
        significantly faster.
    max_workers : int | None
        Number of workers to run in parallel.
    disc_on : bool
        Compute cloudy sky dni with disc model or farms-dni model.

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

    if os.path.exists(f_source):
        Handler = Resource
    else:
        Handler = MultiFileResource

    with Handler(f_source) as res:
        data_shape = res.shape
        missing = [arg for arg in ALL_SKY_ARGS if arg not in res.dsets]
        if any(missing):
            msg = ('Cannot run all_sky, missing datasets {} from source: {}'
                   .format(missing, f_source))
            logger.error(msg)
            raise KeyError(msg)

    if rows.start is None:
        rows = slice(0, rows.stop)
    if rows.stop is None:
        rows = slice(rows.start, data_shape[0])

    if cols.start is None:
        cols = slice(0, cols.stop)
    if cols.stop is None:
        cols = slice(cols.start, data_shape[1])

    logger.info('Running all-sky for rows: {}'.format(rows))
    logger.info('Running all-sky for cols: {}'.format(cols))

    out_shape = (rows.stop - rows.start, cols.stop - cols.start)
    c_range = range(cols.start, cols.stop, col_chunk)
    c_slices_all = {}
    for c in c_range:
        c_slice = slice(c, np.min((c + col_chunk, cols.stop)))
        c_slices_all[c] = c_slice

    out = {}
    completed = 0

    logger.info('Running all-sky in parallel on {} workers.'
                .format(max_workers))

    loggers = ['farms', 'nsrdb', 'rest2', 'rex']
    with SpawnProcessPool(max_workers=max_workers, loggers=loggers) as exe:
        futures = {exe.submit(all_sky_h5, f_source, rows=rows, disc_on=disc_on,
                              cols=c_slices_all[c]): c for c in c_range}

        for future in as_completed(futures):

            c = futures[future]
            c_slice = c_slices_all[c]
            all_sky_out = future.result()

            for var, arr in all_sky_out.items():
                if var not in out:
                    logger.info('Initializing output array for "{}" with '
                                'shape {} and dtype {}.'
                                .format(var, out_shape, arr.dtype))
                    out[var] = np.ndarray(out_shape, dtype=arr.dtype)
                out[var][:, c_slice] = arr

            completed += 1
            mem = psutil.virtual_memory()

            if completed % 10 == 0:
                logger.info('All-sky futures completed: '
                            '{0} out of {1}. '
                            'Current memory usage is '
                            '{2:.3f} GB out of {3:.3f} GB total.'
                            .format(completed, len(futures),
                                    mem.used / 1e9, mem.total / 1e9))

    return out
