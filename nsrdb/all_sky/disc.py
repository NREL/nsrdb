# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 13:26:21 2014

@author: alopez, aweekley

These are functions adapted from PVL_Python.
There were four main changes from the original code
    1) functions were vectorized
    2) pvl_ephemeris expects UTC time
    3) removed unused result calculations
    4) Water and Pressure were changed to vectors from scalers
"""

import numpy as np
from nsrdb.all_sky import SOLAR_CONSTANT


def disc(ghi, sza, doy, pressure=101325, sza_lim=87):
    """Estimate DNI from GHI using the DISC model.

    *Warning: should only be used for cloudy FARMS data.

    The DISC algorithm converts global horizontal irradiance to direct
    normal irradiance through empirical relationships between the global
    and direct clearness indices.

    Parameters
    ----------
    ghi : np.ndarray
        Global horizontal irradiance in W/m2.
    sza : np.ndarray
        Solar zenith angle in degrees.
    doy : np.ndarray
        Day of year (array of integers).
    pressure : np.ndarray
        Pressure in Pascals.
    sza_lim : float | int
        Upper limit for solar zenith angle in degrees. SZA values greater than
        this will be truncated at this value. 87 deg chosen to simulate the
        FORTRAN code in use by SRRL (from Perez).

    Returns
    -------
    DNI : np.ndarray
        Estimated direct normal irradiance in W/m2.
    DHI : np.ndarray
        Estimated diffuse horizontal irradiance in W/m2.
    """

    A = np.zeros_like(ghi)
    B = np.zeros_like(ghi)
    C = np.zeros_like(ghi)

    day_angle = 2. * np.pi * (doy - 1) / 365

    re_var = (1.00011 + 0.034221 * np.cos(day_angle) +
              0.00128 * np.sin(day_angle) +
              0.000719 * np.cos(2. * day_angle) +
              7.7E-5 * np.sin(2. * day_angle))

    if len(re_var.shape) < len(sza.shape):
        re_var = np.tile(re_var.reshape((len(re_var), 1)), sza.shape[1])

    I0 = re_var * SOLAR_CONSTANT
    I0h = I0 * np.cos(np.radians(sza))
    Ztemp = np.copy(sza)
    Ztemp[Ztemp > sza_lim] = sza_lim

    AM = (1. / (np.cos(np.radians(Ztemp)) +
                0.15 * (np.power((93.885 - Ztemp), -1.253))) *
          pressure / 101325)

    Kt = ghi / I0h
    Kt[Kt < 0] = 0

    A[Kt > 0.6] = (-5.743 + 21.77 * Kt[Kt > 0.6] -
                   27.49 * np.power(Kt[Kt > 0.6], 2) +
                   11.56 * np.power(Kt[Kt > 0.6], 3))
    B[Kt > 0.6] = (41.4 - 118.5 * Kt[Kt > 0.6] +
                   66.05 * np.power(Kt[Kt > 0.6], 2) +
                   31.9 * np.power(Kt[Kt > 0.6], 3))
    C[Kt > 0.6] = (-47.01 + 184.2 * Kt[Kt > 0.6] -
                   222. * np.power(Kt[Kt > 0.6], 2) +
                   73.81 * np.power(Kt[Kt > 0.6], 3))

    A[Kt <= 0.6] = (0.512 - 1.56 * Kt[Kt <= 0.6] +
                    2.286 * np.power(Kt[Kt <= 0.6], 2) -
                    2.222 * np.power(Kt[Kt <= 0.6], 3))
    B[Kt <= 0.6] = 0.37 + 0.962 * Kt[Kt <= 0.6]
    C[Kt <= 0.6] = (-0.28 + 0.932 * Kt[Kt <= 0.6] -
                    2.048 * np.power(Kt[Kt <= 0.6], 2))

    delKn = A + B * np.exp(C * AM)

    Knc = (0.866 - 0.122 * AM + 0.0121 * np.power(AM, 2) -
           0.000653 * np.power(AM, 3) + 0.000014 * np.power(AM, 4))

    Kn = Knc - delKn
    DNI = (Kn) * I0

    DNI[np.logical_or.reduce((sza >= sza_lim, ghi < 1, DNI < 0))] = 0

    DHI = ghi - DNI * np.cos(np.radians(sza))

    if np.min(DHI) < 0:
        # patch for negative DHI values. Set DNI to zero, set DHI to GHI
        pos = np.where(DHI < 0)
        DNI[pos] = 0
        DHI[pos] = ghi[pos]

    return DNI, DHI
