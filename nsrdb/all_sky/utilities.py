"""Common utilities for NSRDB all-sky module.
"""

import pandas as pd
import numpy as np
from nsrdb.all_sky import RADIUS


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
        raise ValueError('Calculation of beta resulted in values outside of '
                         'expected range [0, 2.2]. Min/max of beta are: {}/{}'
                         .format(np.min(beta), np.max(beta)))
    return beta
